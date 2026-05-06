import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
import torch.nn.functional as F

def make_refine_v_star_alphaedit_solve_proxy(
    model,
    tok,
    request,
    hparams,
    z_layer,
    context_templates,
    v_init,                 # [d_out]
    cache_c,                # [n_layers, d_in, d_in] on CPU ok
    P,                      # [n_layers, d_in, d_in] on CPU ok
):
    device = next(model.parameters()).device
    nethook.set_requires_grad(False, model)

    ablation_mode = getattr(hparams, "make_ablation_mode", "none")
    legacy_mode = ablation_mode == "legacy"

   # ---- hyperparams ----
    T = getattr(hparams, "make_num_steps", 15)
    eta = getattr(hparams, "make_lr", 5e-3)
    meta_reg = getattr(hparams, "make_meta_reg",1e-2)
    kl_fac = getattr(hparams, "make_kl_factor", 0.05)
    verbose = getattr(hparams, "make_verbose", True)
    use_clamp = getattr(hparams, "make_use_clamp", True) and not legacy_mode
    # 注意：这里默认关闭 align_cur_z_context，避免错误地把 subject 从模板中“丢掉”
    # 如果需要做消融或进一步对齐，可以在 hparams 里显式加 make_align_cur_z_context=True
    align_cur_z_context = getattr(hparams, "make_align_cur_z_context", False) and not legacy_mode

    if verbose and getattr(hparams, "make_print_config", True):
        print(
            f"align_cur_z_context={align_cur_z_context} use_clamp={use_clamp} "
            f"T={T} lr={eta}  kl={kl_fac}"
        )

    # ---- (对齐 compute_z) finalize rewrite & loss layers ----
    # compute_z: loss_layer = max(hparams.v_loss_layer, layer)
    loss_layer = max(getattr(hparams, "v_loss_layer", z_layer), z_layer)

    # ---- (对齐 compute_z) get lm_head & ln_f ----
    lm_w = nethook.get_module(model, f"{hparams.lm_head_module}").weight.T
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    # ---- build prompts/targets (对齐 compute_z) ----
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to(device)["input_ids"][0]
    if target_ids.numel() > 0 and (target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id):
        target_ids = target_ids[1:]

    rewriting_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ]
    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [p.format(request["subject"]) for p in all_prompts],
        return_tensors="pt",
        padding=True
    ).to(device)

    # ---- (对齐 compute_z) rewriting_targets shape: [len(rewriting_prompts), seq] ----
    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids): ex_len] = target_ids

    # ---- lookup idxs (同 compute_z) ----
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token,
            verbose=False
        )
        for prompt in all_prompts
    ]
    kl_lookup_idxs = lookup_idxs[-len(kl_prompts):]

    # ---- (对齐 compute_z) baseline KL distribution (init) ----
    # compute_z 的 kl_distr_init 是第一次迭代（delta=0）下的 KL log_probs
    # 这里我们用“无 hook”的 forward 来得到 init 分布，含义等价（仍是干净模型分布）
    with torch.no_grad():
        logits0 = model(**input_tok).logits
        kl_logits0 = torch.stack(
            [logits0[len(rewriting_prompts) + j, idx, :] for j, idx in enumerate(kl_lookup_idxs)],
            dim=0,
        )
        kl_distr_init = F.log_softmax(kl_logits0, dim=1).detach()

    # ---- proxy constants from AlphaEdit LAST layer (K,C,P) ----
    K = compute_ks(model, tok, [request], hparams, z_layer, context_templates).T.to(device).detach()  # [d_in, n_k]
    d_in, n_k = K.shape

    if align_cur_z_context:
        cur_z_context_templates = [
            context.format(request["prompt"])
            for context_types in context_templates
            for context in context_types
        ]
        cur_z_words = [request["subject"] for _ in cur_z_context_templates]
    else:
        cur_z_context_templates = [request["prompt"]]
        cur_z_words = [request["subject"]]

    with torch.no_grad():
        cur_z_all = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=cur_z_context_templates,
            words=cur_z_words,
            module_template=hparams.layer_module_tmp, 
            fact_token_strategy=hparams.fact_token,
        )[1].to(device)
        if cur_z_all.ndim == 1:
            cur_z = cur_z_all.view(-1, 1)
        else:
            cur_z = cur_z_all.mean(dim=0, keepdim=True).T
    d_out = cur_z.shape[0]

    z_i = len(hparams.layers) - 1
    C = cache_c[z_i].to(device).float()
    Pz = P[z_i].to(device).float()
    lam = float(getattr(hparams, "L2", 0.0))

    KKt = K @ K.T
    A = (Pz @ (KKt + C) + lam * torch.eye(d_in, device=device, dtype=torch.float32)).detach()
    PK = (Pz @ K).detach()

    # ---- locate rewrite module ----
    rewrite_mod_name = hparams.rewrite_module_tmp.format(z_layer)
    rewrite_mod = nethook.get_module(model, rewrite_mod_name)
    W = nethook.get_parameter(model, f"{rewrite_mod_name}.weight")

    # ---- optimize v (MAKE 不变) ----
    v0 = v_init.detach().to(device)
    v = v0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=eta)

    for t in range(T):
        opt.zero_grad(set_to_none=True)

        # ---- solve proxy ΔW from current v (MAKE 不变) ----
        targets = (v.view(d_out, 1) - cur_z)  # [d_out, 1]
        repeat_factor = n_k // targets.shape[1]  # targets.shape[1] == 1
        resid = targets.repeat_interleave(repeat_factor, dim=1)  # [d_out, n_k]

        rhs = PK @ resid.T                      # [d_in, d_out]
        upd = torch.linalg.solve(A, rhs)        # [d_in, d_out]

        if W.shape == (d_out, d_in):
            delta_weight = upd.T                # [d_out, d_in]
        elif W.shape == (d_in, d_out):
            delta_weight = upd                  # [d_in, d_out]
        else:
            raise ValueError(f"Unexpected W shape {W.shape}")

        def _linear_delta(x, delta_w):
            # 强烈建议：用 W 的形状来决定方向，而不是猜测
            # 但这里先保持你原逻辑 + debug
            D = x.shape[-1]
            if delta_w.shape[1] == D:
                return F.linear(x, delta_w, bias=None)  # x @ delta_w.T
            if delta_w.shape[0] == D:
                return torch.einsum("...d,do->...o", x, delta_w)  # x @ delta_w
            raise ValueError(f"delta_w shape {delta_w.shape} not compatible with x last dim {D}")

        def _hook(module, inputs, output):
            x = inputs[0]
            y = output[0] if isinstance(output, tuple) else output
            if x.dim() != 3 or y.dim() != 3:
                return output
            dy = _linear_delta(x, delta_weight)

            
            y_new = y + dy
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new

        # ---- forward with hook + trace loss_layer output (对齐 compute_z 的 NLL 路径) ----
        with nethook.TraceDict(
            module=model,
            layers=[hparams.layer_module_tmp.format(loss_layer)],
            retain_input=False,
            retain_output=True,
        ) as tr:
            h = rewrite_mod.register_forward_hook(_hook)
            logits = model(**input_tok).logits
            h.remove()

            # ---- KL current (仍从最终 logits 取，和 compute_z 一致的“lookup idx”定义) ----
            kl_logits = torch.stack(
                [logits[len(rewriting_prompts) + j, idx, :] for j, idx in enumerate(kl_lookup_idxs)],
                dim=0,
            )
            kl_log_probs = F.log_softmax(kl_logits, dim=1)

        # ---- (对齐 compute_z) NLL: use full_repr at loss_layer -> ln_f -> lm_head ----
        output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]

        # compute_z 里有一个 transpose 兼容处理，这里原样对齐
        if output.shape[1] != rewriting_targets.shape[1]:
            output = torch.transpose(output, 0, 1)

        full_repr = output[:len(rewriting_prompts)]  # only rewriting prompts

        vocab_logits = ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device)
        log_probs = torch.log_softmax(vocab_logits, dim=2)

        token_logp = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0)
             .unsqueeze(2)
             .to(log_probs.device),
        ).squeeze(2)

        mask = (rewriting_targets != -100).float()
        nll_loss_each = -(token_logp * mask.to(token_logp.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()

        # ---- (对齐 compute_z) KL direction ----
        # compute_z: kl_div(kl_distr_init, kl_log_probs, log_target=True)  ==> KL(current || init)
        kl_loss = kl_fac * F.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )

        # ---- 正则项：模仿 compute_z 中的归一化 L2 正则 ----
        # compute_z: weight_decay = v_weight_decay * (||delta|| / ||target_init||^2)
        # 这里用 v-v0 作为 "delta"，v0 作为 "target_init" 的对应量
        delta_v = v - v0
        v0_norm = v0.norm()
        # 与 compute_z 一致：比例是 ||delta|| / ||v0||^2（不再额外平方）
        reg = meta_reg * (delta_v.norm() / (v0_norm.pow(2) + 1e-8))

        loss = nll_loss + kl_loss.to(nll_loss.device) + reg.to(nll_loss.device)

        loss.backward()
        opt.step()

        if use_clamp:
            max_norm = hparams.clamp_norm_factor * v0.norm()
            delta_v_now = v - v0
            delta_v_norm = delta_v_now.norm()
            if delta_v_norm > max_norm:
                with torch.no_grad():
                    v[...] = v0 + delta_v_now * (max_norm / (delta_v_norm + 1e-12))

        if verbose and (t == 0 or t == T - 1 or (t + 1) % 5 == 0):
            print(
                f"[MAKE] step {t+1}/{T} | loss={loss.item():.4f} "
                f"(nll={nll_loss.item():.4f}, kl={kl_loss.item():.4f}) "
                f"avg prob [{request['target_new']['str']}]={torch.exp(-nll_loss_each).mean().item():.4f}"
            )

    return v.detach()

def apply_MetaKE_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)  # [d_out, n_req]

    if getattr(hparams, "use_make", False):
        print(f"[MAKE] enabled: refining v* for {len(requests)} requests")
        zs_new = []
        for j, req in enumerate(requests):
            if j % 10 == 0:
                print(f"[MAKE] processing request {j+1}/{len(requests)}")
            v_opt = make_refine_v_star_alphaedit_solve_proxy(
                model, tok, req, hparams,
                z_layer=hparams.layers[-1],
                context_templates=context_templates,
                v_init=zs[:, j],
                cache_c=cache_c,
                P=P,
            )
            zs_new.append(v_opt)
        zs = torch.stack(zs_new, dim=1)
        print("[MAKE] refinement finished")
    else:
        print("[MAKE] skipped: hparams.use_make is False")

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = torch.linalg.solve(
                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
        )
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )

def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE