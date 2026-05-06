import os
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks, compute_ks_parallel
from .compute_zs import compute_zs, compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .pmet_hparams import PMETHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
KZ_CACHE = {}


def apply_pmet_seq_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
    cache_c=None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, preserves the original model while creating a new one to edit.
    :return: (1) the updated model, (2) updated cache_c
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    if hparams.blue:
        deltas, cache_c = execute_pmet_lbl(
            model, tok, requests, hparams, cache_template=cache_template, cache_c=cache_c
        )
    else:
        deltas, cache_c = execute_pmet(
            model, tok, requests, hparams, cache_template=cache_template, cache_c=cache_c
        )

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            upd_matrix = upd_matrix.to("cuda")
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"\nNew weights successfully inserted into {list(deltas.keys())}")
    return model, cache_c


def execute_pmet_lbl(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    cache_template: Optional[str] = None,
    cache_c=None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the PMET-LBL update algorithm.
    Invariant: model at beginning of function == model at end of function
    """
    deltas = {}

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    for request in requests[:10]:
        print(
            f"PMET request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    weights = {
        f"{rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
        for rewrite_module_tmp in hparams.rewrite_module_tmps
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    rewrite_module_names = hparams.rewrite_module_tmps

    # Validation-1 logging: per-edit beta / gamma for PMET
    log_f, log_writer = _open_beta_gamma_log(file_name="pmet_lbl_validation1.csv")

    context_templates = get_context_templates(model, tok)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        z_layer = layer
        z_list = {rewrite_module_name: [] for rewrite_module_name in rewrite_module_names}

        # Compute / load z* for each request
        for request in requests:
            for rewrite_module_name in rewrite_module_names:
                block_name = "attn" if "attn" in rewrite_module_name else "mlp"
                cache_fname = (
                    Path(
                        str(cache_template).format(
                            z_layer,
                            block_name,
                            hparams.clamp_norm_factor,
                            request["case_id"],
                        )
                    )
                    if cache_template is not None
                    else None
                )
                data_loaded = False

                if cache_fname is not None and cache_fname.exists():
                    try:
                        data = np.load(cache_fname)
                        z_list[rewrite_module_name].append(
                            torch.from_numpy(data["v_star"]).to("cuda")
                        )
                        data_loaded = True
                    except Exception as e:
                        print(f"Error reading cache file due to {e}. Recomputing...")

                if not data_loaded:
                    if len(rewrite_module_names) == 2:
                        cur_z_attn, cur_z_mlp = compute_zs(
                            model,
                            tok,
                            request,
                            hparams,
                            z_layer,
                            context_templates,
                        )
                        z_list[rewrite_module_names[0]].append(
                            cur_z_attn if "attn" in rewrite_module_names[0] else cur_z_mlp
                        )
                        z_list[rewrite_module_names[1]].append(
                            cur_z_attn if "attn" in rewrite_module_names[1] else cur_z_mlp
                        )

                        for rewrite_module_name_inner in rewrite_module_names:
                            block_name_inner = (
                                "attn" if "attn" in rewrite_module_name_inner else "mlp"
                            )
                            cache_fname_inner = (
                                Path(
                                    str(cache_template).format(
                                        z_layer,
                                        block_name_inner,
                                        hparams.clamp_norm_factor,
                                        request["case_id"],
                                    )
                                )
                                if cache_template is not None
                                else None
                            )
                            if cache_fname_inner is not None:
                                cache_fname_inner.parent.mkdir(exist_ok=True, parents=True)
                                np.savez(
                                    cache_fname_inner,
                                    **{
                                        "v_star": (
                                            cur_z_attn.detach().cpu().numpy()
                                            if block_name_inner == "attn"
                                            else cur_z_mlp.detach().cpu().numpy()
                                        )
                                    },
                                )
                                print(f"Cached k/v pair at {cache_fname_inner}")
                    else:
                        cur_z_attn, cur_z_mlp = compute_zs(
                            model,
                            tok,
                            request,
                            hparams,
                            z_layer,
                            context_templates,
                        )
                        cur_z = cur_z_attn if block_name == "attn" else cur_z_mlp
                        z_list[rewrite_module_name].append(cur_z)
                        if cache_fname is not None:
                            cache_fname.parent.mkdir(exist_ok=True, parents=True)
                            np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                            print(f"Cached k/v pair at {cache_fname}")
                    break

        for k, v in z_list.items():
            z_list[k] = torch.stack(v, dim=1)

        layers_ks = None
        for rewrite_module_name in rewrite_module_names:
            # Keys
            if "gpt-j" in model.config._name_or_path and len(rewrite_module_names) == 2:
                if layers_ks is None:
                    layers_ks = compute_ks_parallel(
                        model, tok, requests, hparams, layer, context_templates
                    )
            else:
                layers_ks = compute_ks(
                    model, tok, requests, hparams, rewrite_module_name, layer, context_templates
                )

            print(
                f"Writing {layers_ks[rewrite_module_name].size(0)} key/value pair(s) into layers"
            )

            # Current hidden states
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=rewrite_module_name,
                fact_token_strategy=hparams.fact_token,
            )[1].T

            targets = z_list[rewrite_module_name] - cur_zs

            try:
                layer_ks, targets = (
                    layers_ks[rewrite_module_name].T.double(),
                    targets.double(),
                )
            except Exception:
                layer_ks, targets = (
                    layers_ks[rewrite_module_name].T.double().to("cuda:1"),
                    targets.double().to("cuda:1"),
                )

            cov = get_cov(
                model,
                tok,
                rewrite_module_name.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples,
                hparams.mom2_dtype,
                force_recompute=False,
            )

            repeat_factor = layer_ks.size(1) // targets.size(1)
            targets = targets.repeat_interleave(repeat_factor, dim=1)

            # PMET-LBL uses full residual at each layer
            resid = targets

            solve_matrix = (
                cache_c[i, :, :].to(layer_ks.device).double()
                + layer_ks @ layer_ks.T
                + hparams.mom2_update_weight * cov.double()
            )

            # Raw update matrix in execution geometry
            upd_matrix_raw = resid @ layer_ks.T @ torch.inverse(solve_matrix)

            # Validation-1 logging using PMET effective execution geometry
            _log_beta_gamma_rows(
                log_writer=log_writer,
                layer=layer,
                rewrite_module_name=rewrite_module_name,
                layer_ks=layer_ks,
                resid=resid,
                upd_matrix=upd_matrix_raw,  # use raw matrix for clean Delta k
                lambda_exec=float(getattr(hparams, "mom2_update_weight", 0.0)),
                solve_matrix=solve_matrix,
            )

            weight_name = f"{rewrite_module_name.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix_raw, weights[weight_name].shape)

            print(weight_name, ":\norig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))

            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to("cuda:0")
                deltas[weight_name] = upd_matrix

            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(
            model, tok, requests, hparams, rewrite_module_names[0], layer, context_templates
        )[rewrite_module_names[0]].T
        cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

    # Restore original weights
    with torch.no_grad():
        for k, _ in weights.items():
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    log_f.close()

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas, cache_c


def execute_pmet(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    cache_template: Optional[str] = None,
    cache_c=None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the PMET update algorithm.
    Invariant: model at beginning of function == model at end of function
    """
    deltas = {}

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    for request in requests[:10]:
        print(
            f"PMET request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    weights = {
        f"{rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
        for rewrite_module_tmp in hparams.rewrite_module_tmps
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    rewrite_module_names = hparams.rewrite_module_tmps

    # Validation-1 logging: per-edit beta / gamma for PMET
    log_f, log_writer = _open_beta_gamma_log(file_name="pmet_validation1.csv")

    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = {rewrite_module_name: [] for rewrite_module_name in rewrite_module_names}

    # Compute / load z* for final layer
    for request in requests:
        for rewrite_module_name in rewrite_module_names:
            block_name = "attn" if "attn" in rewrite_module_name else "mlp"
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer,
                        block_name,
                        hparams.clamp_norm_factor,
                        request["case_id"],
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False

            if cache_fname is not None and cache_fname.exists():
                try:
                    data = np.load(cache_fname)
                    z_list[rewrite_module_name].append(torch.from_numpy(data["v_star"]).to("cuda"))
                    data_loaded = True
                except Exception as e:
                    print(f"Error reading cache file due to {e}. Recomputing...")

            if not data_loaded:
                if len(rewrite_module_names) == 2:
                    cur_z_attn, cur_z_mlp = compute_zs(
                        model,
                        tok,
                        request,
                        hparams,
                        z_layer,
                        context_templates,
                    )
                    z_list[rewrite_module_names[0]].append(
                        cur_z_attn if "attn" in rewrite_module_names[0] else cur_z_mlp
                    )
                    z_list[rewrite_module_names[1]].append(
                        cur_z_attn if "attn" in rewrite_module_names[1] else cur_z_mlp
                    )

                    for rewrite_module_name_inner in rewrite_module_names:
                        block_name_inner = (
                            "attn" if "attn" in rewrite_module_name_inner else "mlp"
                        )
                        cache_fname_inner = (
                            Path(
                                str(cache_template).format(
                                    z_layer,
                                    block_name_inner,
                                    hparams.clamp_norm_factor,
                                    request["case_id"],
                                )
                            )
                            if cache_template is not None
                            else None
                        )
                        if cache_fname_inner is not None:
                            cache_fname_inner.parent.mkdir(exist_ok=True, parents=True)
                            np.savez(
                                cache_fname_inner,
                                **{
                                    "v_star": (
                                        cur_z_attn.detach().cpu().numpy()
                                        if block_name_inner == "attn"
                                        else cur_z_mlp.detach().cpu().numpy()
                                    )
                                },
                            )
                            print(f"Cached k/v pair at {cache_fname_inner}")
                else:
                    cur_z_attn, cur_z_mlp = compute_zs(
                        model,
                        tok,
                        request,
                        hparams,
                        z_layer,
                        context_templates,
                    )
                    cur_z = cur_z_attn if block_name == "attn" else cur_z_mlp
                    z_list[rewrite_module_name].append(cur_z)
                    if cache_fname is not None:
                        cache_fname.parent.mkdir(exist_ok=True, parents=True)
                        np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                        print(f"Cached k/v pair at {cache_fname}")
                break

    for k, v in z_list.items():
        z_list[k] = torch.stack(v, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        layers_ks = None

        for rewrite_module_name in rewrite_module_names:
            if "gpt-j" in model.config._name_or_path and len(rewrite_module_names) == 2:
                if layers_ks is None:
                    layers_ks = compute_ks_parallel(
                        model, tok, requests, hparams, layer, context_templates
                    )
            else:
                layers_ks = compute_ks(
                    model, tok, requests, hparams, rewrite_module_name, layer, context_templates
                )

            print(
                f"Writing {layers_ks[rewrite_module_name].size(0)} key/value pair(s) into layers"
            )

            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=rewrite_module_name,
                fact_token_strategy=hparams.fact_token,
            )[1].T

            targets = z_list[rewrite_module_name] - cur_zs

            try:
                layer_ks, targets = (
                    layers_ks[rewrite_module_name].T.double(),
                    targets.double(),
                )
            except Exception:
                layer_ks, targets = (
                    layers_ks[rewrite_module_name].T.double().to("cuda:1"),
                    targets.double().to("cuda:1"),
                )

            cov = get_cov(
                model,
                tok,
                rewrite_module_name.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples,
                hparams.mom2_dtype,
                force_recompute=False,
            )

            repeat_factor = layer_ks.size(1) // targets.size(1)
            targets = targets.repeat_interleave(repeat_factor, dim=1)

            # PMET square-root spread residual
            resid = targets / np.sqrt(len(hparams.layers) - i)

            solve_matrix = (
                cache_c[i, :, :].to(layer_ks.device).double()
                + layer_ks @ layer_ks.T
                + hparams.mom2_update_weight * cov.double()
            )

            # Raw update matrix in execution geometry
            upd_matrix_raw = resid @ layer_ks.T @ torch.inverse(solve_matrix)

            # Validation-1 logging using PMET effective execution geometry
            _log_beta_gamma_rows(
                log_writer=log_writer,
                layer=layer,
                rewrite_module_name=rewrite_module_name,
                layer_ks=layer_ks,
                resid=resid,
                upd_matrix=upd_matrix_raw,  # use raw matrix for clean Delta k
                lambda_exec=float(getattr(hparams, "mom2_update_weight", 0.0)),
                solve_matrix=solve_matrix,
            )

            weight_name = f"{rewrite_module_name.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix_raw, weights[weight_name].shape)

            print(weight_name, ":\norig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))

            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to("cuda:0")
                deltas[weight_name] = upd_matrix

            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(
            model, tok, requests, hparams, rewrite_module_names[0], layer, context_templates
        )[rewrite_module_names[0]].T
        cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

    # Restore original weights
    with torch.no_grad():
        for k, _ in weights.items():
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    log_f.close()

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas, cache_c


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
            "Update matrix computed by PMET does not match original weight shape. "
            "Check for bugs in the code?"
        )


def _open_beta_gamma_log(file_name: str):
    log_dir = Path("stats")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / file_name
    write_header = not log_path.exists()
    log_f = log_path.open("a", newline="")
    log_writer = csv.writer(log_f)
    if write_header:
        log_writer.writerow([
            "layer",
            "module",
            "edit_idx",
            "beta",
            "gamma",
            "k_norm",
            "gamma_sqrt",
            "lambda_exec",
        ])
    return log_f, log_writer


def _log_beta_gamma_rows(
    log_writer,
    layer: int,
    rewrite_module_name: str,
    layer_ks: torch.Tensor,
    resid: torch.Tensor,
    upd_matrix: torch.Tensor,   # raw update matrix
    lambda_exec: float,
    solve_matrix: torch.Tensor,
):
    """
    PMET Validation 1 diagnostics:
      beta  = ||Delta k|| / ||d_sem||
      gamma = k^T A^{-1} k
    where A is the PMET effective execution geometry:
      A = cache_c + K K^T + lambda_exec * Cov
    """
    eps = 1e-8

    try:
        solve_inv = torch.inverse(solve_matrix)
    except Exception:
        solve_inv = torch.linalg.pinv(solve_matrix)

    for edit_idx in range(layer_ks.shape[1]):
        k = layer_ks[:, edit_idx]     # [d_in]
        d_sem = resid[:, edit_idx]    # [d_out]

        # realized residual: delta_real = Delta k
        if upd_matrix.shape[0] == d_sem.shape[0] and upd_matrix.shape[1] == k.shape[0]:
            delta_real = upd_matrix @ k
        elif upd_matrix.shape[1] == d_sem.shape[0] and upd_matrix.shape[0] == k.shape[0]:
            delta_real = upd_matrix.T @ k
        else:
            print(
                f"[PMET diagnostics] skip incompatible shape: "
                f"upd_matrix={tuple(upd_matrix.shape)}, "
                f"d_sem={tuple(d_sem.shape)}, k={tuple(k.shape)}"
            )
            continue

        beta = float(delta_real.norm() / (d_sem.norm() + eps))
        gamma = float(torch.dot(k, solve_inv @ k))
        gamma_sqrt = float(np.sqrt(max(gamma, 0.0)))

        log_writer.writerow([
            int(layer),
            rewrite_module_name,
            int(edit_idx),
            beta,
            gamma,
            float(k.norm()),
            gamma_sqrt,
            float(lambda_exec) + eps,
        ])


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

    try:
        return torch.inverse(COV_CACHE[key].to("cuda:1")) if inv else COV_CACHE[key].to("cuda:1")
    except Exception:
        return torch.inverse(COV_CACHE[key].to("cuda:0")) if inv else COV_CACHE[key].to("cuda:0")


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
            for length, n_gen in [(10, 5)]
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE