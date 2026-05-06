import os
import csv
from collections import defaultdict
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

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memit_hparams import MEMITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
    validation2_dir: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_memit(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        validation2_dir=validation2_dir,
    )

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    return model, weights_copy


def _open_csv_append(path_str: str, header: List[str]):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    f = path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    return f, writer


def _get_memit_validation2_plan_config(
    hparams: MEMITHyperParams,
) -> Dict[str, Optional[float]]:
    """
    Validation-2 diagnoses the Static Planning-Control Trap using the *official*
    MEMIT target-planning routine. In MEMIT, the planning stage computes the
    target vector with a fixed optimization routine and applies an explicit norm
    clamp controlled by `clamp_norm_factor`.

    Following the appendix definition, we treat:
        c_plan := clamp_norm_factor

    as the primary global planning-control coefficient while keeping the rest of
    the pipeline fixed. Fields not used by MEMIT (e.g. kl_factor) are logged as
    empty values so that evaluate.py can consume the same schema for both
    AlphaEdit and MEMIT.
    """
    return {
        "c_plan_name": "clamp_norm_factor",
        "c_plan_value": float(getattr(hparams, "clamp_norm_factor", 0.0)),
        "kl_factor": None,
        "v_weight_decay": None,
        "clamp_norm_factor": float(getattr(hparams, "clamp_norm_factor", 0.0)),
    }


def _compute_memit_transfer_stats(
    *,
    K: torch.Tensor,          # [d_in, m]
    R: torch.Tensor,          # [d_out, m]
    cov: torch.Tensor,        # [d_in, d_in]
    mom2_update_weight: float,
):
    """
    Shared MEMIT transfer-stat computation used by theorem-1 logging and
    request-level Validation-2 diagnostics.

    Uses the practical MEMIT batch solver:
        Delta = R K^T (C_eff + K K^T)^(-1)
    so that:
        R_real = Delta K = R A_emp,
        A_emp  = K^T (C_eff + K K^T)^(-1) K.

    For the realizability diagnostic, we use the matrix-version theorem quantity
        G = K^T C_eff^(-1) K,
    with C_eff = mom2_update_weight * cov.

    Note:
        - Theorem-1 side remains tied to the downstream execution geometry.
        - Validation-2 reuses the same request-level realizability proxy gamma,
          while separately logging the *upstream* static planning control
          c_plan = clamp_norm_factor.
    """
    K = K.double()
    R = R.double()
    cov = cov.double()

    C_eff = mom2_update_weight * cov
    C_eff_actual = C_eff + K @ K.T

    # A_emp = K^T (C_eff + K K^T)^(-1) K
    adj_k = torch.linalg.solve(C_eff_actual, K)   # [d_in, m]
    A_emp = adj_k.T @ K                           # [m, m]

    # G = K^T C_eff^{-1} K
    CinvK = torch.linalg.solve(C_eff, K)
    G = K.T @ CinvK
    gamma_diag = torch.diag(G)

    diag_A = torch.diag(A_emp)
    offdiag_A = A_emp - torch.diag_embed(diag_A)

    R_real_emp = R @ A_emp
    planned_progress = torch.linalg.norm(R, dim=0)
    realized_progress = (R_real_emp * R).sum(dim=0) / (planned_progress + 1e-12)
    retention_ratio = realized_progress / (planned_progress + 1e-12)

    return {
        "A_emp": A_emp,
        "R_real_emp": R_real_emp,
        "gamma_diag": gamma_diag,
        "diag_A": diag_A,
        "offdiag_A": offdiag_A,
        "planned_progress": planned_progress,
        "realized_progress": realized_progress,
        "retention_ratio": retention_ratio,
    }


def _log_theorem1_matrix(
    *,
    layer: int,
    K: torch.Tensor,
    R: torch.Tensor,
    cov: torch.Tensor,
    mom2_update_weight: float,
    out_dir: str = "stats",
):
    out_path = str(Path(out_dir) / "memit_theorem1_matrix.csv")
    per_path = str(Path(out_dir) / "memit_theorem1_matrix_per_sample.csv")

    stats = _compute_memit_transfer_stats(
        K=K,
        R=R,
        cov=cov,
        mom2_update_weight=mom2_update_weight,
    )
    A_emp = stats["A_emp"]
    R_real_emp = stats["R_real_emp"]
    gamma_diag = stats["gamma_diag"]
    diag_A = stats["diag_A"]
    offdiag_A = stats["offdiag_A"]
    beta_theory = gamma_diag / (1.0 + gamma_diag)

    # Theory-side matrix for comparison
    K = K.double()
    cov = cov.double()
    C_eff = mom2_update_weight * cov
    CinvK = torch.linalg.solve(C_eff, K)
    G = K.T @ CinvK
    I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    A_theory = G @ torch.linalg.solve(I + G, I)
    R_real_theory = R.double() @ A_theory

    f, writer = _open_csv_append(
        out_path,
        [
            "layer",
            "fro_A_emp",
            "fro_A_theory",
            "fro_A_diff",
            "fro_Rreal_emp",
            "fro_Rreal_theory",
            "fro_Rreal_diff",
            "mean_diag_A",
            "mean_offdiag_abs_A",
            "mean_beta_theory",
            "mean_abs_beta_gap",
        ],
    )
    writer.writerow(
        [
            int(layer),
            torch.norm(A_emp, p="fro").item(),
            torch.norm(A_theory, p="fro").item(),
            torch.norm(A_emp - A_theory, p="fro").item(),
            torch.norm(R_real_emp, p="fro").item(),
            torch.norm(R_real_theory, p="fro").item(),
            torch.norm(R_real_emp - R_real_theory, p="fro").item(),
            diag_A.mean().item(),
            offdiag_A.abs().mean().item(),
            beta_theory.mean().item(),
            (diag_A - beta_theory).abs().mean().item(),
        ]
    )
    f.close()

    f2, writer2 = _open_csv_append(
        per_path,
        ["layer", "col", "gamma_diag", "beta_self", "beta_theory", "mixing_l1"],
    )
    for col in range(A_emp.shape[0]):
        writer2.writerow(
            [
                int(layer),
                int(col),
                gamma_diag[col].item(),
                A_emp[col, col].item(),
                beta_theory[col].item(),
                (A_emp[:, col].abs().sum() - abs(A_emp[col, col])).item(),
            ]
        )
    f2.close()


def _log_memit_validation2_request_layer(
    *,
    layer: int,
    request_case_ids: List[int],
    repeat_factor: int,
    stats: Dict[str, torch.Tensor],
    plan_cfg: Dict[str, Optional[float]],
    out_dir: str = "stats",
):
    path = str(Path(out_dir) / "memit_validation2_request_layer.csv")
    f, writer = _open_csv_append(
        path,
        [
            "layer",
            "case_id",
            "c_plan_name",
            "c_plan_value",
            "kl_factor",
            "v_weight_decay",
            "clamp_norm_factor",
            "gamma_layer",
            "planned_progress",
            "realized_progress",
            "retention_ratio",
            "beta_self",
            "mixing_l1",
        ],
    )

    A_emp = stats["A_emp"].detach().cpu()
    gamma_diag = stats["gamma_diag"].detach().cpu()
    planned_progress = stats["planned_progress"].detach().cpu()
    realized_progress = stats["realized_progress"].detach().cpu()
    retention_ratio = stats["retention_ratio"].detach().cpu()

    expanded_case_ids = []
    for case_id in request_case_ids:
        expanded_case_ids.extend([int(case_id)] * repeat_factor)

    if len(expanded_case_ids) != A_emp.shape[0]:
        raise ValueError(
            f"Expanded case ids ({len(expanded_case_ids)}) do not match column count ({A_emp.shape[0]})."
        )

    grouped = defaultdict(lambda: defaultdict(list))
    for col, case_id in enumerate(expanded_case_ids):
        grouped[case_id]["gamma_layer"].append(gamma_diag[col].item())
        grouped[case_id]["planned_progress"].append(planned_progress[col].item())
        grouped[case_id]["realized_progress"].append(realized_progress[col].item())
        grouped[case_id]["retention_ratio"].append(retention_ratio[col].item())
        grouped[case_id]["beta_self"].append(A_emp[col, col].item())
        grouped[case_id]["mixing_l1"].append((A_emp[:, col].abs().sum() - abs(A_emp[col, col])).item())

    for case_id, vals in grouped.items():
        writer.writerow(
            [
                int(layer),
                int(case_id),
                plan_cfg.get("c_plan_name"),
                plan_cfg.get("c_plan_value"),
                plan_cfg.get("kl_factor"),
                plan_cfg.get("v_weight_decay"),
                plan_cfg.get("clamp_norm_factor"),
                float(np.mean(vals["gamma_layer"])),
                float(np.mean(vals["planned_progress"])),
                float(np.mean(vals["realized_progress"])),
                float(np.mean(vals["retention_ratio"])),
                float(np.mean(vals["beta_self"])),
                float(np.mean(vals["mixing_l1"])),
            ]
        )
    f.close()


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
    validation2_dir: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer.
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    request_case_ids = [int(request["case_id"]) for request in requests]
    stats_dir = validation2_dir or "stats"
    plan_cfg = _get_memit_validation2_plan_config(hparams)

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        cache_fname = (
            Path(str(cache_template).format(z_layer, hparams.clamp_norm_factor, request["case_id"]))
            if cache_template is not None else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        if not data_loaded:
            cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
            z_list.append(cur_z)
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                print(f"Cached k/v pair at {cache_fname}")

    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

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

        force_recompute = False
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )

        layer_ks = layer_ks.double()
        targets = targets.double()
        cov = cov.double()

        C_eff_actual = hparams.mom2_update_weight * cov + layer_ks @ layer_ks.T
        adj_k = torch.linalg.solve(C_eff_actual, layer_ks)
        resid = targets / (len(hparams.layers) - i)
        upd_effect = resid @ adj_k.T
        upd_matrix = upd_effect

        try:
            shared_stats = _compute_memit_transfer_stats(
                K=layer_ks,
                R=resid,
                cov=cov,
                mom2_update_weight=hparams.mom2_update_weight,
            )
            _log_theorem1_matrix(
                layer=layer,
                K=layer_ks,
                R=resid,
                cov=cov,
                mom2_update_weight=hparams.mom2_update_weight,
                out_dir=stats_dir,
            )
            _log_memit_validation2_request_layer(
                layer=layer,
                request_case_ids=request_case_ids,
                repeat_factor=repeat_factor,
                stats=shared_stats,
                plan_cfg=plan_cfg,
                out_dir=stats_dir,
            )
        except Exception as e:
            print(f"[MEMIT validation logging] failed at layer {layer}: {e}")

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (adj_k.detach().cpu(), resid.detach().cpu())

        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            try:
                x.cpu()
            except Exception:
                pass
            del x
        torch.cuda.empty_cache()

    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas


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

    return torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        try:
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
        except RuntimeError as e:
            print(
                f"[WARNING] generate_fast failed in get_context_templates due to: {e}. "
                f"Falling back to simple '{{}}' context only."
            )
            CONTEXT_TEMPLATES_CACHE = [["{}"]]

    return CONTEXT_TEMPLATES_CACHE
