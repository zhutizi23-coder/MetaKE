import random
import json
import shutil
import inspect
from itertools import islice
from pathlib import Path
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    MQUAKEDataset,
    get_tfidf_vectorizer,
    KnownsDataset,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_mquake import compute_rewrite_quality_mquake
from memit import MEMITHyperParams
from memit.compute_z import get_module_input_output_at_words, compute_z
from memit.memit_valid1 import apply_memit_to_model, get_context_templates
from memit.memit_seq_main import apply_memit_seq_to_model
from memit.memit_rect_main import apply_memit_rect_to_model
from AlphaEdit import AlphaEditHyperParams
from AlphaEdit.MetaKE import apply_MetaKE_to_model, get_cov
from AlphaEdit.SPHERE import apply_SPHERE_to_model, get_cov
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
from Blue import BlueHyperParams
from Blue.Blue_main import apply_Blue_to_model, get_cov
from pmet import PMETHyperParams
from pmet.pmet_main import apply_pmet_to_model, get_cov
from pmet.pmet_seq_main import apply_pmet_seq_to_model, get_cov
from pmet.pmet_seq_valid import apply_pmet_seq_to_model, get_cov
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
from nse import NSEHyperParams 
from nse.nse_main import apply_nse_to_model
from glue_eval.glue_eval import GLUEEval
ALG_DICT = {
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "SPHERE": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "Blue": (BlueHyperParams, apply_Blue_to_model),
    "PMET": (PMETHyperParams, apply_pmet_to_model),
    "PMET_seq": (PMETHyperParams, apply_pmet_seq_to_model),
    "MetaKE": (AlphaEditHyperParams, apply_MetaKE_to_model),
    "MEMIT_seq": (MEMITHyperParams, apply_memit_seq_to_model),
    "MEMIT_prune": (MEMITHyperParams, apply_memit_to_model),
    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),
    "NSE": (NSEHyperParams, apply_nse_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset, compute_rewrite_quality_mquake),
}

DATASET_CONFIG = {
    "zsre": {
        "total": 19086,
        "save_steps": [100, 500, 1000, 5000],
    },
    "cf": {
        "total": 21919,
        "save_steps": [100, 500, 1000, 5000],
    },
    "mcf": {
        "total": 20877,
        "save_steps": [100, 500, 1000, 5000],
    },
}


def _filter_kwargs_for_callable(fn, kwargs):
    """Pass only kwargs accepted by callee to keep different algos compatible."""
    sig = inspect.signature(fn)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    beta_hse: float = 0.0,
    alpha: float = 0.0,
):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    if "MEMIT" in alg_name:
    # Get run hyperparameters
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / "MEMIT" / hparams_fname
        )
    elif "PMET" in alg_name:
    # Get run hyperparameters
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / "PMET" / hparams_fname
        )
    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        try:
            # Newer/custom checkpoints (e.g., qwen3) may require remote code registration.
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
            ).cuda()
            tok = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load model/tokenizer. If checkpoint model_type is custom/new (e.g., qwen3), "
                "please ensure trust_remote_code is enabled and your model folder has valid custom code/config. "
                f"Original error: {e}"
            ) from e
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"
    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    eval_ds = KnownsDataset(DATA_DIR)

    # Get cache templates
    cache_template = None
    if use_cache:
        if any(alg in alg_name for alg in ["MEMIT","AlphaEdit","MetaKE","Blue", "PMET", "MEMIT_seq", "SPHERE", "MEMIT_prune", "MEMIT_rect"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        elif any (alg in alg_name for alg in ["PMET", "PMET_seq"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}{''}"
                / f"{ds_name}_layer_{{}}_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")
    if alg_name == "NSE":
        cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        for record in ds:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.layers[-1], hparams.clamp_norm_factor, record["case_id"]
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
                continue
            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                context_templates = get_context_templates(model, tok)
                cur_z = compute_z(
                    model,
                    tok,
                    {"case_id": record["case_id"], **record["requested_rewrite"]},
                    hparams,
                    hparams.layers[-1],
                    context_templates,
                )
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
    if any(alg in alg_name for alg in ["AlphaEdit", "SPHERE", "Blue","MetaKE","MEMIT_seq", "MEMIT_prune", "NSE","PMET_seq"]):
        # Iterate through dataset
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if hparams.model_name == "gpt2-xl":
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            if alg_name in ["AlphaEdit", "SPHERE", "MetaKE","Blue"]:
                P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        elif hparams.model_name in ["EleutherAI_gpt-j-6B","Llama3-8B","phi-1.5","gemma-2b","Qwen3-4B-Thinking-2507"]:
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            if alg_name in ["AlphaEdit", "SPHERE", "MetaKE","Blue"]:
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
    if alg_name in ["AlphaEdit", "SPHERE", "MetaKE","Blue"]:
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model,tok,layer,hparams)
        torch.save(P, "null_space_project.pt")
    if alg_name != "FT":
        pre_edit_hs_path = Path(f"./hs/pre_edit_hs_{model_name.replace('/', '_')}.pt")
        hs = get_module_input_output_at_words(
                model,
                tok,
                hparams.layers[-1],
                context_templates=[request["template"] for request in eval_ds],
                words=[request["subject"] for request in eval_ds],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
        os.makedirs("./hs", exist_ok=True)
        torch.save(hs, pre_edit_hs_path)
        del hs
    glue_save_location = str(run_dir) + '/' + 'glue_eval/'
    os.makedirs(glue_save_location, exist_ok=True)
    cnt = 0
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        print(f"=================================================================={cnt+1}_edit==================================================================")
        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue
        
        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT","AlphaEdit", "Blue","SPHERE", "MetaKE","MEMIT_seq", "MEMIT_prune", "NSE","PMET","PMET_seq"]) else dict()
        seq_args = dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit", "SPHERE", "MetaKE", "MEMIT_seq", "NSE","PMET_seq"]) else dict()
        nc_args = dict(P = P) if any(alg in alg_name for alg in ["AlphaEdit", "SPHERE", "MetaKE"]) else dict()
        new_method_args = {
            "beta_hse": beta_hse,
            "alpha": alpha,
            "dataset": ds_name,
            "batch_size": num_edits,
            "DATASET_CONFIG": DATASET_CONFIG,
        }
        if cnt == 0 and args.downstream_eval_steps > 0:#do initial GLUE EVAL WITH ORIGINAL MODEL
            glue_results = {'edit_num': -1}

            out_file = glue_save_location + "base.json"
            
            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)

            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
        start = time()
        if any(alg in alg_name for alg in ["AlphaEdit", "SPHERE", "Blue","MetaKE", "MEMIT_seq", "NSE","PMET_seq"]):
            call_kwargs = {
                **args_conserve_memory,
                **etc_args,
                **seq_args,
                **nc_args,
                **new_method_args,
            }
            edited_model, cache_c = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                **_filter_kwargs_for_callable(apply_algo, call_kwargs),
            )
        elif alg_name == "MEMIT_prune":
            if cnt == 0:
                call_kwargs = {
                    "return_orig_weights": True,
                    **args_conserve_memory,
                    **etc_args,
                    **new_method_args,
                }
                edited_model, weights_copy = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                        **_filter_kwargs_for_callable(apply_algo, call_kwargs),
                )
                # Initialize the upd_matrix dictionary
                upd_matrix = {}
            else:
                call_kwargs = {
                    "return_orig_weights": False,
                    **args_conserve_memory,
                    **etc_args,
                    **new_method_args,
                }
                edited_model, _ = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                        **_filter_kwargs_for_callable(apply_algo, call_kwargs),
                )
            if cnt == (dataset_size_limit/num_edits) - 1:
            # Calculate the weight update matrix
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        current_weight = nethook.get_parameter(model, k)
                        upd_matrix[k] = current_weight - v.to("cuda")
                        # Calculate max singular value of the original weight
                        _, S_orig, _ = torch.svd(v)
                        max_sigma = S_orig.max().item()

                        # Adjust the upd_matrix singular values
                        U_upd, S_upd, V_upd = torch.svd(upd_matrix[k])
                        adjusted_S = torch.where(
                            S_upd > max_sigma,
                            torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device='cuda')) + max_sigma,
                            S_upd
                        )
                        upd_matrix[k] = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))

                # Apply the adjusted updates to the model
                with torch.no_grad():
                    for k in upd_matrix:
                        original_weight = nethook.get_parameter(model, k)
                        adjusted_weight = original_weight + upd_matrix[k]
                        original_weight.copy_(adjusted_weight)
        else:
            call_kwargs = {
                "return_orig_weights": False,
                **args_conserve_memory,
                **etc_args,
                **new_method_args,
            }
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                **_filter_kwargs_for_callable(apply_algo, call_kwargs),
            )
        exec_time = time() - start
        cnt+=1
        print("Execution took", exec_time)
        # Evaluate new model
    
        if args.downstream_eval_steps > 0 and cnt % args.downstream_eval_steps == 0:
            glue_results = {
                        'edit_num': cnt*num_edits,
                        'case_id': case_ids
                        }

            out_file = glue_save_location + "case_{}.json".format(record["case_id"])#stores the last case ID of the batch

            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    
            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
    if alg_name != "FT":
        post_edit_hs_path = Path(f"./hs/post_seq_{num_edits}_edit_hs_{alg_name}_{''}_{model_name.replace('/', '_')}_{ds_name}.pt")
        hs = get_module_input_output_at_words(
                edited_model,
                tok,
                hparams.layers[-1],
                context_templates=[request["template"] for request in eval_ds],
                words=[request["subject"] for request in eval_ds],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
        os.makedirs("./hs", exist_ok=True)
        torch.save(hs, post_edit_hs_path)
    start = time()
    gen_test_vars = [snips, vec]
    for record in ds:
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue
        metrics = {
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": ds_eval_method(
                edited_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
            ),
        }
        # Dump metrics in .json
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)

        # Restore original weights
        # with torch.no_grad():
        #     for k, v in weights_copy.items():
        #         nethook.get_parameter(model, k)[...] = v.to("cuda")

        print("Evaluation took", time() - start)
def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit", "SPHERE", "MetaKE", "Blue","MEMIT_rect", "MEMIT_seq", "MEMIT_prune", "MEMIT", "ROME", "FT", "MEND", "NSE","PMET","PMET_seq"],
        default="Pmet",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=False,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake"],
        default="zsre",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=2000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=2,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=100,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="If we want to do sequential editing or not",
    )
    parser.add_argument(
        "--beta_hse",
        type=float,
        default=0.0,
        help="Optional ratio for methods that use principal-direction suppression.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Optional projection strength for methods that use directional projection.",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        beta_hse=args.beta_hse,
        alpha=args.alpha,
    )
