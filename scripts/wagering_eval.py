#!/usr/bin/env python3
"""
Evaluation script for multi-LLM wagering methods.

Checks if wandb.run is active from training and continues it.

Usage: python wagering_eval.py <config_file.yaml>
"""

import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Ensure the local src/ tree and wagering package are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.utils import load_models_from_config, load_datasets_from_config, load_and_merge_configs
from wagering.utils.model_utils import should_load_prompt_perplexity_models_sequentially
from wagering.calibration import calibration_enabled, fit_or_load_logit_calibrator
from wagering.utils.multi_llm_ensemble import (
    assign_pubmedqa_context_models,
    get_cached_logits_and_hidden_states_for_model,
    get_model_prompt_variant,
)
from wagering.methods.factory import load_wagering_method
from wagering.inference import WageringEvaluator
from wagering.aggregation.factory import load_aggregation_function

log = logging.getLogger("wagering")


def load_api_keys_from_config():
    """Load API keys from .api_keys.yaml file if it exists."""
    api_keys_path = PROJECT_ROOT / ".api_keys.yaml"
    if not api_keys_path.exists():
        return {}
    
    with open(api_keys_path, "r") as f:
        config = yaml.safe_load(f)
        if not config:
            return {}
        
        filtered = {}
        for k, v in config.items():
            if v is None or v == "null" or v == "":
                continue
            if isinstance(v, str) and ("your-" in v.lower() and "-here" in v.lower()):
                continue
            filtered[k] = v
        return filtered


def main(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    calibration_path: Optional[str] = None,
):
    """Main evaluation function."""
    if config_path is None:
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            raise ValueError(
                "Config file path required. "
                "Usage: python wagering_eval.py <config_file.yaml>"
            )
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    args = load_and_merge_configs(config_path)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    # Suppress verbose library logging
    logging.getLogger("wagering").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Load API keys
    api_keys = load_api_keys_from_config()
    
    # Initialize wandb
    wandb_logger = None
    if args.get("report_to_wandb", False):
        try:
            import wandb
            
            wandb_api_key = api_keys.get("wandb_api_key")
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key
            
            # Check if wandb is already active from training
            if wandb.run is not None:
                log.debug(f"Continuing wandb run from training: {wandb.run.id}")
                wandb_logger = wandb
            else:
                log.info("Creating new wandb run for evaluation")
                wandb.init(
                    project=args.get("wandb_project", "multi-llm-wagering"),
                    entity=args.get("wandb_entity", None),
                    name=args.get("wandb_name", None),
                    tags=["evaluation"],
                )
                wandb_logger = wandb
            
            # Add evaluation tag
            if wandb.run and "evaluation" not in (wandb.run.tags or []):
                wandb.run.tags = list(wandb.run.tags or []) + ["evaluation"]
                    
        except ImportError as e:
            raise RuntimeError("wandb not available but report_to_wandb is enabled in config") from e
    
    logit_calibrator = None
    calibration_artifact_path = calibration_path
    if calibration_enabled(args):
        log.info("Preparing cached-logit temperature calibrator...")
        logit_calibrator, calibration_artifact_path, _ = fit_or_load_logit_calibrator(
            args,
            calibration_path=calibration_path,
        )
    
    # Load test datasets
    # Keep eval dataset composition stable across shuffle-seed sweeps.
    dataset_split_seed = int(args.get("dataset_split_seed", 42))
    sst = bool(args.get("shared_source_tripartition", False))
    tr_peer = args.get("datasets") or None
    test_datasets = []
    if "test_datasets" in args:
        test_ds, test_names = load_datasets_from_config(
            args["test_datasets"],
            split="test",
            random_seed=dataset_split_seed,
            shared_source_tripartition=sst,
            tripartition_peer_dataset_configs=tr_peer,
            infer_eval_split_train_without_peer=False,
            force_shared_source_tripartition=sst,
        )
        test_datasets = [(ds, name) for ds, name in zip(test_ds, test_names)]

    # Load OOD datasets (supports both plural and legacy singular keys)
    ood_datasets = []
    if "ood_datasets" in args and args["ood_datasets"]:
        ood_ds, ood_names = load_datasets_from_config(
            args["ood_datasets"],
            split="test",
            random_seed=dataset_split_seed,
            shared_source_tripartition=sst,
            tripartition_peer_dataset_configs=tr_peer,
            infer_eval_split_train_without_peer=False,
        )
        ood_datasets.extend((ds, name) for ds, name in zip(ood_ds, ood_names))
    elif "ood_dataset" in args and args["ood_dataset"]:
        ood_configs = [args["ood_dataset"]] if isinstance(args["ood_dataset"], dict) else args["ood_dataset"]
        ood_ds, ood_names = load_datasets_from_config(
            ood_configs,
            split="test",
            random_seed=dataset_split_seed,
            shared_source_tripartition=sst,
            tripartition_peer_dataset_configs=tr_peer,
            infer_eval_split_train_without_peer=False,
        )
        ood_datasets.extend((ds, name) for ds, name in zip(ood_ds, ood_names))

    # Configure PubMedQA mixed-context prompts with balanced randomized per-example assignment.
    eval_dataset_objects = [ds for ds, _ in test_datasets]
    eval_dataset_names = [name for _, name in test_datasets]
    if ood_datasets:
        eval_dataset_objects.extend(ds for ds, _ in ood_datasets)
        eval_dataset_names.extend(name for _, name in ood_datasets)
    pubmedqa_context_seed = dataset_split_seed
    pubmedqa_assignments = assign_pubmedqa_context_models(
        eval_dataset_objects,
        [model_cfg["path"] for model_cfg in args["models"]],
        random_seed=pubmedqa_context_seed,
    )
    for dataset_idx, assignment_info in pubmedqa_assignments.items():
        dataset_name = eval_dataset_names[dataset_idx] if dataset_idx < len(eval_dataset_names) else f"dataset_{dataset_idx}"
        assignment_hash = assignment_info.get("assignment_hash", "unknown")
        num_examples = assignment_info.get("num_examples", len(eval_dataset_objects[dataset_idx].x))
        routing_seed = assignment_info.get("routing_seed", pubmedqa_context_seed)
        model_context_counts = assignment_info.get("model_context_counts", [])
        log.info(
            "PubMedQA balanced mixed-context assignment for eval dataset %s: assignment_hash=%s, num_examples=%s, routing_seed=%s",
            dataset_name,
            assignment_hash,
            num_examples,
            routing_seed,
        )
        if isinstance(model_context_counts, list):
            for model_idx, context_count in enumerate(model_context_counts):
                model_path = args["models"][model_idx]["path"] if model_idx < len(args["models"]) else f"model_{model_idx}"
                log.info(
                    "PubMedQA context count for eval dataset %s: model_index=%d, model=%s, context_examples=%d",
                    dataset_name,
                    model_idx,
                    model_path,
                    int(context_count),
                )

    # Load wagering method (before models so we can decide cache requirements)
    wagering_config = args["wagering_method"]
    num_models = len(args["models"])
    wagering_method = load_wagering_method(
        wagering_config["name"],
        num_models=num_models,
        config=wagering_config.get("config", {}),
    )
    hidden_state_layers = getattr(
        wagering_method,
        "hidden_state_layers",
        wagering_config.get("config", {}).get("hidden_state_layers"),
    )
    # Keep cache checks and evaluator runtime on the same layer selection, even
    # for methods that don't explicitly expose hidden_state_layers.
    setattr(wagering_method, "hidden_state_layers", hidden_state_layers)

    requires_checkpoint = len(wagering_method.get_trainable_parameters()) > 0

    if checkpoint_path is None:
        checkpoint_path = args.get("checkpoint_path")
    if requires_checkpoint and checkpoint_path is None:
        log.error("Please provide a checkpoint path in config file")
        sys.exit(1)

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

    # Determine if hidden states are required
    wagering_method_name = type(wagering_method).__name__
    needs_hidden_states = (
        wagering_method_name not in ["EqualWagers", "ZeroOneWagers", "OneZeroWagers"]
        or logit_calibrator is not None
    )

    # Check cache per model across all eval datasets
    option_tokens = args.get("option_tokens", ["A", "B", "C", "D"])
    model_cfgs = args["models"]

    eval_datasets = [ds for ds, _ in test_datasets]
    if ood_datasets:
        eval_datasets.extend(ds for ds, _ in ood_datasets)

    cache_miss_indices = []
    cached_model_names = []
    cached_hidden_dims = {}
    for idx, model_cfg in enumerate(model_cfgs):
        model_path = model_cfg["path"]
        model_cached = True
        for ds in eval_datasets:
            prompt_variant = get_model_prompt_variant(ds, model_index=idx)
            cached_logits, cached_hidden_states, _ = get_cached_logits_and_hidden_states_for_model(
                model_path,
                ds,
                option_tokens,
                prompt_variant=prompt_variant,
                model_index=idx,
                hidden_state_layers=hidden_state_layers,
            )
            if cached_logits is None or (needs_hidden_states and cached_hidden_states is None):
                model_cached = False
                break
            if needs_hidden_states and cached_hidden_states is not None and idx not in cached_hidden_dims:
                try:
                    cached_hidden_dims[idx] = cached_hidden_states.shape[1]
                except Exception:
                    pass
        if not model_cached:
            cache_miss_indices.append(idx)
        cached_model_names.append(model_path.replace("/", "_"))

    force_load_all_models = bool(getattr(wagering_method, "requires_model_perplexities", False))
    if force_load_all_models:
        # Prompt-perplexity methods need live model objects even when logits are cached.
        log.info(
            "Wagering method requires model prompt perplexities; loading all %d models for evaluation.",
            num_models,
        )

    use_sequential_eval_perplexity = (
        force_load_all_models and should_load_prompt_perplexity_models_sequentially(num_models)
    )
    perplexity_cache_kwargs = (
        {"cache_dir": args.get("cache_path", "./workdir/cache")} if args.get("cache_path") else {}
    )

    models = []
    model_names = cached_model_names[:]
    load_model_indices = list(range(num_models)) if force_load_all_models else cache_miss_indices
    if use_sequential_eval_perplexity:
        log.info(
            "Deferring full ensemble load for prompt perplexity (%d models on %d visible CUDA device(s)); "
            "evaluator will load one model at a time.",
            num_models,
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        )
        models = [cfg["path"] for cfg in model_cfgs]
    elif load_model_indices:
        if force_load_all_models:
            log.info(
                "Loading %d/%d models to compute prompt perplexities.",
                len(load_model_indices),
                num_models,
            )
        else:
            log.info(f"Cache miss for {len(load_model_indices)}/{num_models} models. Loading missing models...")
        missing_cfgs = [model_cfgs[i] for i in load_model_indices]
        missing_models, missing_names = load_models_from_config(
            missing_cfgs,
            cache_kwargs=perplexity_cache_kwargs,
        )
        missing_name_map = {idx: name for idx, name in zip(load_model_indices, missing_names)}
        missing_iter = iter(missing_models)
        for i in range(num_models):
            if i in load_model_indices:
                models.append(next(missing_iter))
                model_names[i] = missing_name_map.get(i, model_names[i])
            else:
                models.append(model_cfgs[i]["path"])
    else:
        log.debug("All models are cached for evaluation. Skipping model loading.")
        models = [cfg["path"] for cfg in model_cfgs]

    log.debug(f"Prepared {len(models)} models: {model_names}")

    # Log hidden dimensions for debugging projection key mismatches
    if needs_hidden_states:
        log.debug("Model hidden dimensions:")
        for i in range(num_models):
            if i in cached_hidden_dims:
                hidden_dim = cached_hidden_dims[i]
                log.debug(f"  Model {i} ({model_names[i]}): hidden_dim={hidden_dim}, expected proj key=proj_{i}")
            elif i in cache_miss_indices:
                log.debug(f"  Model {i} ({model_names[i]}): hidden_dim=unknown (cache miss, will compute during eval)")
            elif force_load_all_models:
                log.debug(
                    f"  Model {i} ({model_names[i]}): hidden_dim=unknown (model loaded for prompt perplexity computation)"
                )
    
    # Load checkpoint
    # Check if the method has trainable parameters
    baseline_wagering_method = None
    
    if requires_checkpoint:
        log.info(f"\n{'='*80}")
        log.info("CHECKPOINT LOADING")
        log.info(f"{'='*80}")
        
        # Training returns the final checkpoint directory path directly
        # So we just need to load wagering_state.pt from that directory
        checkpoint_file = checkpoint_path / "final" / "wagering_state.pt"
        
        log.debug(f"Looking for checkpoint at: {checkpoint_file}")
        log.debug(f"Checkpoint exists: {checkpoint_file.exists()}")
        
        if not checkpoint_file.exists():
            log.error(f"✗ Checkpoint file not found: {checkpoint_file}")
            log.error(f"   Training should save the best checkpoint to this location.")
            log.error(f"   Received checkpoint_path: {checkpoint_path}")
            log.error(f"   Directory contents: {list(checkpoint_path.iterdir()) if checkpoint_path.exists() else 'directory does not exist'}")
            sys.exit(1)
        
        def _nested_tensor_sum(obj) -> float:
            """Sum all tensor leaves in deterministic key order using float64 on CPU.

            Avoids false checksum mismatches from (1) GPU vs CPU reduction order in
            tensor.sum() and (2) differing dict iteration order between checkpoint and
            loaded state_dict trees.
            """
            total = 0.0
            if isinstance(obj, dict):
                for k in sorted(obj.keys()):
                    total += _nested_tensor_sum(obj[k])
            elif torch.is_tensor(obj):
                total += float(obj.detach().cpu().double().sum().item())
            return total

        def _subset_tensor_sum(obj, keys) -> float:
            total = 0.0
            if isinstance(obj, dict):
                for k in sorted(keys):
                    if k in obj:
                        total += _nested_tensor_sum(obj[k])
            return total
        
        # Get state dict before loading for comparison
        state_before = wagering_method.state_dict()
        log.debug(f"State dict keys before loading: {list(state_before.keys())}")
        if state_before:
            first_key = list(state_before.keys())[0]
            if hasattr(state_before[first_key], 'sum'):
                log.debug(f"  Sample parameter '{first_key}' sum before: {state_before[first_key].sum():.6f}")
        
        log.debug(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        
        log.debug(f"Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            log.debug(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if isinstance(checkpoint, dict) and "wagering_method_state" in checkpoint:
            log.debug("Loading from checkpoint['wagering_method_state']")
            wagering_method.load_state_dict(checkpoint["wagering_method_state"])
            checkpoint_state = checkpoint["wagering_method_state"]
        else:
            log.debug("Loading checkpoint directly as state_dict")
            wagering_method.load_state_dict(checkpoint)
            checkpoint_state = checkpoint

        # Verify loaded state matches checkpoint (routers + projections)
        try:
            loaded_state = wagering_method.state_dict()
            ckpt_routers = checkpoint_state.get("routers_state_dict", {})
            ckpt_projs = checkpoint_state.get("model_projections_state_dict", {})
            loaded_routers = loaded_state.get("routers_state_dict", {})
            loaded_projs = loaded_state.get("model_projections_state_dict", {})

            ckpt_router_keys = set(ckpt_routers.keys())
            loaded_router_keys = set(loaded_routers.keys())
            ckpt_proj_keys = set(ckpt_projs.keys())
            loaded_proj_keys = set(loaded_projs.keys())

            missing_router_keys = ckpt_router_keys - loaded_router_keys
            extra_router_keys = loaded_router_keys - ckpt_router_keys
            missing_proj_keys = ckpt_proj_keys - loaded_proj_keys
            extra_proj_keys = loaded_proj_keys - ckpt_proj_keys

            if missing_router_keys or extra_router_keys or missing_proj_keys or extra_proj_keys:
                log.error("Checkpoint/model key mismatch detected:")
                if missing_router_keys:
                    log.error(f"  Missing router keys in model: {sorted(missing_router_keys)}")
                if extra_router_keys:
                    log.error(f"  Extra router keys in model: {sorted(extra_router_keys)}")
                if missing_proj_keys:
                    log.error(f"  Missing projection keys in model: {sorted(missing_proj_keys)}")
                    log.error(f"  This means the models loaded for evaluation have different hidden dimensions")
                    log.error(f"  than the models used during training.")
                    log.error(f"  Checkpoint expects: {sorted(ckpt_proj_keys)}")
                    log.error(f"  Current model has: {sorted(loaded_proj_keys)}")
                if extra_proj_keys:
                    log.error(f"  Extra projection keys in model: {sorted(extra_proj_keys)}")
                log.error("Refusing to evaluate with mismatched checkpoint/model state.")
                log.error("Solution: Ensure the same models are specified in the config for training and evaluation.")
                sys.exit(1)

            ckpt_sum = _subset_tensor_sum(ckpt_routers, ckpt_router_keys) + _subset_tensor_sum(ckpt_projs, ckpt_proj_keys)
            loaded_sum = _subset_tensor_sum(loaded_routers, ckpt_router_keys) + _subset_tensor_sum(loaded_projs, ckpt_proj_keys)
            log.debug(f"Checkpoint tensors sum (routers+projections): {ckpt_sum:.6f}")
            log.debug(f"Loaded tensors sum (routers+projections):    {loaded_sum:.6f}")
            if not torch.isclose(torch.tensor(ckpt_sum), torch.tensor(loaded_sum), rtol=1e-5, atol=1e-5):
                raise RuntimeError("Loaded state checksum does not match checkpoint checksum. This indicates corruption or incorrect checkpoint loading.")
            else:
                log.debug("✓ Loaded state checksum matches checkpoint")
        except Exception as e:
            raise Exception(f"Could not verify loaded state checksum: {e}")
        
        # Verify state changed
        state_after = wagering_method.state_dict()
        if state_before and state_after:
            first_key = list(state_after.keys())[0]
            if hasattr(state_after[first_key], 'sum'):
                log.info(f"  Sample parameter '{first_key}' sum after: {state_after[first_key].sum():.6f}")
                before_sum = state_before[first_key].sum()
                after_sum = state_after[first_key].sum()
                if torch.allclose(before_sum, after_sum):
                    raise Exception("⚠ WARNING: State dict appears unchanged after loading!")
                else:
                    log.info("✓ State dict successfully updated")
        
        log.debug(f"✓ Successfully loaded wagering method checkpoint")
        log.debug(f"{'='*80}\n")
    else:
        log.info("Wagering method has no trainable parameters - skipping checkpoint loading")

    # Load aggregation function
    aggregation_config = args["aggregation"]
    aggregation_function = load_aggregation_function(
        aggregation_config["name"],
        config=aggregation_config.get("config", {}),
    )
    
    # Set up evaluation checkpoint directory
    eval_checkpoint_dir = None
    if args.get("eval_checkpoint_dir"):
        eval_checkpoint_dir = Path(args["eval_checkpoint_dir"])
    elif checkpoint_path:
        eval_checkpoint_dir = checkpoint_path / "eval"
    else:
        eval_checkpoint_dir = Path("./eval_outputs")
    
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Evaluation outputs: {eval_checkpoint_dir}")
    
    # Prepare metadata
    metadata = {"model_names": model_names}
    training_datasets = args.get("training_datasets", [])
    if training_datasets:
        if isinstance(training_datasets, str):
            training_datasets = [training_datasets]
        elif isinstance(training_datasets, list) and training_datasets and isinstance(training_datasets[0], dict):
            training_datasets = [ds.get("name", ds.get("path", str(ds))) for ds in training_datasets]
        metadata["training_datasets"] = training_datasets
    
    seed = args.get("seed", args.get("shuffle_seed", None))

    env_prob_dbg = os.environ.get("WAGERING_DEBUG_PROB_ALIGN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    debug_batch_prob_alignment = bool(args.get("debug_batch_prob_alignment", False)) or env_prob_dbg
    
    # Get starting step from wandb if active
    wandb_starting_step = None
    if wandb_logger and wandb.run is not None:
        try:
            run_step = wandb.run.step
            if run_step is not None:
                wandb_starting_step = int(run_step)
                log.info(f"Continuing from wandb step {wandb_starting_step}")
            else:
                log.warning("wandb.run.step is None; evaluator will infer a safe starting step")
        except Exception as e:
            log.warning(f"Could not get wandb step: {e}")
    
    # Create evaluator
    evaluator = WageringEvaluator(
        models=models,
        wagering_method=wagering_method,
        aggregation_function=aggregation_function,
        option_tokens=args.get("option_tokens", ["A", "B", "C", "D"]),
        wandb_logger=wandb_logger,
        checkpoint_dir=eval_checkpoint_dir,
        metadata=metadata,
        training_checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        seed=seed,
        wandb_starting_step=wandb_starting_step,
        logit_calibrator=logit_calibrator,
        model_configs_for_sequential_perplexity=(
            model_cfgs if force_load_all_models else None
        ),
        perplexity_load_cache_kwargs=perplexity_cache_kwargs if perplexity_cache_kwargs else None,
        debug_batch_prob_alignment=debug_batch_prob_alignment,
    )
    
    # Evaluate
    log.info("Starting evaluation...")
    log.info(f"  Test datasets: {len(test_datasets)}")
    if ood_datasets:
        log.info(f"  OOD datasets: {len(ood_datasets)}")
        for _, ood_name in ood_datasets:
            log.info(f"    - {ood_name}")
    
    results = evaluator.evaluate_multiple(
        test_datasets=test_datasets,
        ood_datasets=ood_datasets,
        resume=False,  # Always evaluate from scratch
    )

    # Print results
    log.info("Evaluation Results:")
    results_summary = {}
    aggregate_metrics = {
        "accuracy": [],
        "nll": [],
        "brier": [],
        "auc": [],
        "ece": [],
        "inverse_hhi": [],
        "avg_inference_time_per_batch_s": [],
        "d_regret": [],
        "brier_d_regret": [],
        "meta_acc": [],
        "meta_nll": [],
        "meta_auc": [],
        "kendall_tau": [],
        "best_model_mrr": [],
        "bernoulli_kl": [],
        "bernoulli_tv": [],
    }
    
    for dataset_name, result in results.items():
        subset = result.get("subset_any_model_wrong") if isinstance(result, dict) else None
        def get_metric_str(result, key):
            val = result.get(key, None)
            if val is None:
                return "N/A"
            try:
                val_scalar = float(val) if not isinstance(val, (list, np.ndarray)) else float(val[0])
                return f"{val_scalar:.4f}" if not np.isnan(val_scalar) else "N/A"
            except (ValueError, TypeError):
                return "N/A"
        
        def get_metric_float(result, key):
            val = result.get(key, None)
            if val is None:
                return None
            try:
                val_scalar = float(val) if not isinstance(val, (list, np.ndarray)) else float(val[0])
                return None if np.isnan(val_scalar) else val_scalar
            except (ValueError, TypeError):
                return None

        accuracy_val = get_metric_float(result, "accuracy")
        nll_val = get_metric_float(result, "nll")
        brier_val = get_metric_float(result, "brier")
        auc_val = get_metric_float(result, "auc")
        ece_val = get_metric_float(result, "ece")
        inverse_hhi_val = get_metric_float(result, "inverse_hhi")
        avg_inference_time_val = get_metric_float(result, "avg_inference_time_per_batch_s")
        d_regret_val = get_metric_float(result, "d_regret")
        brier_d_regret_val = get_metric_float(result, "brier_d_regret")
        meta_acc_val = get_metric_float(result, "meta_acc")
        meta_nll_val = get_metric_float(result, "meta_nll")
        meta_auc_val = get_metric_float(result, "meta_auc")
        kendall_tau_val = get_metric_float(result, "kendall_tau")
        best_model_mrr_val = get_metric_float(result, "best_model_mrr")
        bernoulli_kl_val = get_metric_float(result, "bernoulli_kl")
        bernoulli_tv_val = get_metric_float(result, "bernoulli_tv")

        if accuracy_val is not None:
            aggregate_metrics["accuracy"].append(accuracy_val)
        if nll_val is not None:
            aggregate_metrics["nll"].append(nll_val)
        if brier_val is not None:
            aggregate_metrics["brier"].append(brier_val)
        if auc_val is not None:
            aggregate_metrics["auc"].append(auc_val)
        if ece_val is not None:
            aggregate_metrics["ece"].append(ece_val)
        if inverse_hhi_val is not None:
            aggregate_metrics["inverse_hhi"].append(inverse_hhi_val)
        if avg_inference_time_val is not None:
            aggregate_metrics["avg_inference_time_per_batch_s"].append(avg_inference_time_val)
        if d_regret_val is not None:
            aggregate_metrics["d_regret"].append(d_regret_val)
        if brier_d_regret_val is not None:
            aggregate_metrics["brier_d_regret"].append(brier_d_regret_val)
        if meta_acc_val is not None:
            aggregate_metrics["meta_acc"].append(meta_acc_val)
        if meta_nll_val is not None:
            aggregate_metrics["meta_nll"].append(meta_nll_val)
        if meta_auc_val is not None:
            aggregate_metrics["meta_auc"].append(meta_auc_val)
        if kendall_tau_val is not None:
            aggregate_metrics["kendall_tau"].append(kendall_tau_val)
        if best_model_mrr_val is not None:
            aggregate_metrics["best_model_mrr"].append(best_model_mrr_val)
        if bernoulli_kl_val is not None:
            aggregate_metrics["bernoulli_kl"].append(bernoulli_kl_val)
        if bernoulli_tv_val is not None:
            aggregate_metrics["bernoulli_tv"].append(bernoulli_tv_val)

        accuracy_str = get_metric_str(result, "accuracy")
        nll_str = get_metric_str(result, "nll")
        brier_str = get_metric_str(result, "brier")
        auc_str = get_metric_str(result, "auc")
        ece_str = get_metric_str(result, "ece")
        inverse_hhi_str = get_metric_str(result, "inverse_hhi")
        avg_inference_time_str = get_metric_str(result, "avg_inference_time_per_batch_s")
        d_regret_str = get_metric_str(result, "d_regret")
        brier_d_regret_str = get_metric_str(result, "brier_d_regret")
        meta_acc_str = get_metric_str(result, "meta_acc")
        meta_nll_str = get_metric_str(result, "meta_nll")
        meta_auc_str = get_metric_str(result, "meta_auc")
        kendall_tau_str = get_metric_str(result, "kendall_tau")
        best_model_mrr_str = get_metric_str(result, "best_model_mrr")
        bernoulli_kl_str = get_metric_str(result, "bernoulli_kl")
        bernoulli_tv_str = get_metric_str(result, "bernoulli_tv")
        
        log.info(
            f"{dataset_name}: Accuracy={accuracy_str}, "
            f"NLL={nll_str}, Brier={brier_str}, KL={bernoulli_kl_str}, TV={bernoulli_tv_str}, AUC={auc_str}, ECE={ece_str}, "
            f"InverseHHI={inverse_hhi_str}, AvgInferenceTimePerBatchS={avg_inference_time_str}, "
            f"DRegret={d_regret_str}, BrierDRegret={brier_d_regret_str}, MetaAcc={meta_acc_str}, MetaNLL={meta_nll_str}, MetaAUC={meta_auc_str}, "
            f"KendallTau={kendall_tau_str}, BestModelMRR={best_model_mrr_str}"
        )

        if isinstance(subset, dict) and int(subset.get("num_examples", 0) or 0) > 0:
            sub_n = int(subset.get("num_examples"))
            sub_acc = subset.get("accuracy", None)
            sub_nll = subset.get("nll", None)
            sub_ece = subset.get("ece", None)
            sub_dr = subset.get("d_regret", None)
            sub_kl = subset.get("bernoulli_kl", None)
            sub_tv = subset.get("bernoulli_tv", None)
            def _fmt(v):
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return "N/A"
            log.info(
                f"{dataset_name} [subset any-model-wrong n={sub_n}]: "
                f"Accuracy={_fmt(sub_acc)}, NLL={_fmt(sub_nll)}, KL={_fmt(sub_kl)}, TV={_fmt(sub_tv)}, "
                f"ECE={_fmt(sub_ece)}, DRegret={_fmt(sub_dr)}"
            )
        
        results_summary[dataset_name] = (
            f"Accuracy={accuracy_str}, AUC={auc_str}, ECE={ece_str}, "
            f"InverseHHI={inverse_hhi_str}, AvgInferenceTimePerBatchS={avg_inference_time_str}, "
            f"NLL={nll_str}, Brier={brier_str}, KL={bernoulli_kl_str}, TV={bernoulli_tv_str}, "
            f"DRegret={d_regret_str}, BrierDRegret={brier_d_regret_str}, "
            f"MetaAcc={meta_acc_str}, MetaAuc={meta_auc_str}, MetaNLL={meta_nll_str}, "
            f"KendallTau={kendall_tau_str}, BestModelMRR={best_model_mrr_str}"
        )

    def aggregate_metric_str(values):
        if not values:
            return "N/A"
        mean_val = float(np.mean(values))
        return f"{mean_val:.4f}" if not np.isnan(mean_val) else "N/A"

    overall_accuracy = aggregate_metric_str(aggregate_metrics["accuracy"])
    overall_nll = aggregate_metric_str(aggregate_metrics["nll"])
    overall_brier = aggregate_metric_str(aggregate_metrics["brier"])
    overall_auc = aggregate_metric_str(aggregate_metrics["auc"])
    overall_ece = aggregate_metric_str(aggregate_metrics["ece"])
    overall_inverse_hhi = aggregate_metric_str(aggregate_metrics["inverse_hhi"])
    overall_avg_inference_time_per_batch_s = aggregate_metric_str(aggregate_metrics["avg_inference_time_per_batch_s"])
    overall_d_regret = aggregate_metric_str(aggregate_metrics["d_regret"])
    overall_brier_d_regret = aggregate_metric_str(aggregate_metrics["brier_d_regret"])
    overall_meta_acc = aggregate_metric_str(aggregate_metrics["meta_acc"])
    overall_meta_nll = aggregate_metric_str(aggregate_metrics["meta_nll"])
    overall_meta_auc = aggregate_metric_str(aggregate_metrics["meta_auc"])
    overall_kendall_tau = aggregate_metric_str(aggregate_metrics["kendall_tau"])
    overall_best_model_mrr = aggregate_metric_str(aggregate_metrics["best_model_mrr"])
    overall_bernoulli_kl = aggregate_metric_str(aggregate_metrics["bernoulli_kl"])
    overall_bernoulli_tv = aggregate_metric_str(aggregate_metrics["bernoulli_tv"])

    results_summary["overall"] = (
        f"Accuracy={overall_accuracy}, AUC={overall_auc}, ECE={overall_ece}, "
        f"InverseHHI={overall_inverse_hhi}, AvgInferenceTimePerBatchS={overall_avg_inference_time_per_batch_s}, "
        f"NLL={overall_nll}, Brier={overall_brier}, KL={overall_bernoulli_kl}, TV={overall_bernoulli_tv}, "
        f"DRegret={overall_d_regret}, BrierDRegret={overall_brier_d_regret}, MetaAcc={overall_meta_acc}, "
        f"MetaAuc={overall_meta_auc}, MetaNLL={overall_meta_nll}, KendallTau={overall_kendall_tau}, BestModelMRR={overall_best_model_mrr}"
    )
    
    # Log summary to wandb
    if wandb_logger and wandb.run is not None:
        summary_df = pd.DataFrame([
            {"Dataset": name, "Metrics": metrics} 
            for name, metrics in results_summary.items()
        ])
        # Use commit=True to ensure table is logged independently without step tracking
        wandb.log({"evaluation/results_summary": wandb.Table(dataframe=summary_df)}, commit=True)
    
    # Finish wandb
    if wandb_logger and wandb.run is not None:
        log.info(f"Finishing wandb run: {wandb.run.id}")
        wandb.finish()
    
    return results


if __name__ == "__main__":
    main()
