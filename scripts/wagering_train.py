#!/usr/bin/env python3
"""
Training script for multi-LLM wagering methods.

Does NOT call wandb.finish() to keep the run active for evaluation.

Usage: python wagering_train.py <config_file.yaml>
"""

import logging
import os
import sys
import yaml
import torch
from pathlib import Path
from typing import List, Optional

# Ensure the local src/ tree and wagering package are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.utils import (
    load_models_from_config,
    load_datasets_from_config,
    load_and_merge_configs,
    generate_checkpoint_dir,
    get_checkpoint_metadata,
)
from wagering.utils.model_utils import should_load_prompt_perplexity_models_sequentially
from wagering.calibration import calibration_enabled, fit_or_load_logit_calibrator
from wagering.utils.multi_llm_ensemble import (
    assign_pubmedqa_context_models,
    get_cached_logits_and_hidden_states_for_model,
    get_model_prompt_variant,
    resolve_hidden_state_layers_for_model,
)
from wagering.core.dataset import Dataset
from wagering.methods.factory import load_wagering_method
from wagering.training import WageringTrainer
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


def main(config_path: Optional[str] = None, calibration_path: Optional[str] = None):
    """Main training function."""
    if config_path is None:
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            raise ValueError(
                "Config file path required. "
                "Usage: python wagering_train.py <config_file.yaml>"
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
    
    # Generate unique checkpoint directory
    base_checkpoint_dir = Path(args.get("checkpoint_base_dir", "/common/users/yl2310/MultiLLMs/checkpoints"))
    calibration_config = args.get("calibration") if calibration_enabled(args) else None
    checkpoint_dir = generate_checkpoint_dir(
        base_dir=base_checkpoint_dir,
        models=args["models"],
        datasets=args["datasets"],
        wagering_method=args["wagering_method"],
        aggregation=args["aggregation"],
        create_hash=True,
        calibration=calibration_config,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Get metadata
    checkpoint_metadata = get_checkpoint_metadata(
        models=args["models"],
        datasets=args["datasets"],
        wagering_method=args["wagering_method"],
        aggregation=args["aggregation"],
        calibration=calibration_config,
    )
    
    # Initialize wandb
    wandb_logger = None
    if args.get("report_to_wandb", False):
        try:
            import wandb
            api_keys = load_api_keys_from_config()
            wandb_api_key = api_keys.get("wandb_api_key")
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key
            
            wandb_config = args.copy()
            wandb_config.update(checkpoint_metadata)
            
            wandb.init(
                project=args.get("wandb_project", "multi-llm-wagering"),
                entity=args.get("wandb_entity", None),
                name=args.get("wandb_name", None),
                config=wandb_config,
                tags=[
                    f"wagering_{args['wagering_method']['name']}",
                    f"agg_{args['aggregation']['name']}",
                    f"models_{len(args['models'])}",
                    f"datasets_{len(args['datasets'])}",
                    "training",
                ],
            )
            wandb_logger = wandb
            log.info(f"Initialized wandb run: {wandb.run.id}")
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
    
    # Load training datasets
    log.info("Loading training datasets...")
    # Keep dataset membership/splits stable across shuffle-seed sweeps.
    # `shuffle_seed` is only for in-training shuffling after cache collection.
    dataset_split_seed = int(args.get("dataset_split_seed", 42))
    train_datasets, dataset_names = load_datasets_from_config(
        args["datasets"],
        split="train",
        random_seed=dataset_split_seed,
        shared_source_tripartition=bool(args.get("shared_source_tripartition", False)),
        tripartition_peer_dataset_configs=args.get("test_datasets") or None,
    )
    log.info(f"Loaded {len(train_datasets)} training datasets: {dataset_names}")

    validation_split_ratio = args.get("validation_split_ratio", 0.1)
    derived_val_split_ratio = None
    for dataset in train_datasets:
        ratio = getattr(dataset, "pubmedqa_train_val_split_ratio", None)
        if ratio is not None:
            derived_val_split_ratio = float(ratio)
            break
        ratio = getattr(dataset, "source_tripartition_val_ratio", None)
        if ratio is not None and len(train_datasets) == 1:
            derived_val_split_ratio = float(ratio)
            break
    if derived_val_split_ratio is not None and abs(validation_split_ratio - derived_val_split_ratio) > 1e-12:
        log.info(
            "Overriding validation_split_ratio from %.4f to %.4f to match disjoint train/val "
            "partitioning (e.g. shared-source tripartition 8:1:1 or PubMedQA balanced splits).",
            float(validation_split_ratio),
            float(derived_val_split_ratio),
        )
        validation_split_ratio = derived_val_split_ratio

    # For PubMedQA mixed-context prompts, assign context on a balanced randomized
    # per-example basis across model indices.
    pubmedqa_context_seed = dataset_split_seed
    pubmedqa_assignments = assign_pubmedqa_context_models(
        train_datasets,
        [model_cfg["path"] for model_cfg in args["models"]],
        random_seed=pubmedqa_context_seed,
    )
    for dataset_idx, assignment_info in pubmedqa_assignments.items():
        dataset_name = dataset_names[dataset_idx] if dataset_idx < len(dataset_names) else f"dataset_{dataset_idx}"
        assignment_hash = assignment_info.get("assignment_hash", "unknown")
        num_examples = assignment_info.get("num_examples", len(train_datasets[dataset_idx].x))
        routing_seed = assignment_info.get("routing_seed", pubmedqa_context_seed)
        model_context_counts = assignment_info.get("model_context_counts", [])
        log.info(
            "PubMedQA balanced mixed-context assignment for %s: assignment_hash=%s, num_examples=%s, routing_seed=%s",
            dataset_name,
            assignment_hash,
            num_examples,
            routing_seed,
        )
        if isinstance(model_context_counts, list):
            for model_idx, context_count in enumerate(model_context_counts):
                model_path = args["models"][model_idx]["path"] if model_idx < len(args["models"]) else f"model_{model_idx}"
                log.info(
                    "PubMedQA context count for %s: model_index=%d, model=%s, context_examples=%d",
                    dataset_name,
                    model_idx,
                    model_path,
                    int(context_count),
                )

    # Cache checks are performed per-dataset (matches trainer logic)

    # Load wagering method early so we can decide cache requirements
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
    hidden_state_layers_per_model = getattr(
        wagering_method,
        "hidden_state_layers_per_model",
        wagering_config.get("config", {}).get("hidden_state_layers_per_model"),
    )
    # Keep cache checks and trainer runtime on the same layer selection, even
    # for methods that don't explicitly expose hidden_state_layers.
    setattr(wagering_method, "hidden_state_layers", hidden_state_layers)
    setattr(wagering_method, "hidden_state_layers_per_model", hidden_state_layers_per_model)
    log.info(f"Loaded wagering method: {wagering_config['name']}")

    wagering_method_name = type(wagering_method).__name__
    wagering_needs_hidden_states = wagering_method_name not in [
        "EqualWagers",
        "ZeroOneWagers",
        "OneZeroWagers",
    ]
    resolved_hidden_layers_per_model: List[Optional[List[int]]] = [
        resolve_hidden_state_layers_for_model(
            hidden_state_layers,
            hidden_state_layers_per_model,
            model_index=model_idx,
            num_models=num_models,
        )
        if wagering_needs_hidden_states
        else None
        for model_idx in range(num_models)
    ]
    reuse_calibration_from_wagering = (
        logit_calibrator is not None
        and wagering_needs_hidden_states
        and all(
            tuple(layers) == (-1,)
            for layers in resolved_hidden_layers_per_model
            if layers is not None
        )
    )
    needs_separate_calibration_hidden_states = (
        logit_calibrator is not None and not reuse_calibration_from_wagering
    )
    needs_hidden_states = wagering_needs_hidden_states or logit_calibrator is not None
    needs_model_objects_for_perplexity = bool(
        getattr(wagering_method, "requires_model_perplexities", False)
    )
    if needs_model_objects_for_perplexity:
        log.info(
            "Wagering method %s requires model perplexities; forcing model object loading for all models.",
            wagering_method_name,
        )

    # Determine which models need to be loaded based on cache
    option_tokens = args.get("option_tokens", ["A", "B", "C", "D"])
    model_cfgs = args["models"]

    cache_miss_indices = []
    cached_model_names = []
    for idx, model_cfg in enumerate(model_cfgs):
        model_path = model_cfg["path"]
        cached_model_names.append(model_path.replace("/", "_"))

        model_hidden_layers = resolve_hidden_state_layers_for_model(
            hidden_state_layers,
            hidden_state_layers_per_model,
            model_index=idx,
            num_models=num_models,
        )
        model_cache_ok = True
        for dataset in train_datasets:
            prompt_variant = get_model_prompt_variant(dataset, model_index=idx)
            cached_logits, cached_hidden_states, _ = get_cached_logits_and_hidden_states_for_model(
                model_path,
                dataset,
                option_tokens,
                prompt_variant=prompt_variant,
                model_index=idx,
                hidden_state_layers=model_hidden_layers,
            )
            if cached_logits is None or (needs_hidden_states and cached_hidden_states is None):
                model_cache_ok = False
                break
            if needs_separate_calibration_hidden_states:
                _, cal_cached_hidden_states, _ = get_cached_logits_and_hidden_states_for_model(
                    model_path,
                    dataset,
                    option_tokens,
                    prompt_variant=prompt_variant,
                    model_index=idx,
                    hidden_state_layers=[-1],
                )
                if cal_cached_hidden_states is None:
                    model_cache_ok = False
                    break

        if not model_cache_ok:
            cache_miss_indices.append(idx)

    models = []
    model_names = cached_model_names[:]
    indices_to_load = set(cache_miss_indices)
    if needs_model_objects_for_perplexity:
        indices_to_load = set(range(num_models))

    use_sequential_perplexity = (
        needs_model_objects_for_perplexity
        and should_load_prompt_perplexity_models_sequentially(num_models)
    )
    perplexity_cache_kwargs = (
        {"cache_dir": args.get("cache_path", "./workdir/cache")} if args.get("cache_path") else {}
    )

    if use_sequential_perplexity:
        log.info(
            "Prompt perplexity with %d models on %d visible CUDA device(s): "
            "deferring full ensemble load; trainer will load one model at a time.",
            num_models,
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        )
        models = [model_cfgs[i]["path"] for i in range(num_models)]
    elif indices_to_load:
        sorted_indices_to_load = sorted(indices_to_load)
        log.info(
            "Loading %d/%d models as objects (cache misses=%d, requires_perplexity=%s).",
            len(sorted_indices_to_load),
            num_models,
            len(cache_miss_indices),
            needs_model_objects_for_perplexity,
        )
        missing_cfgs = [model_cfgs[i] for i in sorted_indices_to_load]
        missing_models, missing_names = load_models_from_config(
            missing_cfgs,
            cache_kwargs=perplexity_cache_kwargs,
        )
        missing_name_map = {idx: name for idx, name in zip(sorted_indices_to_load, missing_names)}
        missing_iter = iter(missing_models)
        for i in range(num_models):
            if i in indices_to_load:
                models.append(next(missing_iter))
                model_names[i] = missing_name_map.get(i, model_names[i])
            else:
                models.append(model_cfgs[i]["path"])
    else:
        log.info("All models are cached. Skipping model loading.")
        models = [cfg["path"] for cfg in model_cfgs]

    log.info(f"Prepared {len(models)} models: {model_names}")
    
    # Load aggregation function
    aggregation_config = args["aggregation"]
    aggregation_function = load_aggregation_function(
        aggregation_config["name"],
        config=aggregation_config.get("config", {}),
    )
    log.info(f"Loaded aggregation function: {aggregation_config['name']}")
    
    # Check for resume checkpoint
    auto_resume = args.get("auto_resume", True)
    resume_checkpoint = args.get("resume_from_checkpoint", None)
    
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if not resume_path.is_absolute():
            resume_path = checkpoint_dir / resume_checkpoint
        if resume_path.exists():
            log.info(f"Resuming from checkpoint: {resume_path}")
            resume_checkpoint = str(resume_path)
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    elif auto_resume:
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*_step_*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            log.info(f"Auto-resuming from: {latest_checkpoint}")
            resume_checkpoint = str(latest_checkpoint)
    
    env_prob_dbg = os.environ.get("WAGERING_DEBUG_PROB_ALIGN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    debug_batch_prob_alignment = bool(args.get("debug_batch_prob_alignment", False)) or env_prob_dbg

    # Create trainer
    trainer = WageringTrainer(
        models=models,
        datasets=train_datasets,
        wagering_method=wagering_method,
        aggregation_function=aggregation_function,
        option_tokens=args.get("option_tokens", ["A", "B", "C", "D"]),
        checkpoint_dir=checkpoint_dir,
        wandb_logger=wandb_logger,
        save_every=args.get("save_every", 100),
        metadata=checkpoint_metadata,
        resume_from_checkpoint=resume_checkpoint,
        shuffle_data=args.get("shuffle_data", True),
        shuffle_seed=args.get("shuffle_seed", 42),
        early_stopping_patience=args.get("early_stopping_patience", 10),
        stop_at_last_iteration=args.get("stop_at_last_iteration", False),
        early_stopping_criterion=args.get("early_stopping_criterion", "validation"),
        use_brier_d_regret_for_early_stopping=args.get("use_brier_d_regret_for_early_stopping", False),
        use_min_kl_for_early_stopping=args.get("use_min_kl_for_early_stopping", False),
        batch_size=args.get("training_batch_size", 100),
        validation_split_ratio=validation_split_ratio,
        balance_training_datasets=args.get("balance_training_datasets", True),
        wager_score_plot_every=args.get("wager_score_plot_every", None),
        logit_calibrator=logit_calibrator,
        max_training_batches=args.get("max_training_batches"),
        model_configs_for_sequential_perplexity=(model_cfgs if needs_model_objects_for_perplexity else None),
        perplexity_load_cache_kwargs=perplexity_cache_kwargs if perplexity_cache_kwargs else None,
        debug_batch_prob_alignment=debug_batch_prob_alignment,
    )
    
    # Train
    log.info("Starting training...")
    results = trainer.train(num_epochs=args.get("num_epochs", 100))
    
    log.info(f"Training complete! Final accuracy: {results['final_accuracy']:.4f}")
    
    # Save final checkpoint
    final_checkpoint_dir = Path(checkpoint_dir) / "final"
    trainer.save_final_checkpoint(str(final_checkpoint_dir))
    
    # Return results (wandb run stays active)
    results["checkpoint_path"] = str(checkpoint_dir)
    if calibration_artifact_path is not None:
        results["calibration_path"] = calibration_artifact_path
    if wandb_logger and hasattr(wandb_logger, 'run') and wandb_logger.run is not None:
        results["wandb_run_id"] = wandb_logger.run.id
        results["wandb_run_name"] = wandb_logger.run.name
    
    return results


if __name__ == "__main__":
    main()
