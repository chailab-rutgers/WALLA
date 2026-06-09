#!/usr/bin/env python3
"""
Training script for multi-LLM wagering methods.

Does NOT call wandb.finish() to keep the run active for evaluation.

Usage: python wagering_train.py <config_file.yaml>
"""

import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.utils import (
    load_and_merge_configs,
    generate_checkpoint_dir,
    get_checkpoint_metadata,
)
from wagering.calibration import calibration_enabled, fit_or_load_logit_calibrator
from wagering.utils.cache_manager import configure_wagering_cache_dir
from wagering.utils.model_prep import prepare_ensemble_for_run
from wagering.utils.prompt_manager import assign_pubmedqa_context_models
from wagering.utils.script_runtime import ensure_project_venv
from wagering.utils.wandb_script import init_wandb_for_training
from wagering.methods.factory import load_wagering_method
from wagering.training import WageringTrainer
from wagering.aggregation.factory import load_aggregation_function

ensure_project_venv()

log = logging.getLogger("wagering")


def main(config_path: Optional[str] = None, calibration_path: Optional[str] = None):
    if config_path is None:
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            raise ValueError(
                "Config file path required. Usage: python wagering_train.py <config_file.yaml>"
            )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    args = load_and_merge_configs(config_path)
    configure_wagering_cache_dir(args["cache_path"])
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
    logging.getLogger("wagering").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    base_checkpoint_dir = Path(args.get("checkpoint_base_dir", "/common/users/yl2310/WALLA/checkpoints"))
    calibration_config = args.get("calibration") if calibration_enabled(args) else None
    checkpoint_dir = generate_checkpoint_dir(
        base_dir=base_checkpoint_dir,
        models=args["models"],
        dataset=args["dataset"],
        wagering_method=args["wagering_method"],
        aggregation=args["aggregation"],
        create_hash=True,
        calibration=calibration_config,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoint directory: %s", checkpoint_dir)

    checkpoint_metadata = get_checkpoint_metadata(
        models=args["models"],
        dataset=args["dataset"],
        wagering_method=args["wagering_method"],
        aggregation=args["aggregation"],
        calibration=calibration_config,
    )

    wandb_logger = None
    if args.get("report_to_wandb", False):
        try:
            wandb_logger = init_wandb_for_training(args, checkpoint_metadata)
        except ImportError as exc:
            raise RuntimeError("wandb not available but report_to_wandb is enabled in config") from exc

    logit_calibrator = None
    calibration_artifact_path = calibration_path
    if calibration_enabled(args):
        log.info("Preparing cached-logit temperature calibrator...")
        logit_calibrator, calibration_artifact_path, _ = fit_or_load_logit_calibrator(
            args,
            calibration_path=calibration_path,
        )

    log.info("Loading training dataset...")
    from wagering.utils import load_dataset_from_config

    dataset_split_seed = int(args.get("dataset_split_seed", 42))
    test_peer = [args["test_dataset"]] if args.get("test_dataset") else None
    train_dataset, dataset_name = load_dataset_from_config(
        args["dataset"],
        split="train",
        random_seed=dataset_split_seed,
        partition_peer_dataset_configs=test_peer,
    )
    log.info("Loaded training dataset: %s", dataset_name)

    validation_split_ratio = float(args.get("validation_split_ratio", 0.1))
    derived_val_split_ratio = getattr(train_dataset, "partition_val_ratio", None)
    if derived_val_split_ratio is not None:
        derived_val_split_ratio = float(derived_val_split_ratio)
        if abs(validation_split_ratio - derived_val_split_ratio) > 1e-12:
            log.info(
                "Overriding validation_split_ratio from %.4f to %.4f to match disjoint train/val partitioning.",
                validation_split_ratio,
                derived_val_split_ratio,
            )
            validation_split_ratio = derived_val_split_ratio

    model_paths = [model_cfg["path"] for model_cfg in args["models"]]
    assign_pubmedqa_context_models([train_dataset], model_paths, random_seed=dataset_split_seed)

    wagering_config = args["wagering_method"]
    num_models = len(args["models"])
    wagering_method = load_wagering_method(
        wagering_config["name"],
        num_models=num_models,
        config=wagering_config.get("config", {}),
    )
    log.info("Loaded wagering method: %s", wagering_config["name"])

    needs_hidden_states = bool(getattr(wagering_method, "requires_hidden_states", True)) or (
        logit_calibrator is not None
    )
    needs_model_objects_for_perplexity = bool(
        getattr(wagering_method, "requires_model_perplexities", False)
    )
    if needs_model_objects_for_perplexity:
        log.info(
            "Wagering method %s requires model perplexities; forcing model object loading for all models.",
            type(wagering_method).__name__,
        )

    option_tokens = args.get("option_tokens", ["A", "B", "C", "D"])
    models, model_names = prepare_ensemble_for_run(
        args["models"],
        [train_dataset],
        option_tokens,
        needs_hidden_states=needs_hidden_states,
        force_load_all_for_perplexity=needs_model_objects_for_perplexity,
        cache_path=args["cache_path"],
        num_models=num_models,
    )

    aggregation_config = args["aggregation"]
    aggregation_function = load_aggregation_function(aggregation_config["name"])
    log.info("Loaded aggregation function: %s", aggregation_config["name"])

    perplexity_cache_kwargs = {"cache_dir": args["cache_path"]}
    trainer = WageringTrainer(
        models=models,
        dataset=train_dataset,
        wagering_method=wagering_method,
        aggregation_function=aggregation_function,
        option_tokens=option_tokens,
        checkpoint_dir=checkpoint_dir,
        wandb_logger=wandb_logger,
        metadata=checkpoint_metadata,
        shuffle_data=args.get("shuffle_data", True),
        shuffle_seed=args.get("shuffle_seed", 42),
        early_stopping_patience=args.get("early_stopping_patience", 10),
        early_stopping_criterion=args.get("early_stopping_criterion", "validation"),
        use_brier_d_regret_for_early_stopping=args.get("use_brier_d_regret_for_early_stopping", False),
        use_min_kl_for_early_stopping=args.get("use_min_kl_for_early_stopping", False),
        batch_size=args.get("training_batch_size", 100),
        validation_split_ratio=validation_split_ratio,
        wager_score_plot_every=args.get("wager_score_plot_every", None),
        logit_calibrator=logit_calibrator,
        max_training_batches=args.get("max_training_batches"),
        model_configs_for_sequential_perplexity=(
            args["models"] if needs_model_objects_for_perplexity else None
        ),
        perplexity_load_cache_kwargs=perplexity_cache_kwargs if perplexity_cache_kwargs else None,
    )

    log.info("Starting training...")
    results = trainer.train(num_epochs=args.get("num_epochs", 100))
    log.info("Training complete! Final accuracy: %.4f", results["final_accuracy"])

    final_checkpoint_dir = Path(checkpoint_dir) / "final"
    trainer.save_final_checkpoint(str(final_checkpoint_dir))

    results["checkpoint_path"] = str(checkpoint_dir)
    if calibration_artifact_path is not None:
        results["calibration_path"] = calibration_artifact_path
    if wandb_logger and hasattr(wandb_logger, "run") and wandb_logger.run is not None:
        results["wandb_run_id"] = wandb_logger.run.id
        results["wandb_run_name"] = wandb_logger.run.name
    return results


if __name__ == "__main__":
    main()
