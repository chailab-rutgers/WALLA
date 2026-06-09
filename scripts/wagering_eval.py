#!/usr/bin/env python3
"""
Evaluation script for multi-LLM wagering methods.

Checks if wandb.run is active from training and continues it.

Usage: python wagering_eval.py <config_file.yaml>
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.utils import (
    load_dataset_from_config,
    load_datasets_from_config,
    load_and_merge_configs,
)
from wagering.calibration import calibration_enabled, fit_or_load_logit_calibrator
from wagering.utils.checkpoint_loading import load_wagering_method_from_final_dir
from wagering.utils.cache_manager import configure_wagering_cache_dir
from wagering.utils.model_prep import prepare_ensemble_for_run
from wagering.utils.prompt_manager import assign_pubmedqa_context_models
from wagering.utils.script_runtime import ensure_project_venv
from wagering.utils.wandb_script import (
    init_wandb_for_eval,
    load_api_keys,
    resolve_eval_wandb_starting_step,
)
from wagering.methods.factory import load_wagering_method
from wagering.inference import WageringEvaluator
from wagering.aggregation.factory import load_aggregation_function

ensure_project_venv()

log = logging.getLogger("wagering")


def main(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    calibration_path: Optional[str] = None,
):
    if config_path is None:
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            raise ValueError(
                "Config file path required. Usage: python wagering_eval.py <config_file.yaml>"
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

    api_keys = load_api_keys()
    wandb_logger = None
    if args.get("report_to_wandb", False):
        try:
            wandb_logger = init_wandb_for_eval(args, api_keys)
        except ImportError as exc:
            raise RuntimeError("wandb not available but report_to_wandb is enabled in config") from exc

    logit_calibrator = None
    if calibration_enabled(args):
        log.info("Preparing cached-logit temperature calibrator...")
        logit_calibrator, calibration_path, _ = fit_or_load_logit_calibrator(
            args,
            calibration_path=calibration_path,
        )

    dataset_split_seed = int(args.get("dataset_split_seed", 42))
    tr_peer = [args["dataset"]] if args.get("dataset") else None
    test_datasets = []
    if "test_dataset" in args:
        test_ds, test_name = load_dataset_from_config(
            args["test_dataset"],
            split="test",
            random_seed=dataset_split_seed,
            partition_peer_dataset_configs=tr_peer,
            infer_eval_split_train_without_peer=False,
            force_partition=True,
        )
        test_datasets = [(test_ds, test_name)]

    ood_datasets = []
    if args.get("ood_datasets"):
        ood_ds, ood_names = load_datasets_from_config(
            args["ood_datasets"],
            split="test",
            random_seed=dataset_split_seed,
            partition_peer_dataset_configs=tr_peer,
            infer_eval_split_train_without_peer=False,
        )
        ood_datasets.extend((ds, name) for ds, name in zip(ood_ds, ood_names))

    eval_dataset_objects = [ds for ds, _ in test_datasets]
    if ood_datasets:
        eval_dataset_objects.extend(ds for ds, _ in ood_datasets)

    model_paths = [model_cfg["path"] for model_cfg in args["models"]]
    assign_pubmedqa_context_models(
        eval_dataset_objects,
        model_paths,
        random_seed=dataset_split_seed,
    )

    wagering_config = args["wagering_method"]
    num_models = len(args["models"])
    wagering_method = load_wagering_method(
        wagering_config["name"],
        num_models=num_models,
        config=wagering_config.get("config", {}),
    )
    requires_checkpoint = len(wagering_method.get_trainable_parameters()) > 0

    if checkpoint_path is None:
        checkpoint_path = args.get("checkpoint_path")
    if requires_checkpoint and checkpoint_path is None:
        raise ValueError("Please provide a checkpoint path in config file")

    checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path is not None else None

    needs_hidden_states = bool(getattr(wagering_method, "requires_hidden_states", True)) or (
        logit_calibrator is not None
    )
    option_tokens = args.get("option_tokens", ["A", "B", "C", "D"])
    model_cfgs = args["models"]

    eval_datasets = [ds for ds, _ in test_datasets]
    if ood_datasets:
        eval_datasets.extend(ds for ds, _ in ood_datasets)

    force_load_all_models = bool(getattr(wagering_method, "requires_model_perplexities", False))
    if force_load_all_models:
        log.info(
            "Wagering method requires model prompt perplexities; loading all %d models for evaluation.",
            num_models,
        )

    models, model_names = prepare_ensemble_for_run(
        model_cfgs,
        eval_datasets,
        option_tokens,
        needs_hidden_states=needs_hidden_states,
        force_load_all_for_perplexity=force_load_all_models,
        cache_path=args["cache_path"],
        num_models=num_models,
    )

    if requires_checkpoint:
        load_wagering_method_from_final_dir(wagering_method, checkpoint_path_obj)
        log.info("Loaded wagering method checkpoint from %s", checkpoint_path_obj)
    else:
        log.info("Wagering method has no trainable parameters - skipping checkpoint loading")

    aggregation_function = load_aggregation_function(args["aggregation"]["name"])

    if args.get("eval_checkpoint_dir"):
        eval_checkpoint_dir = Path(args["eval_checkpoint_dir"])
    elif checkpoint_path_obj:
        eval_checkpoint_dir = checkpoint_path_obj / "eval"
    else:
        eval_checkpoint_dir = Path("./eval_outputs")
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info("Evaluation outputs: %s", eval_checkpoint_dir)

    metadata = {"model_names": model_names}
    training_datasets = args.get("training_datasets", [])
    if training_datasets:
        if isinstance(training_datasets, str):
            training_datasets = [training_datasets]
        elif isinstance(training_datasets, list) and training_datasets and isinstance(
            training_datasets[0], dict
        ):
            training_datasets = [
                ds.get("name", ds.get("path", str(ds))) for ds in training_datasets
            ]
        metadata["training_datasets"] = training_datasets

    seed = args.get("seed", args.get("shuffle_seed", None))
    wandb_starting_step = resolve_eval_wandb_starting_step(wandb_logger)
    perplexity_cache_kwargs = {"cache_dir": args["cache_path"]}

    evaluator = WageringEvaluator(
        models=models,
        wagering_method=wagering_method,
        aggregation_function=aggregation_function,
        option_tokens=option_tokens,
        wandb_logger=wandb_logger,
        checkpoint_dir=eval_checkpoint_dir,
        metadata=metadata,
        training_checkpoint_path=str(checkpoint_path_obj) if checkpoint_path_obj is not None else None,
        seed=seed,
        wandb_starting_step=wandb_starting_step,
        logit_calibrator=logit_calibrator,
        model_configs_for_sequential_perplexity=(model_cfgs if force_load_all_models else None),
        perplexity_load_cache_kwargs=perplexity_cache_kwargs if perplexity_cache_kwargs else None,
    )

    log.info("Starting evaluation...")
    log.info("  Test datasets: %d", len(test_datasets))
    if ood_datasets:
        log.info("  OOD datasets: %d", len(ood_datasets))
        for _, ood_name in ood_datasets:
            log.info("    - %s", ood_name)

    results = evaluator.evaluate_multiple(
        test_datasets=test_datasets,
        ood_datasets=ood_datasets,
        resume=False,
    )

    log.info("Evaluation Results:")
    results_summary = {}
    aggregate_metrics = {key: [] for key in (
        "accuracy", "nll", "brier", "auc", "ece", "inverse_hhi",
        "avg_inference_time_per_batch_s", "brier_d_regret", "kendall_tau",
        "best_model_mrr", "bernoulli_kl", "bernoulli_tv",
    )}

    def get_metric_float(result, key):
        val = result.get(key, None)
        if val is None:
            return None
        val_scalar = float(val) if not isinstance(val, (list, np.ndarray)) else float(val[0])
        return None if np.isnan(val_scalar) else val_scalar

    def get_metric_str(result, key):
        val = get_metric_float(result, key)
        return "N/A" if val is None else f"{val:.4f}"

    for dataset_name, result in results.items():
        if not isinstance(result, dict):
            continue
        subset = result.get("subset_any_model_wrong")
        for key in aggregate_metrics:
            val = get_metric_float(result, key)
            if val is not None:
                aggregate_metrics[key].append(val)

        log.info(
            "%s: Accuracy=%s, NLL=%s, Brier=%s, KL=%s, TV=%s, AUC=%s, ECE=%s, "
            "InverseHHI=%s, AvgInferenceTimePerBatchS=%s, BrierDRegret=%s, "
            "KendallTau=%s, BestModelMRR=%s",
            dataset_name,
            get_metric_str(result, "accuracy"),
            get_metric_str(result, "nll"),
            get_metric_str(result, "brier"),
            get_metric_str(result, "bernoulli_kl"),
            get_metric_str(result, "bernoulli_tv"),
            get_metric_str(result, "auc"),
            get_metric_str(result, "ece"),
            get_metric_str(result, "inverse_hhi"),
            get_metric_str(result, "avg_inference_time_per_batch_s"),
            get_metric_str(result, "brier_d_regret"),
            get_metric_str(result, "kendall_tau"),
            get_metric_str(result, "best_model_mrr"),
        )

        if isinstance(subset, dict) and int(subset.get("num_examples", 0) or 0) > 0:
            sub_n = int(subset.get("num_examples"))
            log.info(
                "%s [subset any-model-wrong n=%d]: Accuracy=%s, NLL=%s, KL=%s, TV=%s, ECE=%s, BrierDRegret=%s",
                dataset_name,
                sub_n,
                get_metric_str(subset, "accuracy"),
                get_metric_str(subset, "nll"),
                get_metric_str(subset, "bernoulli_kl"),
                get_metric_str(subset, "bernoulli_tv"),
                get_metric_str(subset, "ece"),
                get_metric_str(subset, "brier_d_regret"),
            )

        results_summary[dataset_name] = (
            f"Accuracy={get_metric_str(result, 'accuracy')}, AUC={get_metric_str(result, 'auc')}, "
            f"ECE={get_metric_str(result, 'ece')}, InverseHHI={get_metric_str(result, 'inverse_hhi')}, "
            f"AvgInferenceTimePerBatchS={get_metric_str(result, 'avg_inference_time_per_batch_s')}, "
            f"NLL={get_metric_str(result, 'nll')}, Brier={get_metric_str(result, 'brier')}, "
            f"KL={get_metric_str(result, 'bernoulli_kl')}, TV={get_metric_str(result, 'bernoulli_tv')}, "
            f"BrierDRegret={get_metric_str(result, 'brier_d_regret')}, "
            f"KendallTau={get_metric_str(result, 'kendall_tau')}, "
            f"BestModelMRR={get_metric_str(result, 'best_model_mrr')}"
        )

    def aggregate_metric_str(values):
        if not values:
            return "N/A"
        mean_val = float(np.mean(values))
        return "N/A" if np.isnan(mean_val) else f"{mean_val:.4f}"

    results_summary["overall"] = (
        f"Accuracy={aggregate_metric_str(aggregate_metrics['accuracy'])}, "
        f"AUC={aggregate_metric_str(aggregate_metrics['auc'])}, "
        f"ECE={aggregate_metric_str(aggregate_metrics['ece'])}, "
        f"InverseHHI={aggregate_metric_str(aggregate_metrics['inverse_hhi'])}, "
        f"AvgInferenceTimePerBatchS={aggregate_metric_str(aggregate_metrics['avg_inference_time_per_batch_s'])}, "
        f"NLL={aggregate_metric_str(aggregate_metrics['nll'])}, "
        f"Brier={aggregate_metric_str(aggregate_metrics['brier'])}, "
        f"KL={aggregate_metric_str(aggregate_metrics['bernoulli_kl'])}, "
        f"TV={aggregate_metric_str(aggregate_metrics['bernoulli_tv'])}, "
        f"BrierDRegret={aggregate_metric_str(aggregate_metrics['brier_d_regret'])}, "
        f"KendallTau={aggregate_metric_str(aggregate_metrics['kendall_tau'])}, "
        f"BestModelMRR={aggregate_metric_str(aggregate_metrics['best_model_mrr'])}"
    )

    if wandb_logger:
        import wandb

        if wandb.run is not None:
            summary_df = pd.DataFrame(
                [{"Dataset": name, "Metrics": metrics} for name, metrics in results_summary.items()]
            )
            wandb.log({"evaluation/results_summary": wandb.Table(dataframe=summary_df)}, commit=True)
            log.info("Finishing wandb run: %s", wandb.run.id)
            wandb.finish()

    return results


if __name__ == "__main__":
    main()
