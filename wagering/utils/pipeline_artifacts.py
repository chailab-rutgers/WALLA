"""Pipeline run artifact persistence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

EVAL_METRICS_KEYS = [
    "accuracy",
    "nll",
    "brier",
    "auc",
    "ece",
    "inverse_hhi",
    "avg_inference_time_per_batch_s",
    "d_regret",
    "brier_d_regret",
    "meta_acc",
    "meta_nll",
    "meta_auc",
    "kendall_tau",
    "best_model_mrr",
    "bernoulli_kl",
    "bernoulli_tv",
]


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return float(value)


def eval_results_to_metrics_json(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    per_dataset: Dict[str, Dict[str, float]] = {}
    per_dataset_subset_any_model_wrong: Dict[str, Dict[str, Any]] = {}
    overall_accum = {key: [] for key in EVAL_METRICS_KEYS}
    subset_overall_accum = {key: [] for key in EVAL_METRICS_KEYS}

    for ds_name, ds_res in eval_results.items():
        if not isinstance(ds_res, dict):
            continue
        out_row: Dict[str, float] = {}
        for key in EVAL_METRICS_KEYS:
            fv = _to_float(ds_res.get(key))
            if fv is not None:
                out_row[key] = fv
                overall_accum[key].append(fv)
        if out_row:
            per_dataset[str(ds_name)] = out_row

        subset = ds_res.get("subset_any_model_wrong")
        if isinstance(subset, dict):
            sub_row: Dict[str, Any] = {}
            for key in EVAL_METRICS_KEYS:
                fv = _to_float(subset.get(key))
                if fv is not None:
                    sub_row[key] = fv
                    subset_overall_accum[key].append(fv)
            if sub_row:
                sub_row["num_examples"] = subset.get("num_examples")
                per_dataset_subset_any_model_wrong[str(ds_name)] = sub_row

    overall = {
        key: sum(vals) / float(len(vals))
        for key, vals in overall_accum.items()
        if vals
    }
    subset_overall = {
        key: sum(vals) / float(len(vals))
        for key, vals in subset_overall_accum.items()
        if vals
    }

    return {
        "per_dataset": per_dataset,
        "overall_mean_across_datasets": overall,
        "subset_any_model_wrong": {
            "per_dataset": per_dataset_subset_any_model_wrong,
            "overall_mean_across_datasets": subset_overall,
        },
    }


def write_pipeline_artifacts(
    *,
    config_path: Path,
    merged_config: Dict[str, Any],
    train_results: Optional[Dict[str, Any]],
    eval_results: Optional[Dict[str, Any]],
    calibration_path: Optional[str],
    checkpoint_path: Optional[str],
) -> Path:
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required to write pipeline artifacts")

    run_dir = Path(checkpoint_path).expanduser().resolve() / "pipeline_artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    src_copy = run_dir / "config.original.yaml"
    if config_path.exists() and not src_copy.exists():
        shutil.copy2(config_path, src_copy)

    with (run_dir / "config.merged.yaml").open("w") as f:
        yaml.safe_dump(merged_config, f, sort_keys=False)

    summary = {
        "config_path": str(config_path),
        "checkpoint_path": checkpoint_path,
        "calibration_path": calibration_path,
        "wandb_name": merged_config.get("wandb_name"),
    }
    with (run_dir / "pipeline.summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    if train_results is not None:
        with (run_dir / "train.results.json").open("w") as f:
            json.dump(train_results, f, indent=2, default=json_default)

    if eval_results is not None:
        with (run_dir / "eval.results.json").open("w") as f:
            json.dump(eval_results, f, indent=2, default=json_default)
        metrics_out = eval_results_to_metrics_json(eval_results)
        with (run_dir / "eval.metrics.json").open("w") as f:
            json.dump(metrics_out, f, indent=2, default=json_default)

    return run_dir
