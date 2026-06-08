"""Repeat-run eval.metrics.json aggregation and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

AGGREGATE_PRINT_SCALE_100 = frozenset(
    {
        "accuracy",
        "auc",
        "ece",
        "best_model_mrr",
        "brier_d_regret",
        "kendall_tau",
        "bernoulli_kl",
        "bernoulli_tv",
    }
)

DEFAULT_AGGREGATE_METRICS = [
    "accuracy",
    "auc",
    "bernoulli_kl",
    "bernoulli_tv",
    "brier_d_regret",
    "kendall_tau",
    "best_model_mrr",
    "ece",
    "inverse_hhi",
    "avg_inference_time_per_batch_s",
    "d_regret",
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    with path.open("w") as f:
        json.dump(data, f, indent=2, default=_default)


def mean_std_ci95(values: List[float]) -> Dict[str, Any]:
    import numpy as np
    from scipy import stats

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    if n >= 2:
        sem = std / float(np.sqrt(n))
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        margin = tcrit * sem
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
        }
    return {"n": n, "mean": mean, "std": std, "ci95_low": None, "ci95_high": None}


def format_mean_ci_for_metric(
    col: str,
    mean: float,
    ci_low: Optional[float],
    ci_high: Optional[float],
    n: int,
) -> str:
    scale = 100.0 if col in AGGREGATE_PRINT_SCALE_100 else 1.0
    dec = 2 if scale == 100.0 else 4
    m = float(mean) * scale
    if ci_low is not None and ci_high is not None:
        lo_f, hi_f = float(ci_low) * scale, float(ci_high) * scale
        margin = (hi_f - lo_f) / 2.0
        return f"{m:.{dec}f}\\tiny{{$\\pm${margin:.{dec}f}}}  $n={int(n)}$"
    return f"{m:.{dec}f}  $n={int(n)}$"


def collect_eval_metrics_json_paths(repeat_root: Path) -> List[Path]:
    paths = sorted(repeat_root.glob("repeat_*/**/pipeline_artifacts/eval.metrics.json"))
    return [path for path in paths if path.is_file()]


def aggregate_eval_metrics_json(
    repeat_root: Path,
    metrics: Sequence[str],
) -> Optional[Dict[str, Any]]:
    paths = collect_eval_metrics_json_paths(repeat_root)
    if not paths:
        return None

    loaded = [json.loads(path.read_text()) for path in paths]

    per_dataset_values: Dict[str, Dict[str, List[float]]] = {}
    subset_per_dataset_values: Dict[str, Dict[str, List[float]]] = {}

    for blob in loaded:
        per_ds = blob.get("per_dataset", {})
        if isinstance(per_ds, dict):
            for ds, row in per_ds.items():
                if not isinstance(row, dict):
                    continue
                per_dataset_values.setdefault(str(ds), {})
                for metric in metrics:
                    value = row.get(metric)
                    if value is not None:
                        per_dataset_values[str(ds)].setdefault(metric, []).append(float(value))

        subset = blob.get("subset_any_model_wrong", {})
        if isinstance(subset, dict):
            spd = subset.get("per_dataset", {})
            if isinstance(spd, dict):
                for ds, row in spd.items():
                    if not isinstance(row, dict):
                        continue
                    subset_per_dataset_values.setdefault(str(ds), {})
                    for metric in metrics:
                        value = row.get(metric)
                        if value is not None:
                            subset_per_dataset_values[str(ds)].setdefault(metric, []).append(
                                float(value)
                            )

    def _build_section(
        values_by_ds: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, Any]:
        out_per_ds: Dict[str, Any] = {}
        overall_vals: Dict[str, List[float]] = {metric: [] for metric in metrics}

        for ds, by_metric in values_by_ds.items():
            out_per_ds[ds] = {}
            n_ds = 0
            for metric, vals in by_metric.items():
                stats = mean_std_ci95(vals)
                n_ds = max(n_ds, int(stats["n"]))
                if stats["mean"] is not None:
                    out_per_ds[ds][metric] = {
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "ci95_low": stats["ci95_low"],
                        "ci95_high": stats["ci95_high"],
                        "formatted": format_mean_ci_for_metric(
                            metric,
                            stats["mean"],
                            stats["ci95_low"],
                            stats["ci95_high"],
                            int(stats["n"]),
                        ),
                    }
                    overall_vals[metric].extend(float(x) for x in vals)
            out_per_ds[ds]["num_repeats_with_metrics"] = n_ds

        overall_out: Dict[str, Any] = {}
        for metric, vals in overall_vals.items():
            if not vals:
                continue
            stats = mean_std_ci95(vals)
            if stats["mean"] is None:
                continue
            overall_out[metric] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "formatted": format_mean_ci_for_metric(
                    metric,
                    stats["mean"],
                    stats["ci95_low"],
                    stats["ci95_high"],
                    int(stats["n"]),
                ),
            }

        return {"per_dataset": out_per_ds, "overall": overall_out}

    return {
        "repeat_root": str(repeat_root),
        "num_repeats_with_eval_metrics_json": len(loaded),
        "full": _build_section(per_dataset_values),
        "subset_any_model_wrong": _build_section(subset_per_dataset_values),
    }


def print_aggregated_metrics(
    agg: Dict[str, Any],
    metrics: Sequence[str],
    section_key: str,
) -> None:
    def _print_block(overall_dict: Any) -> None:
        if not isinstance(overall_dict, dict) or not overall_dict:
            return
        first = True
        for metric in metrics:
            row = overall_dict.get(metric)
            if not isinstance(row, dict) or "formatted" not in row:
                continue
            prefix = "" if first else "  "
            print(f"{prefix}{metric}: {row['formatted']}", flush=True)
            first = False

    def _print_per_dataset(section: Any) -> bool:
        if not isinstance(section, dict):
            return False
        per_ds = section.get("per_dataset", {})
        if not isinstance(per_ds, dict) or not per_ds or len(per_ds) <= 1:
            return False
        for ds_name, ds_blob in per_ds.items():
            if not isinstance(ds_blob, dict):
                continue
            print(f"[{ds_name}]", flush=True)
            _print_block(ds_blob)
        return True

    section = agg.get(section_key, {})
    if section_key == "subset_any_model_wrong":
        print("subset_any_model_wrong", flush=True)
    if not isinstance(section, dict):
        return
    if not _print_per_dataset(section):
        _print_block(section.get("overall", {}))


def make_calibrated_config_variants(
    base_cfg_path: Path,
    *,
    num_epochs: int,
    out_dir: Path,
    gemma_model_index: int = 0,
) -> Tuple[Path, Path]:
    """Write uncalibrated and Gemma2-calibrated config YAMLs derived from a base config."""
    from wagering.utils.config_utils import load_yaml_file

    base_cfg_path = base_cfg_path.expanduser().resolve()
    base = load_yaml_file(base_cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    uncal = dict(base)
    uncal["calibrated"] = False
    uncal.pop("calibration", None)
    uncal["num_epochs"] = int(num_epochs)

    cal = dict(base)
    cal["calibrated"] = True
    cal["num_epochs"] = int(num_epochs)
    cal["calibration"] = {"apply_to_model_indices": [int(gemma_model_index)]}

    stem = base_cfg_path.stem
    uncal_path = out_dir / f"{stem}__uncalibrated.yaml"
    cal_path = out_dir / f"{stem}__gemma2_calibrated.yaml"
    import yaml

    for path, data in ((uncal_path, uncal), (cal_path, cal)):
        with path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    return uncal_path, cal_path
