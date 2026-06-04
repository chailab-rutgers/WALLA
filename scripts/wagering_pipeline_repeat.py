#!/usr/bin/env python3
"""
Repeat-run wrapper for scripts/wagering_pipeline.py.

Runs the same config N times (parallel by default), forces wandb off, varies
shuffle_seed per repeat (keeping dataset_split_seed fixed for stable splits/caches), and gives each repeat its own
checkpoint directory. Use ``--max-workers-per-gpu K`` to cap concurrency at
``K`` jobs per visible GPU (each subprocess gets its own ``CUDA_VISIBLE_DEVICES``).
Optionally set ``--gpus`` to constrain visible GPU IDs (e.g. ``--gpus 2,3``).

After repeats, aggregates eval metrics from
``<checkpoint_base>/<run_tag>/repeat_*/eval/analytics_*.csv`` (mean, sample std,
and 95%% CI for the mean via Student's t), unless ``--no-aggregate`` or
``--skip-evaluation``.

Each full repeat run creates a **new** ``run_tag`` directory (timestamp). To
recompute aggregated analytics from a previous run **without** rerunning:

  ./.venv/bin/python scripts/wagering_pipeline_repeat.py --aggregate-only /path/to/checkpoints/<run_tag>

or resolve the base directory from the same config YAML:

  ./.venv/bin/python scripts/wagering_pipeline_repeat.py --config path/to/config.yaml --aggregate-run-tag <run_tag>

Usage:
  ./.venv/bin/python scripts/wagering_pipeline_repeat.py --config path/to/config.yaml --n-repeats 5

If you run with a non-venv ``python3``, this script re-execs into ``./.venv/bin/python``
when that interpreter exists.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if __name__ == "__main__" and _VENV_PYTHON.exists():
    try:
        if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
            os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)
    except OSError:
        pass

PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "wagering_pipeline.py"
DEFAULT_VENV_PYTHON = _VENV_PYTHON

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

DEFAULT_AGGREGATE_METRICS = [
    "accuracy",
    "auc",
    "bernoulli_kl",
    "bernoulli_tv",
    "d_regret",
    "brier_d_regret",
    "kendall_tau",
    "best_model_mrr",
    "ece",
    "inverse_hhi",
    "avg_inference_time_per_batch_s",
]

# Printed as (value * scale); e.g. accuracy 0.76 -> "76.12" with scale 100.
_AGGREGATE_PRINT_SCALE_100 = frozenset(
    {"accuracy", "auc", "ece", "best_model_mrr", "d_regret", "brier_d_regret", "kendall_tau", "bernoulli_kl", "bernoulli_tv"}
)


def _format_aggregate_metric_latex_line(
    col: str,
    mean: float,
    std: Any,
    lo: Any,
    hi: Any,
    nruns: Any,
) -> str:
    """One summary line: ``metric: mean \\pm \\mathrm{ME}`` where ME is half the 95% CI width."""
    import pandas as pd

    scale = 100.0 if col in _AGGREGATE_PRINT_SCALE_100 else 1.0
    dec = 2 if scale == 100.0 else 4
    m = float(mean) * scale
    parts_out: List[str] = []
    if pd.notna(lo) and pd.notna(hi):
        lo_f, hi_f = float(lo) * scale, float(hi) * scale
        margin = (hi_f - lo_f) / 2.0
        parts_out.append(f"{m:.{dec}f}\\tiny{{$\\pm${margin:.{dec}f}}}")
    elif pd.notna(std):
        s = float(std) * scale
        parts_out.append(f"{m:.{dec}f} $\\pm$ {s:.{dec}f}  (std; no CI)")
    else:
        parts_out.append(f"{m:.{dec}f}")
    if pd.notna(nruns):
        parts_out.append(f"$n={int(nruns)}$")
    return f"{col}: " + "  ".join(parts_out)


def _format_mean_ci_for_metric(col: str, mean: float, ci_low: Optional[float], ci_high: Optional[float], n: int) -> str:
    """Format like: 94.68\\tiny{$\\pm$0.00} (scaled when appropriate)."""
    scale = 100.0 if col in _AGGREGATE_PRINT_SCALE_100 else 1.0
    dec = 2 if scale == 100.0 else 4
    m = float(mean) * scale
    if ci_low is not None and ci_high is not None:
        lo_f, hi_f = float(ci_low) * scale, float(ci_high) * scale
        margin = (hi_f - lo_f) / 2.0
        return f"{m:.{dec}f}\\tiny{{$\\pm${margin:.{dec}f}}}  $n={int(n)}$"
    return f"{m:.{dec}f}  $n={int(n)}$"


def _mean_std_ci95(values: List[float]) -> Dict[str, Any]:
    """Return mean/std and (ci_low, ci_high) for the mean using Student's t."""
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
        return {"n": n, "mean": mean, "std": std, "ci95_low": mean - margin, "ci95_high": mean + margin}
    return {"n": n, "mean": mean, "std": std, "ci95_low": None, "ci95_high": None}


def _collect_eval_metrics_json_paths(repeat_root: Path) -> List[Path]:
    # Matches: repeat_0000/<wandb_name>/pipeline_artifacts/eval.metrics.json
    paths = sorted(repeat_root.glob("repeat_*/**/pipeline_artifacts/eval.metrics.json"))
    return [p for p in paths if p.is_file()]


def _aggregate_eval_metrics_json(repeat_root: Path, metrics: List[str]) -> Optional[Dict[str, Any]]:
    """Aggregate per-repeat eval.metrics.json into a run-level summary dict."""
    import json

    paths = _collect_eval_metrics_json_paths(repeat_root)
    if not paths:
        return None

    loaded: List[Dict[str, Any]] = []
    for p in paths:
        try:
            loaded.append(json.loads(p.read_text()))
        except Exception:
            continue
    if not loaded:
        return None

    # Aggregate per dataset.
    per_dataset_values: Dict[str, Dict[str, List[float]]] = {}
    subset_per_dataset_values: Dict[str, Dict[str, List[float]]] = {}

    for blob in loaded:
        pd = blob.get("per_dataset", {})
        if isinstance(pd, dict):
            for ds, row in pd.items():
                if not isinstance(row, dict):
                    continue
                per_dataset_values.setdefault(str(ds), {})
                for m in metrics:
                    try:
                        v = row.get(m, None)
                        if v is None:
                            continue
                        per_dataset_values[str(ds)].setdefault(m, []).append(float(v))
                    except Exception:
                        continue

        subset = blob.get("subset_any_model_wrong", {})
        if isinstance(subset, dict):
            spd = subset.get("per_dataset", {})
            if isinstance(spd, dict):
                for ds, row in spd.items():
                    if not isinstance(row, dict):
                        continue
                    subset_per_dataset_values.setdefault(str(ds), {})
                    for m in metrics:
                        try:
                            v = row.get(m, None)
                            if v is None:
                                continue
                            subset_per_dataset_values[str(ds)].setdefault(m, []).append(float(v))
                        except Exception:
                            continue

    def _build_section(values_by_ds: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        out_per_ds: Dict[str, Any] = {}
        # Overall = pooled across all repeats (and datasets, if multiple).
        overall_vals: Dict[str, List[float]] = {m: [] for m in metrics}
        n_repeats_by_ds: Dict[str, int] = {}

        for ds, by_metric in values_by_ds.items():
            out_per_ds[ds] = {}
            # n for a dataset = max count across metrics present
            n_ds = 0
            for m, vals in by_metric.items():
                stats = _mean_std_ci95(vals)
                n_ds = max(n_ds, int(stats["n"]))
                if stats["mean"] is not None:
                    out_per_ds[ds][m] = {
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "ci95_low": stats["ci95_low"],
                        "ci95_high": stats["ci95_high"],
                        "formatted": _format_mean_ci_for_metric(
                            m, stats["mean"], stats["ci95_low"], stats["ci95_high"], int(stats["n"])
                        ),
                    }
                    # Pool raw per-repeat values so n reflects repeats.
                    overall_vals[m].extend([float(x) for x in vals])
            n_repeats_by_ds[ds] = n_ds
            out_per_ds[ds]["num_repeats_with_metrics"] = n_ds

        overall_out: Dict[str, Any] = {}
        for m, vals in overall_vals.items():
            if not vals:
                continue
            stats = _mean_std_ci95(vals)
            if stats["mean"] is None:
                continue
            overall_out[m] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "formatted": _format_mean_ci_for_metric(
                    m, stats["mean"], stats["ci95_low"], stats["ci95_high"], int(stats["n"])
                ),
            }

        return {"per_dataset": out_per_ds, "overall": overall_out}

    return {
        "repeat_root": str(repeat_root),
        "num_repeats_with_eval_metrics_json": len(loaded),
        "full": _build_section(per_dataset_values),
        "subset_any_model_wrong": _build_section(subset_per_dataset_values),
    }


def _add_mean_ci95_columns(agg: Any, result_columns: List[str]) -> Any:
    """
    For each metric with mean and sample std across repeats, add 95% two-sided
    CI bounds for the mean (t_{0.975, n-1} * std / sqrt(n)). Requires num_runs.
    """
    import numpy as np
    from scipy import stats

    if "num_runs" not in agg.columns:
        return agg
    out = agg.copy()
    n = out["num_runs"].to_numpy(dtype=float)
    for col in result_columns:
        std_col = f"{col}_std"
        if col not in out.columns or std_col not in out.columns:
            continue
        mean_v = out[col].to_numpy(dtype=float)
        std_v = out[std_col].to_numpy(dtype=float)
        sem = np.divide(std_v, np.sqrt(n), out=np.full_like(mean_v, np.nan), where=n > 0)
        tcrit = np.full_like(n, np.nan, dtype=float)
        mask = n >= 2
        tcrit[mask] = stats.t.ppf(0.975, df=n[mask] - 1.0)
        margin = tcrit * sem
        out[f"{col}_ci95_low"] = mean_v - margin
        out[f"{col}_ci95_high"] = mean_v + margin
    return out


def _flatten_multiindex_agg_columns(grouped: Any) -> Any:
    """Match wagering.training.analytics flattening for groupby().agg() MultiIndex columns."""
    import pandas as pd

    if not isinstance(grouped.columns, pd.MultiIndex):
        return grouped
    new_columns: List[str] = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            if col[1] == "":
                new_columns.append(col[0])
            elif col[1] == "mean":
                new_columns.append(col[0])
            elif col[1] == "first":
                new_columns.append(col[0])
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(str(col))
    out = grouped.copy()
    out.columns = new_columns
    return out


def _aggregate_eval_across_repeats(combined: Any, result_cols: List[str]) -> Any:
    """
    Mean / sample std / num_runs over repeat directories.

    ``WageringAnalytics.aggregate_results_by_settings`` groups by ``settings_hash``,
    which changes every repeat (seeds), yielding num_runs=1. Here we group by eval
    identity columns that are stable across repeats.
    """
    import pandas as pd

    keys = [
        k
        for k in ("evaluation_dataset", "wagering_method", "aggregation_method")
        if k in combined.columns
    ]
    if not keys:
        combined = combined.copy()
        combined["_repeat_whole"] = 0
        keys = ["_repeat_whole"]

    present_results = [c for c in result_cols if c in combined.columns]
    if not present_results:
        return pd.DataFrame()

    agg_dict = {c: ["mean", "std"] for c in present_results}
    grouped = combined.groupby(keys, as_index=False, dropna=False).agg(agg_dict)
    grouped = _flatten_multiindex_agg_columns(grouped)
    counts = combined.groupby(keys, dropna=False).size().reset_index(name="num_runs")
    grouped = grouped.merge(counts, on=keys, how="left")
    return grouped


def _visible_gpu_count() -> int:
    """Number of GPUs visible to this process (env or torch), or 0 if unknown."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and str(raw).strip() != "":
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if parts:
            return len(parts)
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _parse_gpu_ids(csv: str) -> List[str]:
    gpu_ids = [p.strip() for p in str(csv).split(",") if p.strip()]
    if not gpu_ids:
        raise ValueError("No GPUs provided. Example: --gpus 0,1,2,3")
    return gpu_ids


def _resolve_visible_gpu_ids(gpus_arg: Optional[str]) -> List[str]:
    if gpus_arg is not None:
        return _parse_gpu_ids(gpus_arg)
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and str(raw).strip() != "":
        ids = _parse_gpu_ids(str(raw))
        if ids:
            return ids
    count = _visible_gpu_count()
    return [str(i) for i in range(count)]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top-level: {path}")
    return data


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_json(path: Path, data: Any) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> str:
        try:
            if isinstance(obj, Path):
                return str(obj)
        except Exception:
            pass
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            try:
                return obj.__dict__
            except Exception:
                pass
        return str(obj)

    with path.open("w") as f:
        json.dump(data, f, indent=2, default=_default)


def _force_wandb_off(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["report_to_wandb"] = False
    return out


def _vary_seeds(cfg: Dict[str, Any], repeat_idx: int) -> Dict[str, Any]:
    out = dict(cfg)

    def _base_int(key: str, default: int) -> int:
        val = out.get(key, default)
        try:
            return int(val)
        except Exception:
            return int(default)

    base_shuffle = _base_int("shuffle_seed", 42)
    out["shuffle_seed"] = base_shuffle + int(repeat_idx)
    return out


def _resolve_base_checkpoint_dir(
    cfg: Dict[str, Any],
    config_path: Path,
    checkpoint_base_dir_override: Optional[Path],
) -> Path:
    if checkpoint_base_dir_override is not None:
        return checkpoint_base_dir_override

    cfg_val = cfg.get("checkpoint_base_dir")
    if isinstance(cfg_val, str) and cfg_val.strip():
        p = Path(cfg_val).expanduser()
        if not p.is_absolute():
            p = (config_path.parent / p).resolve()
        return p

    # Back-compat: if the config only has a concrete checkpoint_path, infer the base
    # directory from its parent.
    ckpt_path = cfg.get("checkpoint_path")
    if isinstance(ckpt_path, str) and ckpt_path.strip():
        p = Path(ckpt_path).expanduser()
        if not p.is_absolute():
            p = (config_path.parent / p).resolve()
        return p.parent

    return Path("/common/users/yl2310/MultiLLMs/checkpoints")


def _repeat_checkpoint_dir(base: Path, repeat_idx: int, run_tag: str) -> Path:
    return base / run_tag / f"repeat_{repeat_idx:04d}"


def _wandb_disabled_env(base_env: Dict[str, str]) -> Dict[str, str]:
    env = dict(base_env)
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_SILENT", "true")
    return env


def _find_analytics_in_repeats_dir(repeat_dir: Path) -> List[Path]:
    """Resolve eval analytics under repeat_dir or repeat_dir/<run_subdir>/."""
    eval_dir = repeat_dir / "eval"
    if eval_dir.is_dir():
        combined = eval_dir / "analytics_all.csv"
        if combined.exists():
            return [combined]
        per_ds = sorted(eval_dir.glob("analytics_*.csv"))
        if per_ds:
            return per_ds
    nested_all = sorted(repeat_dir.glob("**/eval/analytics_all.csv"))
    if nested_all:
        return nested_all
    return sorted(repeat_dir.glob("**/eval/analytics_*.csv"))


def collect_analytics_paths(repeat_root: Path) -> List[Path]:
    paths: List[Path] = []
    subdirs = sorted(repeat_root.glob("repeat_*"))
    subdirs = [p for p in subdirs if p.is_dir()]
    if subdirs:
        for d in subdirs:
            paths.extend(_find_analytics_in_repeats_dir(d))
        return paths
    paths.extend(_find_analytics_in_repeats_dir(repeat_root))
    return paths


def aggregate_eval_repeats(
    repeat_root: Path,
    output: Optional[Path],
    metrics: List[str],
) -> int:
    import pandas as pd

    repeat_root = repeat_root.resolve()
    if not repeat_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {repeat_root}")

    analytics_files = collect_analytics_paths(repeat_root)
    if not analytics_files:
        raise FileNotFoundError(
            f"No analytics CSVs found under {repeat_root}. "
            "Expected repeat_*/eval/analytics_all.csv (or repeat_*/*/eval/... when checkpoints "
            "use a run subdirectory) or analytics_*.csv under eval/."
        )

    dfs: List = []
    for fp in analytics_files:
        df = pd.read_csv(fp)
        df["_source_file"] = str(fp)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    keep_cols = [
        c
        for c in combined.columns
        if c in metrics
        or c
        in (
            "evaluation_dataset",
            "settings_hash",
            "seed",
            "_source_file",
            "wagering_method",
            "aggregation_method",
            "checkpoint_path",
        )
    ]
    keep_cols = [c for c in keep_cols if c in combined.columns]
    per_repeat = combined[keep_cols].copy()
    per_repeat_path = repeat_root / "aggregate_eval_per_repeat.csv"
    per_repeat.to_csv(per_repeat_path, index=False)

    result_cols = [c for c in metrics if c in combined.columns]
    if not result_cols:
        raise ValueError(
            "None of the requested metrics exist in the loaded CSVs. "
            f"Requested: {metrics}. Columns present: {list(combined.columns)}"
        )

    agg = _aggregate_eval_across_repeats(combined, result_cols)
    agg = _add_mean_ci95_columns(agg, result_cols)

    out_path = output if output is not None else (repeat_root / "aggregated_eval_repeats.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)

    print(f"[aggregate] Loaded {len(analytics_files)} analytics file(s)", flush=True)
    print(f"[aggregate] Per-repeat rows: {len(per_repeat)} -> {per_repeat_path}", flush=True)
    print(f"[aggregate] Aggregated: {agg.shape[0]} row(s) -> {out_path}", flush=True)

    for _, row in agg.iterrows():
        prefix = ""
        if len(agg) > 1 and "evaluation_dataset" in agg.columns:
            prefix = f"[{row['evaluation_dataset']}] "
        for col in metrics:
            mean_col = col
            std_col = f"{col}_std"
            lo_col = f"{col}_ci95_low"
            hi_col = f"{col}_ci95_high"
            if mean_col not in agg.columns:
                continue
            st = row[std_col] if std_col in agg.columns else float("nan")
            lo = row[lo_col] if lo_col in agg.columns else float("nan")
            hi = row[hi_col] if hi_col in agg.columns else float("nan")
            line = _format_aggregate_metric_latex_line(
                col,
                float(row[mean_col]),
                st,
                lo,
                hi,
                row.get("num_runs"),
            )
            print(f"  {prefix}{line}", flush=True)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeat-run wagering_pipeline.py (wandb off) and optionally aggregate eval CSVs"
    )
    parser.add_argument(
        "--config",
        nargs="?",
        default=None,
        type=str,
        help="Path to config YAML (not used with --aggregate-only; used with --aggregate-run-tag)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of times to repeat (required unless --aggregate-only or --aggregate-run-tag)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max concurrent repeats (default: n_repeats; not with --max-workers-per-gpu)",
    )
    parser.add_argument(
        "--max-workers-per-gpu",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Run at most K pipeline jobs per visible GPU; sets CUDA_VISIBLE_DEVICES per "
            "subprocess. Max concurrent = min(n_repeats, K * num_gpus). Incompatible with "
            "--max-workers."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids to use (e.g. 0,1,2,3). Limits scheduling for "
            "--max-workers-per-gpu and sets CUDA_VISIBLE_DEVICES for child runs."
        ),
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python for pipeline subprocess (default: ./.venv/bin/python if present)",
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=str,
        default=None,
        help="Base checkpoint directory; each repeat uses <base>/<run_tag>/repeat_NNNN/",
    )
    parser.add_argument("--skip-training", action="store_true", help="Pass through to pipeline")
    parser.add_argument("--skip-evaluation", action="store_true", help="Pass through to pipeline")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Pass through to pipeline (with --skip-training)",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between repeats (sequential mode only)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop submitting new repeats on first failure (parallel mode drains in-flight)")
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip aggregating eval analytics after repeats",
    )
    parser.add_argument(
        "--aggregate-output",
        type=str,
        default=None,
        help="Write aggregated eval CSV here (default: <run_tag_dir>/aggregated_eval_repeats.csv)",
    )
    parser.add_argument(
        "--aggregate-metrics",
        type=str,
        default=",".join(DEFAULT_AGGREGATE_METRICS),
        help="Comma-separated metric columns for aggregation",
    )
    parser.add_argument(
        "--aggregate-only",
        type=str,
        default=None,
        metavar="REPEAT_ROOT",
        help=(
            "Only aggregate eval CSVs under this directory (the <run_tag> folder containing "
            "repeat_*/); do not run the pipeline"
        ),
    )
    parser.add_argument(
        "--aggregate-run-tag",
        type=str,
        default=None,
        metavar="TAG",
        help=(
            "With --config, only aggregate from <checkpoint_base_dir>/<TAG>/repeat_* using the "
            "config's checkpoint_base_dir (and optional --checkpoint-base-dir override). "
            "Does not run the pipeline; do not pass --n-repeats."
        ),
    )

    args = parser.parse_args()

    if args.aggregate_only:
        if args.aggregate_run_tag:
            parser.error("Use either --aggregate-only or --aggregate-run-tag, not both")
        metrics = [m.strip() for m in args.aggregate_metrics.split(",") if m.strip()]
        out = Path(args.aggregate_output) if args.aggregate_output else None
        return aggregate_eval_repeats(Path(args.aggregate_only), out, metrics)

    if not args.config:
        parser.error("config YAML is required unless --aggregate-only is set")

    if args.aggregate_run_tag:
        if args.n_repeats is not None:
            parser.error("--aggregate-run-tag is aggregation-only; omit --n-repeats")
        cfg = _load_yaml(Path(args.config))
        checkpoint_base_dir_override = Path(args.checkpoint_base_dir) if args.checkpoint_base_dir else None
        base_ckpt_dir = _resolve_base_checkpoint_dir(
            cfg, Path(args.config), checkpoint_base_dir_override
        )
        repeat_root = base_ckpt_dir / args.aggregate_run_tag
        metrics = [m.strip() for m in args.aggregate_metrics.split(",") if m.strip()]
        out = Path(args.aggregate_output) if args.aggregate_output else None
        return aggregate_eval_repeats(repeat_root, out, metrics)

    if args.n_repeats is None:
        parser.error("--n-repeats is required unless --aggregate-only or --aggregate-run-tag is set")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if args.n_repeats <= 0:
        raise ValueError("--n-repeats must be > 0")
    if not PIPELINE_SCRIPT.exists():
        raise FileNotFoundError(f"Pipeline script not found: {PIPELINE_SCRIPT}")

    if args.max_workers is not None and args.max_workers_per_gpu is not None:
        parser.error("Use either --max-workers or --max-workers-per-gpu, not both")

    gpu_slot_queue: Optional[Queue[str]] = None
    visible_gpu_ids = _resolve_visible_gpu_ids(args.gpus)
    if args.gpus is not None:
        print(f"[schedule] Using explicit GPU ids: {','.join(visible_gpu_ids)}", flush=True)
    if args.max_workers_per_gpu is not None:
        if args.max_workers_per_gpu <= 0:
            raise ValueError("--max-workers-per-gpu must be > 0")
        num_gpus = len(visible_gpu_ids)
        if num_gpus < 1:
            raise ValueError(
                "Found 0 CUDA device(s) for --max-workers-per-gpu. "
                "Set --gpus or CUDA_VISIBLE_DEVICES, or install PyTorch with CUDA, "
                "or use --max-workers instead."
            )
        args.max_workers = min(int(args.n_repeats), num_gpus * int(args.max_workers_per_gpu))
        gpu_slot_queue = Queue()
        for g in visible_gpu_ids:
            for _ in range(int(args.max_workers_per_gpu)):
                gpu_slot_queue.put(g)
        print(
            f"[schedule] {num_gpus} visible GPU(s) ({','.join(visible_gpu_ids)}), "
            f"{args.max_workers_per_gpu} worker(s)/GPU -> max concurrent repeats {args.max_workers}",
            flush=True,
        )
    elif args.max_workers is None:
        args.max_workers = int(args.n_repeats)

    cfg = _load_yaml(config_path)
    cfg = _force_wandb_off(cfg)

    checkpoint_base_dir_override = Path(args.checkpoint_base_dir) if args.checkpoint_base_dir else None
    base_ckpt_dir = _resolve_base_checkpoint_dir(cfg, config_path, checkpoint_base_dir_override)
    wandb_name = cfg.get("wandb_name")
    if isinstance(wandb_name, str) and wandb_name.strip():
        run_tag = wandb_name.strip()
    else:
        run_tag = time.strftime("%Y%m%d_%H%M%S")

    run_tag_dir = base_ckpt_dir / run_tag
    if run_tag_dir.exists():
        # Avoid clobbering a previous run with the same wandb_name.
        run_tag = f"{run_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_tag_dir = base_ckpt_dir / run_tag
    run_tag_dir.mkdir(parents=True, exist_ok=True)

    # Persist the repeat-run config inputs for reproducibility.
    try:
        _dump_yaml(run_tag_dir / "repeat.base_config.original.yaml", _load_yaml(config_path))
        _dump_yaml(run_tag_dir / "repeat.base_config.mutated.yaml", cfg)
        _write_json(
            run_tag_dir / "repeat.run_manifest.json",
            {
                "config_path": str(config_path),
                "run_tag": run_tag,
                        "wandb_name": wandb_name,
                "checkpoint_base_dir": str(base_ckpt_dir),
                "run_tag_dir": str(run_tag_dir),
                "n_repeats": int(args.n_repeats),
                "max_workers": int(args.max_workers),
                "max_workers_per_gpu": int(args.max_workers_per_gpu) if args.max_workers_per_gpu is not None else None,
                "gpus_arg": args.gpus,
                "visible_gpu_ids": visible_gpu_ids,
                "skip_training": bool(args.skip_training),
                "skip_evaluation": bool(args.skip_evaluation),
                "checkpoint_path_override": args.checkpoint_path,
            },
        )
    except Exception:
        pass

    env = _wandb_disabled_env(os.environ)
    if visible_gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpu_ids)

    python_exe = Path(args.python) if args.python else (DEFAULT_VENV_PYTHON if DEFAULT_VENV_PYTHON.exists() else Path(sys.executable))
    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0")

    def _run_one(repeat_idx: int) -> int:
        repeat_ckpt_dir = _repeat_checkpoint_dir(base_ckpt_dir, repeat_idx, run_tag)
        repeat_cfg = _vary_seeds(cfg, repeat_idx)
        repeat_cfg["checkpoint_base_dir"] = str(repeat_ckpt_dir)
        # Ensure eval artifacts land inside each repeat directory.
        # For non-trainable methods (no checkpoint required), wagering_eval.py chooses
        # eval output dir based on (eval_checkpoint_dir) or (checkpoint_path/eval).
        repeat_cfg.setdefault("eval_checkpoint_dir", str(repeat_ckpt_dir / "eval"))
        if args.checkpoint_path is None:
            repeat_cfg.setdefault("checkpoint_path", str(repeat_ckpt_dir))

        temp_cfg_path = config_path.parent / f".tmp_wagering_repeat_{run_tag}_{repeat_idx:04d}.yaml"
        repeat_cfg_logged = run_tag_dir / f"repeat_{repeat_idx:04d}.config.yaml"

        gpu_index: Optional[str] = None
        run_env = dict(env)
        if gpu_slot_queue is not None:
            gpu_index = str(gpu_slot_queue.get())
            run_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        try:
            _dump_yaml(temp_cfg_path, repeat_cfg)
            try:
                _dump_yaml(repeat_cfg_logged, repeat_cfg)
            except Exception:
                pass

            cmd = [str(python_exe), str(PIPELINE_SCRIPT), str(temp_cfg_path)]
            if args.skip_training:
                cmd.append("--skip-training")
            if args.skip_evaluation:
                cmd.append("--skip-evaluation")
            if args.checkpoint_path is not None:
                cmd.extend(["--checkpoint-path", args.checkpoint_path])

            gpu_note = f" CUDA_VISIBLE_DEVICES={gpu_index}" if gpu_index is not None else ""
            print(f"[repeat {repeat_idx+1}/{args.n_repeats}] running:{gpu_note} {' '.join(cmd)}", flush=True)
            print(f"[repeat {repeat_idx+1}/{args.n_repeats}] checkpoint_base_dir: {repeat_ckpt_dir}", flush=True)

            completed = subprocess.run(cmd, env=run_env)
            return int(completed.returncode)
        finally:
            if gpu_slot_queue is not None and gpu_index is not None:
                gpu_slot_queue.put(gpu_index)
            try:
                if temp_cfg_path.exists():
                    temp_cfg_path.unlink()
            except Exception:
                pass

    failures = 0
    fail_fast_stop = False
    if args.max_workers == 1:
        for i in range(args.n_repeats):
            rc = _run_one(i)
            if rc != 0:
                failures += 1
                print(f"[repeat {i+1}/{args.n_repeats}] FAILED with exit code {rc}", flush=True)
                if args.fail_fast:
                    fail_fast_stop = True
                    break
            else:
                print(f"[repeat {i+1}/{args.n_repeats}] OK", flush=True)
            if args.sleep_seconds and i < args.n_repeats - 1 and not fail_fast_stop:
                time.sleep(args.sleep_seconds)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            pending = {}
            next_idx = 0
            stop_submitting = False

            def _submit_one(idx: int) -> None:
                fut = ex.submit(_run_one, idx)
                pending[fut] = idx

            while next_idx < args.n_repeats and len(pending) < args.max_workers:
                _submit_one(next_idx)
                next_idx += 1

            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = pending.pop(fut)
                    rc = int(fut.result())
                    if rc != 0:
                        failures += 1
                        print(f"[repeat {idx+1}/{args.n_repeats}] FAILED with exit code {rc}", flush=True)
                        if args.fail_fast:
                            stop_submitting = True
                    else:
                        print(f"[repeat {idx+1}/{args.n_repeats}] OK", flush=True)

                    if args.sleep_seconds and next_idx < args.n_repeats and not stop_submitting:
                        time.sleep(args.sleep_seconds)

                    if not stop_submitting and next_idx < args.n_repeats:
                        _submit_one(next_idx)
                        next_idx += 1

    if failures:
        print(f"Completed with {failures} failure(s) out of {args.n_repeats} repeats.", flush=True)

    if not args.no_aggregate and not args.skip_evaluation:
        try:
            metrics = [m.strip() for m in args.aggregate_metrics.split(",") if m.strip()]
            out = Path(args.aggregate_output) if args.aggregate_output else None
            aggregate_eval_repeats(run_tag_dir, out, metrics)
            # Also aggregate the JSON metrics we write per repeat.
            try:
                agg_json = _aggregate_eval_metrics_json(run_tag_dir, metrics)
                if agg_json is not None:
                    metrics_dir = run_tag_dir / "pipeline_artifacts"
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(metrics_dir / "eval.metrics.json", agg_json)

                    # Console: print in the same style as the CSV aggregator snippet:
                    #   accuracy: 94.68\tiny{$\pm$0.00}  $n=2$
                    #     d_regret: ...
                    def _print_block(overall_dict: Any) -> None:
                        if not isinstance(overall_dict, dict) or not overall_dict:
                            return
                        first = True
                        for m in metrics:
                            row = overall_dict.get(m)
                            if not isinstance(row, dict) or "formatted" not in row:
                                continue
                            prefix = "" if first else "  "
                            print(f"{prefix}{m}: {row['formatted']}", flush=True)
                            first = False

                    def _print_per_dataset(section: Any) -> bool:
                        """Print per-dataset blocks; return True if printed any."""
                        if not isinstance(section, dict):
                            return False
                        per_ds = section.get("per_dataset", {})
                        if not isinstance(per_ds, dict) or not per_ds:
                            return False
                        # Only print per-dataset breakdown when multiple datasets exist.
                        if len(per_ds) <= 1:
                            return False
                        for ds_name, ds_blob in per_ds.items():
                            if not isinstance(ds_blob, dict):
                                continue
                            print(f"[{ds_name}]", flush=True)
                            _print_block(ds_blob)
                        return True

                    full_section = agg_json.get("full", {})
                    printed_full_per_ds = _print_per_dataset(full_section)
                    if not printed_full_per_ds:
                        full_overall = full_section.get("overall", {}) if isinstance(full_section, dict) else {}
                        _print_block(full_overall)

                    subset_section = agg_json.get("subset_any_model_wrong", {})
                    subset_has_any = isinstance(subset_section, dict) and bool(subset_section)
                    if subset_has_any:
                        print("subset_any_model_wrong", flush=True)
                        printed_subset_per_ds = _print_per_dataset(subset_section)
                        if not printed_subset_per_ds:
                            subset_overall = subset_section.get("overall", {})
                            _print_block(subset_overall)
            except Exception as e_json:
                print(f"[aggregate json] skipped or failed: {e_json}", flush=True)
            try:
                _write_json(
                    run_tag_dir / "repeat.aggregate_manifest.json",
                    {
                        "repeat_root": str(run_tag_dir),
                        "aggregate_output": str(out) if out is not None else str(run_tag_dir / "aggregated_eval_repeats.csv"),
                        "aggregate_metrics": metrics,
                        "per_repeat_csv": str(run_tag_dir / "aggregate_eval_per_repeat.csv"),
                        "aggregate_eval_metrics_json": str((run_tag_dir / "pipeline_artifacts" / "eval.metrics.json")),
                    },
                )
            except Exception:
                pass
        except Exception as e:
            print(f"[aggregate] skipped or failed: {e}", flush=True)

    if failures:
        return 1

    print(f"All {args.n_repeats} repeats completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
