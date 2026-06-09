#!/usr/bin/env python3
"""
Repeat-run wrapper for scripts/wagering_pipeline.py.

Runs the same config N times (parallel by default), forces wandb off, varies
shuffle_seed per repeat, and gives each repeat its own checkpoint directory.

After repeats, aggregates eval.metrics.json across repeats (mean ± 95%% CI).

Usage:
  ./.venv/bin/python scripts/wagering_pipeline_repeat.py --config path/to/config.yaml --n-repeats 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "wagering_pipeline.py"
DEFAULT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.utils import load_and_merge_configs
from wagering.utils.config_utils import load_yaml_file
from wagering.utils.model_prep import require_pipeline_caches_for_repeat
from wagering.utils.repeat_metrics import (
    DEFAULT_AGGREGATE_METRICS,
    aggregate_eval_metrics_json,
    print_aggregated_metrics,
    write_json,
)
from wagering.utils.script_runtime import (
    ParallelGpuRunner,
    dump_yaml,
    ensure_project_venv,
    require_visible_gpu_ids,
    run_subprocess,
    vary_shuffle_seed,
    wandb_disabled_env,
)

ensure_project_venv()


def _force_wandb_off(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["report_to_wandb"] = False
    return out


def _resolve_base_checkpoint_dir(
    cfg: Dict[str, Any],
    config_path: Path,
    checkpoint_base_dir_override: Optional[Path],
) -> Path:
    if checkpoint_base_dir_override is not None:
        return checkpoint_base_dir_override

    cfg_val = cfg.get("checkpoint_base_dir")
    if not isinstance(cfg_val, str) or not cfg_val.strip():
        raise ValueError(
            "Config must set checkpoint_base_dir. Repeat runs write under "
            "<checkpoint_base_dir>/<run_tag>/repeat_XXXX/."
        )
    path = Path(cfg_val).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _repeat_checkpoint_dir(base: Path, repeat_idx: int, run_tag: str) -> Path:
    return base / run_tag / f"repeat_{repeat_idx:04d}"


def aggregate_repeat_root(repeat_root: Path, metrics: List[str]) -> int:
    repeat_root = repeat_root.resolve()
    if not repeat_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {repeat_root}")

    agg = aggregate_eval_metrics_json(repeat_root, metrics)
    if agg is None:
        raise FileNotFoundError(
            f"No eval.metrics.json found under {repeat_root}. "
            "Expected repeat_*/**/pipeline_artifacts/eval.metrics.json."
        )

    metrics_dir = repeat_root / "pipeline_artifacts"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "eval.metrics.json", agg)

    print_aggregated_metrics(agg, metrics, "full")
    print("", flush=True)
    print_aggregated_metrics(agg, metrics, "subset_any_model_wrong")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeat-run wagering_pipeline.py (wandb off) and aggregate eval.metrics.json"
    )
    parser.add_argument("--config", nargs="?", default=None, type=str)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-workers-per-gpu", type=int, default=None, metavar="K")
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--checkpoint-base-dir", type=str, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument(
        "--aggregate-metrics",
        type=str,
        default=",".join(DEFAULT_AGGREGATE_METRICS),
    )
    parser.add_argument("--aggregate-only", type=str, default=None, metavar="REPEAT_ROOT")
    parser.add_argument("--aggregate-run-tag", type=str, default=None, metavar="TAG")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.aggregate_metrics.split(",") if m.strip()]

    if args.aggregate_only:
        if args.aggregate_run_tag:
            parser.error("Use either --aggregate-only or --aggregate-run-tag, not both")
        return aggregate_repeat_root(Path(args.aggregate_only), metrics)

    if not args.config:
        parser.error("config YAML is required unless --aggregate-only is set")

    if args.aggregate_run_tag:
        if args.n_repeats is not None:
            parser.error("--aggregate-run-tag is aggregation-only; omit --n-repeats")
        cfg = load_yaml_file(Path(args.config))
        base_ckpt_dir = _resolve_base_checkpoint_dir(
            cfg,
            Path(args.config),
            Path(args.checkpoint_base_dir) if args.checkpoint_base_dir else None,
        )
        return aggregate_repeat_root(base_ckpt_dir / args.aggregate_run_tag, metrics)

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

    visible_gpu_ids = require_visible_gpu_ids(args.gpus)
    if args.gpus is not None:
        print(f"[schedule] Using explicit GPU ids: {','.join(visible_gpu_ids)}", flush=True)

    max_workers = int(args.n_repeats)
    gpu_runner: Optional[ParallelGpuRunner] = None
    if args.max_workers_per_gpu is not None:
        if args.max_workers_per_gpu <= 0:
            raise ValueError("--max-workers-per-gpu must be > 0")
        max_workers = min(int(args.n_repeats), len(visible_gpu_ids) * int(args.max_workers_per_gpu))
        gpu_runner = ParallelGpuRunner(
            gpu_ids=visible_gpu_ids,
            max_workers_per_gpu=int(args.max_workers_per_gpu),
            max_jobs=int(args.n_repeats),
        )
        print(
            f"[schedule] {len(visible_gpu_ids)} visible GPU(s) ({','.join(visible_gpu_ids)}), "
            f"{args.max_workers_per_gpu} worker(s)/GPU -> max concurrent repeats {max_workers}",
            flush=True,
        )
    elif args.max_workers is not None:
        max_workers = int(args.max_workers)
    if max_workers <= 0:
        raise ValueError("--max-workers must be > 0")

    cfg = _force_wandb_off(load_and_merge_configs(config_path))
    require_pipeline_caches_for_repeat(
        cfg,
        skip_training=bool(args.skip_training),
        skip_evaluation=bool(args.skip_evaluation),
    )
    base_ckpt_dir = _resolve_base_checkpoint_dir(
        cfg,
        config_path,
        Path(args.checkpoint_base_dir) if args.checkpoint_base_dir else None,
    )
    wandb_name = cfg.get("wandb_name")
    run_tag = wandb_name.strip() if isinstance(wandb_name, str) and wandb_name.strip() else time.strftime("%Y%m%d_%H%M%S")
    run_tag_dir = base_ckpt_dir / run_tag
    if run_tag_dir.exists():
        run_tag = f"{run_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_tag_dir = base_ckpt_dir / run_tag
    run_tag_dir.mkdir(parents=True, exist_ok=True)

    dump_yaml(run_tag_dir / "repeat.base_config.original.yaml", load_yaml_file(config_path))
    dump_yaml(run_tag_dir / "repeat.base_config.mutated.yaml", cfg)
    write_json(
        run_tag_dir / "repeat.run_manifest.json",
        {
            "config_path": str(config_path),
            "run_tag": run_tag,
            "wandb_name": wandb_name,
            "checkpoint_base_dir": str(base_ckpt_dir),
            "run_tag_dir": str(run_tag_dir),
            "n_repeats": int(args.n_repeats),
            "max_workers": max_workers,
            "max_workers_per_gpu": int(args.max_workers_per_gpu) if args.max_workers_per_gpu is not None else None,
            "gpus_arg": args.gpus,
            "visible_gpu_ids": visible_gpu_ids,
            "skip_training": bool(args.skip_training),
            "skip_evaluation": bool(args.skip_evaluation),
            "checkpoint_path_override": args.checkpoint_path,
        },
    )

    env = wandb_disabled_env(os.environ)
    if visible_gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpu_ids)

    python_exe = Path(args.python) if args.python else (
        DEFAULT_VENV_PYTHON if DEFAULT_VENV_PYTHON.exists() else Path(sys.executable)
    )
    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")

    def _run_one(repeat_idx: int, gpu_index: Optional[str] = None) -> int:
        repeat_ckpt_dir = _repeat_checkpoint_dir(base_ckpt_dir, repeat_idx, run_tag)
        repeat_cfg = vary_shuffle_seed(cfg, repeat_idx)
        repeat_cfg["checkpoint_base_dir"] = str(repeat_ckpt_dir)
        repeat_cfg["eval_checkpoint_dir"] = str(repeat_ckpt_dir / "eval")
        if args.checkpoint_path is None:
            repeat_cfg["checkpoint_path"] = str(repeat_ckpt_dir)

        temp_cfg_path = config_path.parent / f".tmp_wagering_repeat_{run_tag}_{repeat_idx:04d}.yaml"
        repeat_cfg_logged = run_tag_dir / f"repeat_{repeat_idx:04d}.config.yaml"
        run_env = dict(env)
        if gpu_index is not None:
            run_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        dump_yaml(temp_cfg_path, repeat_cfg)
        dump_yaml(repeat_cfg_logged, repeat_cfg)

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
        rc = run_subprocess(cmd, env=run_env, cwd=PROJECT_ROOT)
        if temp_cfg_path.exists():
            temp_cfg_path.unlink()
        return rc

    failures = 0
    jobs = list(range(int(args.n_repeats)))

    if gpu_runner is not None and max_workers > 1:

        def _on_complete(job: int, rc: int) -> None:
            if rc != 0:
                print(f"[repeat {job+1}/{args.n_repeats}] FAILED with exit code {rc}", flush=True)
            else:
                print(f"[repeat {job+1}/{args.n_repeats}] OK", flush=True)

        failures = gpu_runner.run_all(
            jobs,
            lambda repeat_idx, gpu_id: _run_one(repeat_idx, gpu_id),
            fail_fast=args.fail_fast,
            on_complete=_on_complete,
        )
    elif max_workers == 1:
        for repeat_idx in jobs:
            rc = _run_one(repeat_idx)
            if rc != 0:
                failures += 1
                print(f"[repeat {repeat_idx+1}/{args.n_repeats}] FAILED with exit code {rc}", flush=True)
                if args.fail_fast:
                    break
            else:
                print(f"[repeat {repeat_idx+1}/{args.n_repeats}] OK", flush=True)
            if args.sleep_seconds and repeat_idx < args.n_repeats - 1:
                time.sleep(args.sleep_seconds)
    else:
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            pending = {}
            next_idx = 0
            stop_submitting = False

            def _submit(idx: int) -> None:
                pending[ex.submit(_run_one, idx)] = idx

            while next_idx < args.n_repeats and len(pending) < max_workers:
                _submit(next_idx)
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
                        _submit(next_idx)
                        next_idx += 1

    if failures:
        print(f"Completed with {failures} failure(s) out of {args.n_repeats} repeats.", flush=True)

    if not args.no_aggregate and not args.skip_evaluation:
        aggregate_repeat_root(run_tag_dir, metrics)
        write_json(
            run_tag_dir / "repeat.aggregate_manifest.json",
            {
                "repeat_root": str(run_tag_dir),
                "aggregate_metrics": metrics,
                "aggregate_eval_metrics_json": str(run_tag_dir / "pipeline_artifacts" / "eval.metrics.json"),
            },
        )

    if failures:
        return 1
    print(f"All {args.n_repeats} repeats completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
