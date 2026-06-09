#!/usr/bin/env python3
"""
Run calibration comparison (uncalibrated vs Gemma2-only calibrated) with N repeats per arm,
in parallel across GPUs.

Arms (4 total):
  - MMLU uncalibrated / MMLU Gemma2-calibrated
  - MedMCQA uncalibrated / MedMCQA Gemma2-calibrated

After runs: aggregates eval.metrics.json per arm (mean ± 95%% CI).

Usage:
  ./.venv/bin/python scripts/calibration_compare_repeats.py \\
    --out-dir /research/projects/ecoai/yl2310/WALLA/artifacts/calibration_compare \\
    --n-repeats 4 --gpus 0,1,2,3 --max-workers-per-gpu 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "wagering_pipeline.py"
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.utils.calibration_compare import DEFAULT_OUTPUT_ROOT, resolve_output_dir
from wagering.utils.config_utils import load_yaml_file
from wagering.utils.repeat_metrics import (
    DEFAULT_AGGREGATE_METRICS,
    aggregate_eval_metrics_json,
    make_calibrated_config_variants,
    print_aggregated_metrics,
    write_json,
)
from wagering.utils.script_runtime import (
    ParallelGpuRunner,
    dump_yaml,
    ensure_project_venv,
    require_visible_gpu_ids,
    run_subprocess,
    wandb_disabled_env,
)

ensure_project_venv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration compare with repeats + JSON aggregation")
    parser.add_argument(
        "--mmlu-config",
        type=str,
        default=str(
            PROJECT_ROOT / "examples" / "configs" / "wagering_training" / "walla_v1_4models_mmlu.yaml"
        ),
    )
    parser.add_argument(
        "--medmcqa-config",
        type=str,
        default=str(
            PROJECT_ROOT / "examples" / "configs" / "wagering_training" / "walla_v1_4models_medmcqa.yaml"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT / "artifacts" / "calibration_compare"),
    )
    parser.add_argument("--n-repeats", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--max-workers-per-gpu", type=int, default=2, metavar="K")
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument(
        "--aggregate-only",
        type=str,
        default=None,
        metavar="RUN_ROOT",
        help="Skip pipeline; only load existing run_root and print aggregated metrics",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    out_root = resolve_output_dir(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    mmlu_base = Path(args.mmlu_config).expanduser().resolve()
    med_base = Path(args.medmcqa_config).expanduser().resolve()
    cfg_out = out_root / "generated_configs"
    mmlu_uncal, mmlu_cal = make_calibrated_config_variants(
        mmlu_base, num_epochs=int(args.num_epochs), out_dir=cfg_out
    )
    med_uncal, med_cal = make_calibrated_config_variants(
        med_base, num_epochs=int(args.num_epochs), out_dir=cfg_out
    )

    arms: List[Tuple[str, Path]] = [
        ("MMLU_uncalibrated", mmlu_uncal),
        ("MMLU_gemma2_calibrated", mmlu_cal),
        ("MedMCQA_uncalibrated", med_uncal),
        ("MedMCQA_gemma2_calibrated", med_cal),
    ]
    metrics = list(DEFAULT_AGGREGATE_METRICS)

    if args.aggregate_only:
        run_root = resolve_output_dir(args.aggregate_only)
    else:
        run_root = out_root / f"runs_{time.strftime('%Y%m%d_%H%M%S')}"
        run_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_root": str(run_root),
            "arms": [{"name": name, "config": str(path)} for name, path in arms],
            "n_repeats": int(args.n_repeats),
        }
        (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if not args.aggregate_only:
        visible = require_visible_gpu_ids(args.gpus)
        k = int(args.max_workers_per_gpu)
        if k <= 0:
            raise ValueError("--max-workers-per-gpu must be > 0")

        max_workers = min(len(arms) * int(args.n_repeats), len(visible) * k)
        runner = ParallelGpuRunner(gpu_ids=visible, max_workers_per_gpu=k, max_jobs=len(arms) * int(args.n_repeats))
        print(
            f"[schedule] {len(visible)} GPU(s), {k} worker(s)/GPU -> max concurrent {max_workers}",
            flush=True,
        )

        python_exe = Path(args.python) if args.python else (
            _VENV_PYTHON if _VENV_PYTHON.exists() else Path(sys.executable)
        )
        base_env = wandb_disabled_env(os.environ)
        jobs: List[Tuple[str, Path, int]] = []
        for arm_name, cfg_path in arms:
            for repeat_idx in range(int(args.n_repeats)):
                jobs.append((arm_name, cfg_path, repeat_idx))

        def _run_job(job: Tuple[str, Path, int], gpu_id: str) -> int:
            arm_name, cfg_path, repeat_idx = job
            repeat_dir = run_root / arm_name / f"repeat_{repeat_idx:04d}"
            repeat_dir.mkdir(parents=True, exist_ok=True)
            cfg = load_yaml_file(cfg_path)
            cfg = dict(cfg)
            cfg["report_to_wandb"] = False
            cfg["shuffle_seed"] = int(cfg.get("shuffle_seed", 42)) + int(repeat_idx)
            cfg["checkpoint_base_dir"] = str(repeat_dir)
            cfg["eval_checkpoint_dir"] = str(repeat_dir / "eval")
            cfg["checkpoint_path"] = str(repeat_dir)

            tmp = cfg_path.parent / f".tmp_cal_compare_{arm_name}_{repeat_idx:04d}.yaml"
            run_env = dict(base_env)
            run_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            dump_yaml(tmp, cfg)
            cmd = [str(python_exe), str(PIPELINE_SCRIPT), str(tmp)]
            print(
                f"[run] {arm_name} repeat={repeat_idx} CUDA_VISIBLE_DEVICES={gpu_id} -> {repeat_dir}",
                flush=True,
            )
            rc = run_subprocess(cmd, env=run_env, cwd=PROJECT_ROOT)
            if tmp.exists():
                tmp.unlink()
            return rc

        def _on_complete(job: Tuple[str, Path, int], rc: int) -> None:
            arm_name, _, repeat_idx = job
            if rc != 0:
                print(f"[run] FAILED {arm_name} repeat={repeat_idx} code={rc}", flush=True)
            else:
                print(f"[run] OK {arm_name} repeat={repeat_idx}", flush=True)

        failures = runner.run_all(jobs, _run_job, fail_fast=args.fail_fast, on_complete=_on_complete)
        if failures:
            print(f"Completed with {failures} failure(s).", flush=True)
            return 1

    print("\n=== Aggregated eval metrics (from eval.metrics.json per repeat) ===\n", flush=True)
    for arm_name, _ in arms:
        arm_dir = run_root / arm_name
        if not arm_dir.is_dir():
            continue
        agg = aggregate_eval_metrics_json(arm_dir, metrics)
        if agg is None:
            print(f"[{arm_name}] (no eval.metrics.json found)\n", flush=True)
            continue
        print(f"[{arm_name}]", flush=True)
        print_aggregated_metrics(agg, metrics, "full")
        print("", flush=True)
        print_aggregated_metrics(agg, metrics, "subset_any_model_wrong")
        print("", flush=True)
        agg_path = arm_dir / "pipeline_artifacts" / "eval.metrics.aggregated.json"
        write_json(agg_path, agg)

    print(f"\nDone. Results under: {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
