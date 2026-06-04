#!/usr/bin/env python3
"""
Run calibration comparison (uncalibrated vs Gemma2-only calibrated) with N repeats per arm,
in parallel across GPUs (same scheduling idea as wagering_pipeline_repeat.py).

Arms (4 total):
  - MMLU uncalibrated / MMLU Gemma2-calibrated
  - MedMCQA uncalibrated / MedMCQA Gemma2-calibrated

Default: 4 repeats per arm → 16 pipeline runs; with 4 GPUs and --max-workers-per-gpu 2,
up to 8 jobs run concurrently.

After runs: aggregates eval.metrics.json per arm (mean ± 95%% CI, Student's t) for the
full test set and subset_any_model_wrong (same print style as wagering_pipeline_repeat.py),
then writes plots with mean ± 95%% CI (no shaded fills; dashed CI bounds on line plots;
error bars on bar plots).

Usage:
  ./.venv/bin/python scripts/calibration_compare_repeats.py \\
    --out-dir /research/projects/ecoai/yl2310/MultiLLMs/artifacts/calibration_compare \\
    --n-repeats 4 --gpus 0,1,2,3 --max-workers-per-gpu 2

Default --out-dir is under /research/projects/ecoai/yl2310/MultiLLMs/artifacts/….
Relative --out-dir is resolved under MULTILLMS_OUTPUT_ROOT (not your cwd), so checkpoints avoid home quota.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("MULTILLMS_OUTPUT_ROOT", "/research/projects/ecoai/yl2310/MultiLLMs")
).expanduser()
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _resolve_output_dir(path_str: str) -> Path:
    """Absolute paths as-is; relative paths are under DEFAULT_OUTPUT_ROOT (not cwd)."""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (DEFAULT_OUTPUT_ROOT / p).resolve()
if __name__ == "__main__" and _VENV_PYTHON.exists():
    try:
        if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
            os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)
    except OSError:
        pass

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "wagering_pipeline.py"
_CKPT_RE = re.compile(r"Checkpoint directory:\s*(?P<path>/.+)")

DEFAULT_METRICS = [
    "accuracy",
    "auc",
    "d_regret",
    "brier_d_regret",
    "kendall_tau",
    "best_model_mrr",
    "ece",
    "inverse_hhi",
    "avg_inference_time_per_batch_s",
]


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _import_plot_calibration():
    spec = importlib.util.spec_from_file_location(
        "_plot_cal",
        PROJECT_ROOT / "scripts" / "plot_calibration_comparison.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Required before exec_module: @dataclass looks up cls.__module__ in sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_wagering_pipeline_repeat():
    spec = importlib.util.spec_from_file_location(
        "_wpr",
        PROJECT_ROOT / "scripts" / "wagering_pipeline_repeat.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_gpu_ids(csv: str) -> List[str]:
    gpu_ids = [p.strip() for p in str(csv).split(",") if p.strip()]
    if not gpu_ids:
        raise ValueError("No GPUs provided. Example: --gpus 0,1,2,3")
    return gpu_ids


def _default_visible_gpu_ids() -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and str(raw).strip() != "":
        return _parse_gpu_ids(str(raw))
    try:
        import torch

        n = int(torch.cuda.device_count())
        if n > 0:
            return [str(i) for i in range(n)]
    except Exception:
        pass
    return ["0"]


def _wandb_disabled_env(base: Dict[str, str]) -> Dict[str, str]:
    env = dict(base)
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_SILENT", "true")
    return env


def _base_shuffle_seed(cfg: Dict[str, Any]) -> int:
    try:
        return int(cfg.get("shuffle_seed", 42))
    except Exception:
        return 42


def _print_agg_block(wpr: Any, agg_json: Dict[str, Any], metrics: List[str], section_key: str) -> None:
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
        if not isinstance(section, dict):
            return False
        per_ds = section.get("per_dataset", {})
        if not isinstance(per_ds, dict) or not per_ds:
            return False
        if len(per_ds) <= 1:
            return False
        for ds_name, ds_blob in per_ds.items():
            if not isinstance(ds_blob, dict):
                continue
            print(f"[{ds_name}]", flush=True)
            _print_block(ds_blob)
        return True

    section = agg_json.get(section_key, {})
    if section_key == "subset_any_model_wrong":
        print("subset_any_model_wrong", flush=True)
    if not isinstance(section, dict):
        return
    printed = _print_per_dataset(section)
    if not printed:
        _print_block(section.get("overall", {}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration compare with repeats + CI plots")
    parser.add_argument(
        "--mmlu-config",
        type=str,
        default=str(
            PROJECT_ROOT / "examples" / "configs" / "wagering_training" / "mse_br_wagers_v2_4models_mmlu.yaml"
        ),
    )
    parser.add_argument(
        "--medmcqa-config",
        type=str,
        default=str(
            PROJECT_ROOT / "examples" / "configs" / "wagering_training" / "mse_br_wagers_v2_4models_medmcqa.yaml"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT / "artifacts" / "calibration_compare"),
    )
    parser.add_argument("--n-repeats", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument(
        "--max-workers-per-gpu",
        type=int,
        default=2,
        metavar="K",
        help="Max concurrent pipeline jobs per GPU (default: 2)",
    )
    parser.add_argument("--python", type=str, default=None, help="Python executable (default: .venv/bin/python)")
    parser.add_argument(
        "--only-plots",
        type=str,
        default=None,
        metavar="RUN_ROOT",
        help="Skip pipeline; only load existing run_root and regenerate plots + aggregate prints",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    pcc = _import_plot_calibration()
    wpr = _import_wagering_pipeline_repeat()

    out_root = _resolve_output_dir(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    mmlu_base = Path(args.mmlu_config).expanduser().resolve()
    med_base = Path(args.medmcqa_config).expanduser().resolve()
    mmlu_uncal, mmlu_cal = pcc._make_setting_configs(mmlu_base, num_epochs=int(args.num_epochs))
    med_uncal, med_cal = pcc._make_setting_configs(med_base, num_epochs=int(args.num_epochs))

    arms: List[Tuple[str, Path]] = [
        ("MMLU_uncalibrated", mmlu_uncal),
        ("MMLU_gemma2_calibrated", mmlu_cal),
        ("MedMCQA_uncalibrated", med_uncal),
        ("MedMCQA_gemma2_calibrated", med_cal),
    ]

    if args.only_plots:
        # Relative path: under MULTILLMS_OUTPUT_ROOT; absolute: unchanged.
        run_root = _resolve_output_dir(args.only_plots)
    else:
        run_root = out_root / f"runs_{time.strftime('%Y%m%d_%H%M%S')}"
        run_root.mkdir(parents=True, exist_ok=True)

    if not args.only_plots:
        manifest: Dict[str, Any] = {
            "run_root": str(run_root),
            "arms": [{"name": n, "config": str(p)} for n, p in arms],
            "n_repeats": int(args.n_repeats),
        }
        (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    python_exe = Path(args.python) if args.python else (_VENV_PYTHON if _VENV_PYTHON.exists() else Path(sys.executable))

    n_repeats_eff = int(args.n_repeats)
    if args.only_plots:
        man_path = run_root / "manifest.json"
        if man_path.is_file():
            try:
                n_repeats_eff = int(json.loads(man_path.read_text()).get("n_repeats", n_repeats_eff))
            except Exception:
                pass

    if not args.only_plots:
        visible = _parse_gpu_ids(args.gpus) if args.gpus else _default_visible_gpu_ids()
        k = int(args.max_workers_per_gpu)
        if k <= 0:
            raise ValueError("--max-workers-per-gpu must be > 0")
        max_workers = min(len(arms) * int(args.n_repeats), len(visible) * k)
        gpu_queue: Queue[str] = Queue()
        for g in visible:
            for _ in range(k):
                gpu_queue.put(g)
        print(
            f"[schedule] {len(visible)} GPU(s), {k} worker(s)/GPU -> max concurrent {max_workers}",
            flush=True,
        )

        base_env = _wandb_disabled_env(os.environ)

        failures = 0
        stop_submit = False

        def _run_one(arm_name: str, cfg_path: Path, repeat_idx: int) -> int:
            repeat_dir = run_root / arm_name / f"repeat_{repeat_idx:04d}"
            repeat_dir.mkdir(parents=True, exist_ok=True)
            cfg = _load_yaml(cfg_path)
            cfg = dict(cfg)
            cfg["report_to_wandb"] = False
            cfg["shuffle_seed"] = _base_shuffle_seed(cfg) + int(repeat_idx)
            cfg["checkpoint_base_dir"] = str(repeat_dir)
            cfg.setdefault("eval_checkpoint_dir", str(repeat_dir / "eval"))
            cfg.setdefault("checkpoint_path", str(repeat_dir))

            tmp = cfg_path.parent / f".tmp_cal_compare_{arm_name}_{repeat_idx:04d}.yaml"
            gpu_id = str(gpu_queue.get())
            run_env = dict(base_env)
            run_env["CUDA_VISIBLE_DEVICES"] = gpu_id

            try:
                _dump_yaml(tmp, cfg)
                cmd = [str(python_exe), str(PIPELINE_SCRIPT), str(tmp)]
                print(
                    f"[run] {arm_name} repeat={repeat_idx} CUDA_VISIBLE_DEVICES={gpu_id} -> {repeat_dir}",
                    flush=True,
                )
                proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=run_env)
                return int(proc.returncode)
            finally:
                gpu_queue.put(gpu_id)
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass

        jobs: List[Tuple[str, Path, int]] = []
        for arm_name, cfg_path in arms:
            for r in range(int(args.n_repeats)):
                jobs.append((arm_name, cfg_path, r))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            pending = {}
            next_i = 0
            stop = False

            def _submit() -> None:
                nonlocal next_i
                if stop_submit or next_i >= len(jobs):
                    return
                arm_name, cfg_path, ridx = jobs[next_i]
                fut = ex.submit(_run_one, arm_name, cfg_path, ridx)
                pending[fut] = (next_i, arm_name, ridx)
                next_i += 1

            for _ in range(min(max_workers, len(jobs))):
                _submit()

            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    _, arm_name, ridx = pending.pop(fut)
                    rc = int(fut.result())
                    if rc != 0:
                        failures += 1
                        print(f"[run] FAILED {arm_name} repeat={ridx} code={rc}", flush=True)
                        if args.fail_fast:
                            stop_submit = True
                    else:
                        print(f"[run] OK {arm_name} repeat={ridx}", flush=True)
                    _submit()

        if failures:
            print(f"Completed with {failures} failure(s).", flush=True)
            return 1

    metrics = DEFAULT_METRICS
    print("\n=== Aggregated eval metrics (from eval.metrics.json per repeat) ===\n", flush=True)
    for arm_name, _ in arms:
        arm_dir = run_root / arm_name
        if not arm_dir.is_dir():
            continue
        agg = wpr._aggregate_eval_metrics_json(arm_dir, metrics)
        if agg is None:
            print(f"[{arm_name}] (no eval.metrics.json found)\n", flush=True)
            continue
        print(f"[{arm_name}]", flush=True)
        _print_agg_block(wpr, agg, metrics, "full")
        print("", flush=True)
        _print_agg_block(wpr, agg, metrics, "subset_any_model_wrong")
        print("", flush=True)
        try:
            agg_path = arm_dir / "pipeline_artifacts" / "eval.metrics.aggregated.json"
            agg_path.parent.mkdir(parents=True, exist_ok=True)
            wpr._write_json(agg_path, agg)
        except Exception:
            pass

    from wagering.utils import load_and_merge_configs

    for base_cfg_path, tag in [(mmlu_base, "MMLU"), (med_base, "MedMCQA")]:
        merged = load_and_merge_configs(base_cfg_path)
        model_names = pcc._display_names_for_models(merged)
        eval_ds = pcc._infer_primary_test_display_name(merged)
        unc_arm = f"{tag}_uncalibrated"
        cal_arm = f"{tag}_gemma2_calibrated"
        runs_unc = pcc.discover_run_paths_under_repeat_arm(run_root / unc_arm)
        runs_cal = pcc.discover_run_paths_under_repeat_arm(run_root / cal_arm)
        if len(runs_unc) != n_repeats_eff or len(runs_cal) != n_repeats_eff:
            print(
                f"[plots {tag}] expected {n_repeats_eff} repeats each; got uncal={len(runs_unc)} cal={len(runs_cal)}",
                flush=True,
            )
        if not runs_unc or not runs_cal:
            print(f"[plots {tag}] skip (missing runs)", flush=True)
            continue
        ds_out = out_root / tag
        ds_out.mkdir(parents=True, exist_ok=True)
        pcc._plot_training_curves_with_ci(
            out_dir=ds_out,
            title_prefix=tag,
            model_display_names=model_names,
            runs_a=runs_unc,
            runs_b=runs_cal,
            window=int(args.smooth_window),
        )
        pcc._plot_test_bars_with_ci(
            out_dir=ds_out,
            title_prefix=tag,
            model_display_names=model_names,
            eval_dataset_name=eval_ds,
            runs_a=runs_unc,
            runs_b=runs_cal,
        )
        pcc.print_test_bar_metrics_latex(
            title_prefix=tag,
            eval_dataset_name=eval_ds,
            model_display_names=model_names,
            runs_unc=runs_unc,
            runs_cal=runs_cal,
            out_txt=ds_out / f"{tag}_plot_bars_latex_metrics.txt",
        )

    print(f"\nDone. Plots under: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
