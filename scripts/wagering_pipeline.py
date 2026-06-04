#!/usr/bin/env python3
"""
End-to-end pipeline for multi-LLM wagering.

Keeps wandb run active between training and evaluation phases.

Usage: python wagering_pipeline.py <config_file.yaml>
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import argparse
from pathlib import Path
from typing import Optional
import importlib.util
import json

# Prefer running in the project venv when available.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if __name__ == "__main__" and _VENV_PYTHON.exists():
    try:
        if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
            os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)
    except OSError:
        pass

# Ensure the local src/ tree is importable
SRC_PATH = PROJECT_ROOT / "src"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))


def _configure_default_hf_cache_env() -> None:
    """Prefer shared HF cache when home cache is unavailable and env vars are unset."""
    if (
        os.environ.get("HF_HOME")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_DATASETS_CACHE")
    ):
        return

    user = os.environ.get("USER", "").strip()
    if not user:
        return

    shared_cache_root = f"/common/users/{user}/.cache"
    if not os.path.isdir(shared_cache_root):
        return

    shared_hf_home = os.path.join(shared_cache_root, "huggingface")
    if not os.path.isdir(shared_hf_home):
        return

    os.environ["HF_HOME"] = shared_hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(shared_hf_home, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(shared_hf_home, "datasets")


_configure_default_hf_cache_env()

from wagering.calibration import calibration_enabled, fit_or_load_logit_calibrator
from wagering.methods.factory import load_wagering_method
from wagering.utils import load_and_merge_configs
from wagering.utils.multi_llm_ensemble import configure_wagering_cache_dir

# Import training and evaluation functions
def load_module_from_path(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

train_module = load_module_from_path("wagering_train", SCRIPTS_PATH / "wagering_train.py")
eval_module = load_module_from_path("wagering_eval", SCRIPTS_PATH / "wagering_eval.py")

train_main = train_module.main
eval_main = eval_module.main

log = logging.getLogger("wagering")


def _json_default(obj):
    """Best-effort JSON serializer for result dicts."""
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


def _maybe_write_run_artifacts(
    *,
    config_path: Path,
    merged_config: dict,
    train_results: Optional[dict],
    eval_results: Optional[dict],
    calibration_path: Optional[str],
    checkpoint_path: Optional[str],
) -> None:
    """
    Persist the resolved config and pipeline results to disk.

    Writes into a stable directory under the checkpoint base dir so repeated runs
    (including wagering_pipeline_repeat) always get per-run artifacts on disk.
    """
    try:
        out_base = merged_config.get("checkpoint_base_dir")
        base_dir: Optional[Path] = (
            Path(out_base).expanduser()
            if isinstance(out_base, str) and out_base.strip()
            else None
        )
        # Prefer a per-run stable directory under the actual checkpoint path to
        # avoid collisions when multiple runs share the same wandb_name.
        if checkpoint_path:
            run_dir = Path(checkpoint_path).expanduser().resolve() / "pipeline_artifacts"
            run_dir.mkdir(parents=True, exist_ok=True)
            out_dir = run_dir.parent
            wandb_name = merged_config.get("wandb_name")
            base_dir = out_dir.parent
        else:
            if base_dir is None:
                return
            wandb_name = merged_config.get("wandb_name")
            if isinstance(wandb_name, str) and wandb_name.strip():
                out_dir = base_dir / wandb_name.strip()
            else:
                out_dir = base_dir
            run_dir = out_dir / "pipeline_artifacts"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save original config for reproducibility (best-effort copy).
        try:
            src_copy = run_dir / "config.original.yaml"
            if config_path.exists() and not src_copy.exists():
                shutil.copy2(config_path, src_copy)
        except Exception:
            pass

        # Save merged/resolved config.
        try:
            import yaml

            with (run_dir / "config.merged.yaml").open("w") as f:
                yaml.safe_dump(merged_config, f, sort_keys=False)
        except Exception:
            # yaml is already a dependency in this repo; if something goes wrong, skip silently.
            pass

        summary = {
            "config_path": str(config_path),
            "checkpoint_base_dir": str(base_dir),
            "output_dir": str(out_dir),
            "wandb_name": wandb_name,
            "checkpoint_path": checkpoint_path,
            "calibration_path": calibration_path,
        }
        with (run_dir / "pipeline.summary.json").open("w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

        if train_results is not None:
            with (run_dir / "train.results.json").open("w") as f:
                json.dump(train_results, f, indent=2, default=_json_default)
        if eval_results is not None:
            with (run_dir / "eval.results.json").open("w") as f:
                json.dump(eval_results, f, indent=2, default=_json_default)

            # Also persist a compact metrics-only view (scalar floats + overall means).
            try:
                metrics_keys = [
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
                ]

                def _to_float(v):
                    try:
                        if v is None:
                            return None
                        if isinstance(v, (list, tuple)) and v:
                            v = v[0]
                        return float(v)
                    except Exception:
                        return None

                per_dataset = {}
                per_dataset_subset_any_model_wrong = {}
                overall_accum = {k: [] for k in metrics_keys}
                subset_overall_accum = {k: [] for k in metrics_keys}
                if isinstance(eval_results, dict):
                    for ds_name, ds_res in eval_results.items():
                        if not isinstance(ds_res, dict):
                            continue
                        out_row = {}
                        for k in metrics_keys:
                            fv = _to_float(ds_res.get(k))
                            if fv is not None:
                                out_row[k] = fv
                                overall_accum[k].append(fv)
                        if out_row:
                            per_dataset[str(ds_name)] = out_row

                        subset = ds_res.get("subset_any_model_wrong")
                        if isinstance(subset, dict):
                            sub_row = {}
                            for k in metrics_keys:
                                fv = _to_float(subset.get(k))
                                if fv is not None:
                                    sub_row[k] = fv
                                    subset_overall_accum[k].append(fv)
                            if sub_row:
                                sub_row["num_examples"] = subset.get("num_examples")
                                per_dataset_subset_any_model_wrong[str(ds_name)] = sub_row

                overall = {}
                for k, vals in overall_accum.items():
                    if vals:
                        overall[k] = sum(vals) / float(len(vals))

                subset_overall = {}
                for k, vals in subset_overall_accum.items():
                    if vals:
                        subset_overall[k] = sum(vals) / float(len(vals))

                metrics_out = {
                    "per_dataset": per_dataset,
                    "overall_mean_across_datasets": overall,
                    "subset_any_model_wrong": {
                        "per_dataset": per_dataset_subset_any_model_wrong,
                        "overall_mean_across_datasets": subset_overall,
                    },
                }
                with (run_dir / "eval.metrics.json").open("w") as f:
                    json.dump(metrics_out, f, indent=2, default=_json_default)
            except Exception:
                pass
    except Exception:
        # Never fail the pipeline due to logging.
        return


def _parse_gpu_ids(csv: str) -> str:
    """Normalize comma-separated GPU ids for CUDA_VISIBLE_DEVICES."""
    gpu_ids = [p.strip() for p in str(csv).split(",") if p.strip()]
    if not gpu_ids:
        raise ValueError("No GPUs provided. Example: --gpus 0,1,2,3")
    return ",".join(gpu_ids)


def _cleanup_checkpoints(checkpoint_path: Optional[str], mode: str = "transition"):
    """Clean up checkpoint artifacts after pipeline completion.

    Modes:
      - none: do not delete anything
      - transition: delete epoch transition checkpoints only
      - all: delete the entire created checkpoint directory
    """
    if checkpoint_path is None:
        return
    if mode == "none":
        return

    ckpt_dir = Path(checkpoint_path)
    if not ckpt_dir.exists():
        return

    if mode == "all":
        try:
            shutil.rmtree(ckpt_dir)
            log.info("Removed checkpoint directory %s", ckpt_dir)
        except Exception as e:
            log.warning("Could not remove checkpoint directory %s: %s", ckpt_dir, e)
        return

    removed = 0
    for path in ckpt_dir.glob("checkpoint_epoch_*_step_*.pt"):
        try:
            path.unlink()
            removed += 1
        except Exception as e:
            log.warning("Could not remove transition checkpoint %s: %s", path, e)
    for path in ckpt_dir.glob("checkpoint_epoch_*_step_*.pt.tmp"):
        try:
            path.unlink()
            removed += 1
        except Exception as e:
            log.warning("Could not remove transition checkpoint tmp %s: %s", path, e)
    log.info("Removed %d transition checkpoints from %s", removed, ckpt_dir)


def run_pipeline(
    config_path: Optional[Path] = None,
    skip_training: bool = False,
    skip_evaluation: bool = False,
    checkpoint_path_override: Optional[str] = None,
    gpus: Optional[str] = None,
    cleanup_checkpoints: str = "transition",
):
    """Run end-to-end pipeline with unified wandb run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose library logging
    logging.getLogger("wagering").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if config_path is None:
        raise ValueError("config_path is required")

    if gpus is not None:
        visible_gpus = _parse_gpu_ids(gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
        log.info("Using CUDA_VISIBLE_DEVICES=%s", visible_gpus)

    args = load_and_merge_configs(config_path)
    configure_wagering_cache_dir(args.get("cache_path"))
    calibration_path = None
    checkpoint_path = None
    created_checkpoint_path = None
    train_results = None
    eval_results = None

    wagering_method = load_wagering_method(
        args["wagering_method"]["name"],
        num_models=len(args["models"]),
        config=args["wagering_method"].get("config", {}),
    )
    requires_training = len(wagering_method.get_trainable_parameters()) > 0 and bool(args.get("datasets"))

    if calibration_enabled(args):
        log.info("\n" + "=" * 80)
        log.info("PHASE 1: CALIBRATION")
        log.info("=" * 80)

        try:
            _, calibration_path, _ = fit_or_load_logit_calibrator(args)
        except Exception as e:
            log.error(f"Calibration failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Training phase
    if not skip_training and requires_training:
        log.info("\n" + "=" * 80)
        log.info("PHASE 2: TRAINING")
        log.info("=" * 80)
        
        try:
            train_results = train_main(
                config_path=str(config_path),
                calibration_path=calibration_path,
            )
            checkpoint_path = train_results.get("checkpoint_path")
            created_checkpoint_path = checkpoint_path
            calibration_path = train_results.get("calibration_path", calibration_path)
            
            import wandb
            if wandb.run is not None:
                log.debug(f"WandB run {wandb.run.id} active - continuing to evaluation")
                
        except Exception as e:
            log.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        if skip_training:
            log.info("Skipping training phase")
        else:
            log.info("Skipping training phase because the wagering method has no trainable parameters or no training datasets were provided")
        checkpoint_path = checkpoint_path_override or args.get("checkpoint_path")

    if checkpoint_path is None and len(wagering_method.get_trainable_parameters()) > 0 and not skip_evaluation:
        log.error("No checkpoint path available for evaluation")
        sys.exit(1)
    
    # Evaluation phase
    if not skip_evaluation:
        log.info("\n" + "=" * 80)
        log.info("PHASE 3: EVALUATION")
        log.info("=" * 80)
        
        try:
            eval_results = eval_main(
                config_path=str(config_path),
                checkpoint_path=checkpoint_path,
                calibration_path=calibration_path,
            )
            log.info("Evaluation complete")
            
        except Exception as e:
            log.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        log.info("Skipping evaluation phase")
    
    log.info("\n" + "=" * 80)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 80)

    _maybe_write_run_artifacts(
        config_path=config_path,
        merged_config=args,
        train_results=train_results,
        eval_results=eval_results,
        calibration_path=calibration_path,
        checkpoint_path=checkpoint_path,
    )

    _cleanup_checkpoints(created_checkpoint_path, mode=cleanup_checkpoints)


def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description="Run multi-LLM wagering pipeline"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config file (YAML)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation phase"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids to expose via CUDA_VISIBLE_DEVICES (example: 1,2,3)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override checkpoint path (use with --skip-training)"
    )
    parser.add_argument(
        "--cleanup-checkpoints",
        type=str,
        choices=["none", "transition", "all"],
        default="transition",
        help=(
            "Checkpoint cleanup mode after pipeline completion: "
            "none (keep all), transition (remove checkpoint_epoch_* files), "
            "all (delete entire checkpoint directory)."
        ),
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    run_pipeline(
        config_path=config_path,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation,
        checkpoint_path_override=args.checkpoint_path,
        gpus=args.gpus,
        cleanup_checkpoints=args.cleanup_checkpoints,
    )


if __name__ == "__main__":
    main()
