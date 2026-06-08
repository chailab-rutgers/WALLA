#!/usr/bin/env python3
"""
End-to-end pipeline for multi-LLM wagering.

Keeps wandb run active between training and evaluation phases.

Usage: python wagering_pipeline.py <config_file.yaml>
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from wagering.calibration import calibration_enabled, fit_or_load_logit_calibrator
from wagering.methods.factory import load_wagering_method
from wagering.utils import load_and_merge_configs
from wagering.utils.cache_manager import configure_wagering_cache_dir
from wagering.utils.pipeline_artifacts import write_pipeline_artifacts
from wagering.utils.script_runtime import ensure_project_venv, parse_gpu_ids

ensure_project_venv()

from wagering_train import main as train_main
from wagering_eval import main as eval_main

log = logging.getLogger("wagering")


def _cleanup_checkpoints(checkpoint_path: Optional[str], mode: str = "transition") -> None:
    if checkpoint_path is None or mode == "none":
        return

    ckpt_dir = Path(checkpoint_path)
    if not ckpt_dir.exists():
        return

    if mode == "all":
        shutil.rmtree(ckpt_dir)
        log.info("Removed checkpoint directory %s", ckpt_dir)
        return

    removed = 0
    for pattern in ("checkpoint_epoch_*_step_*.pt", "checkpoint_epoch_*_step_*.pt.tmp"):
        for path in ckpt_dir.glob(pattern):
            path.unlink()
            removed += 1
    log.info("Removed %d transition checkpoints from %s", removed, ckpt_dir)


def run_pipeline(
    config_path: Optional[Path] = None,
    skip_training: bool = False,
    skip_evaluation: bool = False,
    checkpoint_path_override: Optional[str] = None,
    gpus: Optional[str] = None,
    cleanup_checkpoints: str = "transition",
):
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
    logging.getLogger("wagering").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if config_path is None:
        raise ValueError("config_path is required")

    if gpus is not None:
        visible_gpus = ",".join(parse_gpu_ids(gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
        log.info("Using CUDA_VISIBLE_DEVICES=%s", visible_gpus)

    args = load_and_merge_configs(config_path)
    configure_wagering_cache_dir(args["cache_path"])

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
    requires_training = len(wagering_method.get_trainable_parameters()) > 0 and bool(args.get("dataset"))

    if calibration_enabled(args):
        log.info("\n%s\nPHASE 1: CALIBRATION\n%s", "=" * 80, "=" * 80)
        _, calibration_path, _ = fit_or_load_logit_calibrator(args)

    if not skip_training and requires_training:
        log.info("\n%s\nPHASE 2: TRAINING\n%s", "=" * 80, "=" * 80)
        train_results = train_main(
            config_path=str(config_path),
            calibration_path=calibration_path,
        )
        checkpoint_path = train_results.get("checkpoint_path")
        created_checkpoint_path = checkpoint_path
        calibration_path = train_results.get("calibration_path", calibration_path)
    else:
        if skip_training:
            log.info("Skipping training phase")
        else:
            log.info(
                "Skipping training phase because the wagering method has no trainable parameters "
                "or no training dataset was provided"
            )
        checkpoint_path = checkpoint_path_override or args.get("checkpoint_path")

    if checkpoint_path is None and len(wagering_method.get_trainable_parameters()) > 0 and not skip_evaluation:
        raise ValueError("No checkpoint path available for evaluation")

    if not skip_evaluation:
        log.info("\n%s\nPHASE 3: EVALUATION\n%s", "=" * 80, "=" * 80)
        eval_results = eval_main(
            config_path=str(config_path),
            checkpoint_path=checkpoint_path,
            calibration_path=calibration_path,
        )
        log.info("Evaluation complete")
    else:
        log.info("Skipping evaluation phase")

    log.info("\n%s\nPIPELINE COMPLETE\n%s", "=" * 80, "=" * 80)

    write_pipeline_artifacts(
        config_path=config_path,
        merged_config=args,
        train_results=train_results,
        eval_results=eval_results,
        calibration_path=calibration_path,
        checkpoint_path=checkpoint_path,
    )
    _cleanup_checkpoints(created_checkpoint_path, mode=cleanup_checkpoints)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-LLM wagering pipeline")
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training phase")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation phase")
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
        help="Override checkpoint path (use with --skip-training)",
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
        raise FileNotFoundError(f"Config file not found: {config_path}")

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
