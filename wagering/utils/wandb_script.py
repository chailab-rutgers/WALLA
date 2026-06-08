"""Wandb initialization helpers for wagering train/eval scripts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from wagering.utils.wandb_logging import get_run_step

log = logging.getLogger("wagering")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_api_keys() -> Dict[str, Any]:
    api_keys_path = PROJECT_ROOT / ".api_keys.yaml"
    if not api_keys_path.exists():
        return {}

    with open(api_keys_path, "r") as f:
        config = yaml.safe_load(f)
    if not config:
        return {}

    filtered: Dict[str, Any] = {}
    for key, value in config.items():
        if value is None or value == "null" or value == "":
            continue
        if isinstance(value, str) and ("your-" in value.lower() and "-here" in value.lower()):
            continue
        filtered[key] = value
    return filtered


def init_wandb_for_training(args: Dict[str, Any], checkpoint_metadata: Dict[str, Any]) -> Any:
    import wandb

    api_keys = load_api_keys()
    wandb_api_key = api_keys.get("wandb_api_key")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    wandb_config = dict(args)
    wandb_config.update(checkpoint_metadata)
    wandb.init(
        project=args.get("wandb_project", "multi-llm-wagering"),
        entity=args.get("wandb_entity", None),
        name=args.get("wandb_name", None),
        config=wandb_config,
        tags=[
            f"wagering_{args['wagering_method']['name']}",
            f"agg_{args['aggregation']['name']}",
            f"models_{len(args['models'])}",
            f"dataset_{checkpoint_metadata.get('training_dataset', 'unknown')}",
            "training",
        ],
    )
    log.info("Initialized wandb run: %s", wandb.run.id)
    return wandb


def init_wandb_for_eval(args: Dict[str, Any], api_keys: Dict[str, Any]) -> Any:
    import wandb

    wandb_api_key = api_keys.get("wandb_api_key")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    if wandb.run is not None:
        log.info("Continuing wandb run from training: %s", wandb.run.id)
        wandb_logger = wandb
    else:
        log.info("Creating new wandb run for evaluation")
        wandb.init(
            project=args.get("wandb_project", "multi-llm-wagering"),
            entity=args.get("wandb_entity", None),
            name=args.get("wandb_name", None),
            tags=["evaluation"],
        )
        wandb_logger = wandb

    if wandb.run and "evaluation" not in (wandb.run.tags or []):
        wandb.run.tags = list(wandb.run.tags or []) + ["evaluation"]
    return wandb_logger


def resolve_eval_wandb_starting_step(wandb_logger: Any) -> Optional[int]:
    if not wandb_logger:
        return None
    import wandb

    if wandb.run is None:
        return None
    step = get_run_step(wandb_logger)
    if step is not None:
        log.info("Continuing from wandb step %d", step)
    return step
