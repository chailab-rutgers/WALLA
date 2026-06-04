#!/usr/bin/env python3
"""
Audit wagering YAML configs against runtime readers (entry scripts + wagering package).

Reports:
  1. Top-level YAML keys with no known reader
  2. Factory method names / top-level keys never used by canonical configs
  3. wagering_method.config keys not read by the configured method class
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.utils.config_utils import load_and_merge_configs

CONFIG_ROOT = PROJECT_ROOT / "examples" / "configs" / "wagering_training"
EXCLUDE_DIR_PARTS = ("_generated_calibration_compare",)

# Keys consumed only during merge (not passed to train/eval as runtime knobs).
MERGE_ONLY_TOP_KEYS: FrozenSet[str] = frozenset(
    {
        "_include_models",
        "_include_datasets",
        "_include_test_datasets",
        "_include_calibration",
        "_include_ood_dataset",
        "_include_ood_datasets",
        "calibration",  # merged into calibration block when using _include_calibration
    }
)

# Top-level keys read by wagering_train.py, wagering_eval.py, or wagering_pipeline.py.
RUNTIME_TOP_KEYS: FrozenSet[str] = frozenset(
    {
        "models",
        "datasets",
        "test_datasets",
        "ood_datasets",
        "ood_dataset",
        "wagering_method",
        "aggregation",
        "calibrated",
        "option_tokens",
        "num_epochs",
        "checkpoint_base_dir",
        "checkpoint_path",
        "training_batch_size",
        "balance_training_datasets",
        "shuffle_data",
        "shuffle_seed",
        "seed",
        "early_stopping_patience",
        "early_stopping_criterion",
        "use_brier_d_regret_for_early_stopping",
        "use_min_kl_for_early_stopping",
        "validation_split_ratio",
        "shared_source_tripartition",
        "cache_path",
        "report_to_wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_name",
        "dataset_split_seed",
        "wager_score_plot_every",
        "max_training_batches",
        "eval_checkpoint_dir",
        "training_datasets",
    }
)

# Allowlist: keys that may appear in YAML but are intentionally undocumented / legacy.
TOP_KEY_ALLOWLIST: FrozenSet[str] = frozenset(
    {
        "auto_resume",
        "resume_from_checkpoint",
        "max_epoch_checkpoints",
    }
)

# Per wagering_method.name: config keys the method __init__ reads (plus factory debug).
METHOD_CONFIG_KEYS: Dict[str, FrozenSet[str]] = {
    "centralized_wagers": frozenset(
        {
            "hidden_dim",
            "hidden_layers",
            "learning_rate",
            "temperature",
            "grad_clip_norm",
            "normalize_hidden_states",
            "device",
            "hidden_state_layers",
            "hidden_state_layers_per_model",
            "debug_param_dtypes",
        }
    ),
    "mse_br_wagers_v2": frozenset(
        {
            "common_hidden_dim",
            "hidden_layers",
            "learning_rate",
            "temperature",
            "grad_clip_norm",
            "normalize_hidden_states",
            "device",
            "lr_decay_factor",
            "lr_decay_steps",
            "hidden_state_layers",
            "hidden_state_layers_per_model",
            "debug_param_dtypes",
        }
    ),
    "mse_br_wagers_v3": frozenset(
        {
            "common_hidden_dim",
            "hidden_layers",
            "learning_rate",
            "temperature",
            "grad_clip_norm",
            "normalize_hidden_states",
            "device",
            "lr_decay_factor",
            "lr_decay_steps",
            "hidden_state_layers",
            "hidden_state_layers_per_model",
            "debug_param_dtypes",
        }
    ),
    "mse_br_wagers_v2_augmented": frozenset(
        {
            "common_hidden_dim",
            "hidden_layers",
            "learning_rate",
            "temperature",
            "grad_clip_norm",
            "normalize_hidden_states",
            "score_function",
            "device",
            "lr_decay_factor",
            "lr_decay_steps",
            "hidden_state_layers",
            "hidden_state_layers_per_model",
            "ablation_study",
            "debug_param_dtypes",
        }
    ),
    "route_llm_bert": frozenset(
        {
            "bert_model_name",
            "max_seq_length",
            "learning_rate",
            "temperature",
            "grad_clip_norm",
            "weight_decay",
            "freeze_bert",
            "pubmedqa_strip_context",
            "debug_router_prompts",
            "router_dropout",
            "ranking_loss_weight",
            "ranking_margin",
            "lr_decay_factor",
            "lr_decay_steps",
            "device",
            "debug_param_dtypes",
        }
    ),
    "router_dc": frozenset(
        {
            "encoder_model_name",
            "bert_model_name",
            "max_seq_length",
            "learning_rate",
            "temperature",
            "grad_clip_norm",
            "weight_decay",
            "freeze_encoder",
            "pubmedqa_strip_context",
            "similarity_function",
            "top_k",
            "last_k",
            "min_pos_p",
            "neg_mask_threshold",
            "inactive_model_indices",
            "lr_decay_factor",
            "lr_decay_steps",
            "encoder_lr_mult",
            "expert_weight_decay",
            "max_grad_norm_before_skip",
            "device",
            "router_dropout",
            "bce_loss_weight",
            "mixture_ce_weight",
            "param_l2_weight",
            "model_embedding_dim",
            "router_hidden_dim",
            "knowledge_dim",
            "tokenizer_use_fast",
            "debug_param_dtypes",
        }
    ),
    "nirt_router": frozenset(
        {
            "encoder_model_name",
            "bert_model_name",
            "max_seq_length",
            "learning_rate",
            "temperature",
            "grad_clip_norm",
            "weight_decay",
            "freeze_encoder",
            "pubmedqa_strip_context",
            "knowledge_dim",
            "model_embedding_dim",
            "router_hidden_dim",
            "router_dropout",
            "bce_loss_weight",
            "mixture_ce_weight",
            "param_l2_weight",
            "lr_decay_factor",
            "lr_decay_steps",
            "device",
            "tokenizer_use_fast",
            "debug_param_dtypes",
        }
    ),
    "packllm_perplexity_wagers": frozenset({"tau", "epsilon", "debug_param_dtypes"}),
    "kl_uniform_wagers": frozenset({"confidence_epsilon", "debug_param_dtypes"}),
    "equal_wagers": frozenset({"debug_param_dtypes"}),
}

FACTORY_METHOD_NAMES: FrozenSet[str] = frozenset(
    {
        "equal_wagers",
        "centralized_wagers",
        "mse_br_wagers_v2",
        "mse_br_wagers_v3",
        "mse_br_wagers_v2_augmented",
        "route_llm_bert",
        "router_dc",
        "packllm_perplexity_wagers",
        "kl_uniform_wagers",
        "nirt_router",
    }
)

GET_PATTERN = re.compile(
    r'(?:args|config|cfg|calibration_config|wagering_config|aggregation_config|dataset_cfg|merged_config|self\.config)\.get\(\s*["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']'
)


def canonical_config_paths() -> List[Path]:
    paths: List[Path] = []
    for p in sorted(CONFIG_ROOT.rglob("*.yaml")):
        if any(part in EXCLUDE_DIR_PARTS for part in p.parts):
            continue
        if p.parent.name in ("models", "datasets", "calibration"):
            continue
        paths.append(p)
    return paths


def collect_static_get_keys() -> Set[str]:
    roots = [
        PROJECT_ROOT / "wagering",
        PROJECT_ROOT / "scripts" / "wagering_train.py",
        PROJECT_ROOT / "scripts" / "wagering_eval.py",
        PROJECT_ROOT / "scripts" / "wagering_pipeline.py",
    ]
    keys: Set[str] = set()
    for root in roots:
        if root.is_file():
            files = [root]
        else:
            files = list(root.rglob("*.py"))
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            keys.update(GET_PATTERN.findall(text))
    return keys


def load_canonical_inventory() -> Tuple[
    Set[str],
    Dict[str, int],
    Dict[str, Set[str]],
    List[Tuple[str, str, List[str]]],
]:
    top_keys: Set[str] = set()
    method_counts: Dict[str, int] = defaultdict(int)
    method_config_keys: Dict[str, Set[str]] = defaultdict(set)
    mismatches: List[Tuple[str, str, List[str]]] = []

    merge_failures: List[str] = []

    for path in canonical_config_paths():
        try:
            merged = load_and_merge_configs(path)
        except Exception as exc:
            merge_failures.append(f"{path.name}: {exc}")
            continue
        top_keys.update(merged.keys())
        wm = merged.get("wagering_method") or {}
        name = wm.get("name")
        if not name:
            continue
        method_counts[name] += 1
        cfg = wm.get("config") or {}
        if not isinstance(cfg, dict):
            continue
        method_config_keys[name].update(cfg.keys())
        allowed = METHOD_CONFIG_KEYS.get(name)
        if allowed is None:
            mismatches.append((path.name, name, sorted(cfg.keys())))
            continue
        unknown = sorted(set(cfg.keys()) - allowed)
        if unknown:
            mismatches.append((path.name, name, unknown))

    return top_keys, dict(method_counts), dict(method_config_keys), mismatches, merge_failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit wagering config usage")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any report section is non-empty",
    )
    args = parser.parse_args()

    top_keys, method_counts, _, mismatches, merge_failures = load_canonical_inventory()
    static_keys = collect_static_get_keys()
    known_top = RUNTIME_TOP_KEYS | MERGE_ONLY_TOP_KEYS | TOP_KEY_ALLOWLIST

    orphan_top = sorted(top_keys - known_top)
    unused_factory = sorted(FACTORY_METHOD_NAMES - set(method_counts.keys()))
    unused_methods_in_configs = sorted(set(method_counts.keys()) - set(METHOD_CONFIG_KEYS.keys()))

    print("=== Canonical configs ===")
    print(f"Count: {len(canonical_config_paths())}")
    print("wagering_method.name counts:")
    for name, count in sorted(method_counts.items()):
        print(f"  {count:3d}  {name}")

    if merge_failures:
        print("\n=== Merge failures (skipped) ===")
        for line in merge_failures:
            print(f"  {line}")

    print("\n=== (1) Top-level YAML keys with no known runtime/merge reader ===")
    if orphan_top:
        for k in orphan_top:
            print(f"  {k}")
    else:
        print("  (none)")

    print("\n=== (2a) Factory method names never used in canonical configs ===")
    if unused_factory:
        for k in unused_factory:
            print(f"  {k}")
    else:
        print("  (none)")

    print("\n=== (2b) Config method names missing from METHOD_CONFIG_KEYS map ===")
    if unused_methods_in_configs:
        for k in unused_methods_in_configs:
            print(f"  {k}")
    else:
        print("  (none)")

    print("\n=== (3) Per-file misplaced wagering_method.config keys ===")
    if mismatches:
        for fname, name, keys in mismatches:
            print(f"  {fname} ({name}): {keys}")
    else:
        print("  (none)")

    exit_code = 0
    if args.strict and (orphan_top or mismatches):
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
