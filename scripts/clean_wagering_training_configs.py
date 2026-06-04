#!/usr/bin/env python3
"""Strip dead / misplaced keys from canonical wagering_training YAML configs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "examples" / "configs" / "wagering_training"
EXCLUDE_DIR_PARTS = ("_generated_calibration_compare",)

# Keys to remove from wagering_method.config by method name.
STRIP_BY_METHOD: Dict[str, Set[str]] = {
    "centralized_wagers": {"lr_decay_factor", "lr_decay_steps"},
    "mse_br_wagers_v2": {"ablation_study"},
    "mse_br_wagers_v3": set(),
    "mse_br_wagers_v2_augmented": set(),
    "route_llm_bert": {"hidden_state_layers", "hidden_state_layers_per_model"},
    "router_dc": {"hidden_state_layers", "hidden_state_layers_per_model"},
    "nirt_router": {"hidden_state_layers", "hidden_state_layers_per_model"},
    "packllm_perplexity_wagers": {"confidence_epsilon"},
}

RENAME_BY_METHOD: Dict[str, Dict[str, str]] = {
    "mse_br_wagers_v2": {"hidden_dim": "common_hidden_dim"},
    "mse_br_wagers_v3": {"hidden_dim": "common_hidden_dim"},
    "mse_br_wagers_v2_augmented": {"hidden_dim": "common_hidden_dim"},
}

TOP_LEVEL_STRIP = {"resume_eval", "save_every", "stop_at_last_iteration"}


def canonical_config_paths() -> List[Path]:
    paths: List[Path] = []
    for p in sorted(CONFIG_ROOT.rglob("*.yaml")):
        if any(part in EXCLUDE_DIR_PARTS for part in p.parts):
            continue
        if p.parent.name in ("models", "datasets", "calibration"):
            continue
        paths.append(p)
    return paths


def _clean_wagering_config(cfg: Dict[str, Any]) -> bool:
    wm = cfg.get("wagering_method")
    if not isinstance(wm, dict):
        return False
    name = wm.get("name")
    if not name:
        return False
    method_cfg = wm.get("config")
    if not isinstance(method_cfg, dict):
        return False

    changed = False
    strip = STRIP_BY_METHOD.get(name, set())
    for key in list(method_cfg.keys()):
        if key in strip:
            del method_cfg[key]
            changed = True

    renames = RENAME_BY_METHOD.get(name, {})
    for old_key, new_key in renames.items():
        if old_key in method_cfg and new_key not in method_cfg:
            method_cfg[new_key] = method_cfg.pop(old_key)
            changed = True
        elif old_key in method_cfg:
            del method_cfg[old_key]
            changed = True

    return changed


def _clean_top_level(cfg: Dict[str, Any]) -> bool:
    changed = False
    for key in TOP_LEVEL_STRIP:
        if key in cfg:
            del cfg[key]
            changed = True
    return changed


def _fix_broken_includes(cfg: Dict[str, Any]) -> bool:
    changed = False
    for include_key in ("_include_test_datasets", "_include_datasets", "_include_ood_datasets"):
        paths = cfg.get(include_key)
        if not isinstance(paths, list):
            continue
        new_paths = []
        for p in paths:
            if "cluster_saturation_bayes_extreme" in str(p):
                replacement = str(p).replace(
                    "cluster_saturation_bayes_extreme.yaml",
                    "cluster_saturation_bayesX.yaml",
                )
                new_paths.append(replacement)
                changed = True
            else:
                new_paths.append(p)
        if changed:
            cfg[include_key] = new_paths
    return changed


def main() -> int:
    updated = 0
    for path in canonical_config_paths():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            continue
        changed = _clean_top_level(cfg) or _clean_wagering_config(cfg) or _fix_broken_includes(cfg)
        if changed:
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            updated += 1
            print(f"updated {path.name}")
    print(f"Done. {updated} file(s) changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
