#!/usr/bin/env python3
"""Verify the WALLA Python environment and core package wiring.

No GPU, model downloads, or experiment data required.

Usage (from repo root):
  .venv/bin/python scripts/verify_setup.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
EXAMPLE_CONFIG = (
    PROJECT_ROOT
    / "examples"
    / "configs"
    / "wagering_training"
    / "walla_v1_2models_mmlu.yaml"
)


def check_python() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python >= 3.10 required, found {sys.version}")
    print(f"python {sys.version.split()[0]}: ok")


def check_dependencies() -> None:
    packages = [
        "accelerate",
        "datasets",
        "dill",
        "google.protobuf",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "sentencepiece",
        "torch",
        "tqdm",
        "transformers",
        "wandb",
        "yaml",
    ]
    for package in packages:
        import_module(package)
    print(f"dependencies ({len(packages)}): ok")


def check_wagering_imports() -> None:
    modules = [
        "wagering.calibration",
        "wagering.core.dataset",
        "wagering.core.model",
        "wagering.inference.evaluator",
        "wagering.methods.factory",
        "wagering.training.trainer",
        "wagering.utils.config_utils",
    ]
    for module in modules:
        import_module(module)
    print(f"wagering imports ({len(modules)}): ok")


def check_config_loading() -> None:
    from wagering.utils.config_utils import load_and_merge_configs

    if not EXAMPLE_CONFIG.exists():
        raise FileNotFoundError(f"example config not found: {EXAMPLE_CONFIG}")

    cfg = load_and_merge_configs(str(EXAMPLE_CONFIG))
    if "wagering_method" not in cfg:
        raise RuntimeError("merged config missing wagering_method")
    print(f"config load ({EXAMPLE_CONFIG.name}): ok")


def check_aggregation() -> None:
    from wagering.aggregation.factory import load_aggregation_function

    logits = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    wagers = np.array([0.6, 0.4], dtype=np.float32)

    for name in ("linear_pooling", "log_pooling"):
        agg = load_aggregation_function(name)
        log_probs, probs = agg.aggregate(logits, wagers)
        assert log_probs.shape == (3,)
        assert probs.shape == (3,)
        assert np.all(probs >= 0.0)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)

    print("aggregation: ok")


def check_wagering_methods() -> None:
    from wagering.methods.factory import load_wagering_method

    equal = load_wagering_method("equal_wagers", num_models=3, config={"device": "cpu"})
    out = equal.compute_wagers()
    assert np.allclose(out["wagers"], 1.0 / 3.0)

    num_models = 2
    batch_size = 3
    cfg = {
        "device": "cpu",
        "common_hidden_dim": 16,
        "hidden_layers": [8],
        "temperature": 1.0,
    }
    hidden_states = [np.random.randn(batch_size, 12).astype(np.float32) for _ in range(num_models)]
    logits = np.random.randn(batch_size, num_models, 4).astype(np.float32)
    gold = np.array([0, 1, 2])

    method = load_wagering_method("walla_v1", num_models=num_models, config=cfg)
    method.eval_mode()
    result = method.compute_wagers(hidden_states, model_logits=logits, gold_label=gold)
    wagers = result["wagers"]
    assert wagers.shape == (batch_size, num_models)
    assert np.allclose(wagers.sum(axis=1), 1.0, atol=1e-5)

    print("wagering methods: ok")


def main() -> None:
    check_python()
    check_dependencies()
    check_wagering_imports()
    check_config_loading()
    check_aggregation()
    check_wagering_methods()
    print("all checks passed")


if __name__ == "__main__":
    main()
