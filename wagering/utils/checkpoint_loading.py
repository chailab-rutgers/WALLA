"""Checkpoint state loading helpers for wagering methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim


def load_module_state(module: nn.Module, state: Optional[dict]) -> None:
    if state is not None:
        module.load_state_dict(state)


def load_optimizer_state(optimizer: optim.Optimizer, state: Optional[dict]) -> None:
    if state is not None:
        optimizer.load_state_dict(state)


def load_scheduler_state(scheduler: Any, state: Optional[dict]) -> None:
    if state is not None:
        scheduler.load_state_dict(state)


def _nested_tensor_sum(obj: Any) -> float:
    total = 0.0
    if isinstance(obj, dict):
        for key in sorted(obj.keys()):
            total += _nested_tensor_sum(obj[key])
    elif torch.is_tensor(obj):
        total += float(obj.detach().cpu().double().sum().item())
    return total


def _subset_tensor_sum(obj: Any, keys: set[str]) -> float:
    total = 0.0
    if isinstance(obj, dict):
        for key in sorted(keys):
            if key in obj:
                total += _nested_tensor_sum(obj[key])
    return total


def load_wagering_method_from_final_dir(wagering_method: Any, checkpoint_path: Path) -> None:
    """Load ``checkpoint_path/final/wagering_state.pt`` and verify router/projection state."""
    checkpoint_file = checkpoint_path / "final" / "wagering_state.pt"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_file}. "
            f"Received checkpoint_path: {checkpoint_path}"
        )

    checkpoint_state = torch.load(checkpoint_file, map_location="cpu")
    if not isinstance(checkpoint_state, dict):
        raise TypeError(f"Expected dict checkpoint at {checkpoint_file}, got {type(checkpoint_state)}")

    wagering_method.load_state_dict(checkpoint_state)

    loaded_state = wagering_method.state_dict()
    ckpt_routers = checkpoint_state.get("routers_state_dict", {})
    ckpt_projs = checkpoint_state.get("model_projections_state_dict", {})
    loaded_routers = loaded_state.get("routers_state_dict", {})
    loaded_projs = loaded_state.get("model_projections_state_dict", {})

    ckpt_router_keys = set(ckpt_routers.keys())
    loaded_router_keys = set(loaded_routers.keys())
    ckpt_proj_keys = set(ckpt_projs.keys())
    loaded_proj_keys = set(loaded_projs.keys())

    missing_router_keys = ckpt_router_keys - loaded_router_keys
    extra_router_keys = loaded_router_keys - ckpt_router_keys
    missing_proj_keys = ckpt_proj_keys - loaded_proj_keys
    extra_proj_keys = loaded_proj_keys - ckpt_proj_keys

    if missing_router_keys or extra_router_keys or missing_proj_keys or extra_proj_keys:
        parts = ["Checkpoint/model key mismatch detected:"]
        if missing_router_keys:
            parts.append(f"  Missing router keys in model: {sorted(missing_router_keys)}")
        if extra_router_keys:
            parts.append(f"  Extra router keys in model: {sorted(extra_router_keys)}")
        if missing_proj_keys:
            parts.append(f"  Missing projection keys in model: {sorted(missing_proj_keys)}")
            parts.append(f"  Checkpoint expects: {sorted(ckpt_proj_keys)}")
            parts.append(f"  Current model has: {sorted(loaded_proj_keys)}")
        if extra_proj_keys:
            parts.append(f"  Extra projection keys in model: {sorted(extra_proj_keys)}")
        raise RuntimeError("\n".join(parts))

    ckpt_sum = _subset_tensor_sum(ckpt_routers, ckpt_router_keys) + _subset_tensor_sum(
        ckpt_projs, ckpt_proj_keys
    )
    loaded_sum = _subset_tensor_sum(loaded_routers, ckpt_router_keys) + _subset_tensor_sum(
        loaded_projs, ckpt_proj_keys
    )
    if not torch.isclose(torch.tensor(ckpt_sum), torch.tensor(loaded_sum), rtol=1e-5, atol=1e-5):
        raise RuntimeError(
            "Loaded state checksum does not match checkpoint checksum. "
            "This indicates corruption or incorrect checkpoint loading."
        )
