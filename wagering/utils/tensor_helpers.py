"""Shared tensor utilities for wagering methods."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.norm(x, dim=1, keepdim=True)
    return x / (norms + eps)


def ensure_batch_logits(
    model_logits: np.ndarray,
    gold_label: Optional[Union[int, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    if model_logits.ndim == 3:
        return model_logits, np.asarray(gold_label)
    return model_logits[np.newaxis, :, :], np.array([gold_label])


def build_mlp(
    input_dim: int,
    hidden_layers: List[int],
    output_dim: int,
    dropout: float = 0.1,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def sigmoid_row_normalize(
    raw_wagers: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sigmoid_wagers = torch.sigmoid(raw_wagers / temperature)
    sigmoid_wagers = torch.clamp(sigmoid_wagers, min=1e-16, max=1.0 - 1e-16)
    sigmoid_sum = torch.sum(sigmoid_wagers, dim=1, keepdim=True)
    if torch.any(sigmoid_sum < 1e-16):
        raise RuntimeError("Near-zero sigmoid sum detected during compute_wagers().")
    wagers = sigmoid_wagers / sigmoid_sum
    return wagers, sigmoid_wagers


def row_normalize_nonnegative(wagers: torch.Tensor) -> torch.Tensor:
    if torch.any(~torch.isfinite(wagers)):
        raise ValueError("Non-finite wagers before normalization")
    if torch.any(wagers < 0):
        raise ValueError("Negative wagers before normalization")
    row_sums = wagers.sum(dim=1, keepdim=True)
    if torch.any(~torch.isfinite(row_sums)) or torch.any(row_sums <= 1e-12):
        raise ValueError("Invalid wager row sums during normalization")
    return wagers / row_sums


def project_hidden_states_list(
    hidden_states_list: List[np.ndarray],
    model_projections: nn.ModuleDict,
    hidden_dim: int,
    device: torch.device,
    *,
    normalize: bool = True,
    training: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> List[torch.Tensor]:
    projected_batch_list: List[torch.Tensor] = []
    for i, model_hs_batch in enumerate(hidden_states_list):
        model_hidden_dim = model_hs_batch.shape[-1]
        proj_key = f"proj_{i}"
        if proj_key not in model_projections:
            projection = nn.Linear(model_hidden_dim, hidden_dim).to(device)
            model_projections[proj_key] = projection
            if training and optimizer is not None:
                optimizer.add_param_group({"params": projection.parameters()})

        model_hs_tensor = torch.as_tensor(model_hs_batch, dtype=torch.float32, device=device)
        if normalize:
            model_hs_tensor = l2_normalize_rows(model_hs_tensor)

        with torch.set_grad_enabled(training):
            projected_batch = model_projections[proj_key](model_hs_tensor)
        projected_batch_list.append(projected_batch)
    return projected_batch_list


def stable_row_softmax_2d(scores: np.ndarray) -> np.ndarray:
    max_scores = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    denom = np.sum(exp_scores, axis=1, keepdims=True)
    return exp_scores / np.clip(denom, 1e-20, None)
