"""
Linear pooling aggregation: weighted average of probabilities.
"""

import logging
import numpy as np
import torch
from typing import Tuple

from .base import AggregationFunction


logger = logging.getLogger(__name__)


class LinearPooling(AggregationFunction):
    """
    Linear pooling: weighted average of probabilities from each model.
    
    Linear pooling aggregates probabilities directly: A = sum_i w_i * P_i(H|E_i)
    where w_i >= 0 and wagers are normalized internally.
    """
    
    def aggregate(
        self,
        model_logits: np.ndarray,
        wagers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate using linear pooling (weighted average in probability space).
        
        Args:
            model_logits: Shape [batch_size, num_models, num_options] or [num_models, num_options]
            wagers: Shape [batch_size, num_models] or [num_models]
            
        Returns:
            aggregated_log_probs: Log-probabilities after aggregation
            aggregated_probs: Normalized probabilities after aggregation
        """
        model_logits = np.asarray(model_logits, dtype=np.float32)
        wagers = np.asarray(wagers, dtype=np.float32)

        def _array_stats(name: str, arr: np.ndarray) -> str:
            arr = np.asarray(arr)
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return (
                    f"{name}: shape={arr.shape}, nan={nan_count}, inf={inf_count}, "
                    "finite_min=nan, finite_max=nan"
                )
            return (
                f"{name}: shape={arr.shape}, nan={nan_count}, inf={inf_count}, "
                f"finite_min={float(finite.min()):.6g}, finite_max={float(finite.max()):.6g}"
            )
        
        # Batch mode
        if model_logits.ndim == 3 and wagers.ndim == 2:
            batch_size, num_models, num_options = model_logits.shape
            
            if wagers.shape != (batch_size, num_models):
                raise ValueError(
                    f"Wagers shape mismatch: expected [{batch_size}, {num_models}], "
                    f"got {wagers.shape}"
                )

            if np.any(wagers < 0):
                raise ValueError("Wagers must be non-negative")

            wager_sums = np.sum(wagers, axis=1, keepdims=True)
            if np.any(wager_sums <= 1e-10):
                raise ValueError("Wagers must have positive sum")

            if np.any(~np.isfinite(model_logits)):
                bad_idx = np.argwhere(~np.isfinite(model_logits))
                first_bad = tuple(int(i) for i in bad_idx[0]) if bad_idx.size else None
                logger.error(
                    "LinearPooling received non-finite model logits in batch mode. %s | %s | first_bad_logit_index=%s",
                    _array_stats("model_logits", model_logits),
                    _array_stats("wagers", wagers),
                    first_bad,
                )
                raise ValueError("Invalid model logits (NaN or inf detected)")

            if np.any(~np.isfinite(wagers)):
                bad_idx = np.argwhere(~np.isfinite(wagers))
                first_bad = tuple(int(i) for i in bad_idx[0]) if bad_idx.size else None
                logger.error(
                    "LinearPooling received non-finite wagers in batch mode. %s | %s | first_bad_wager_index=%s",
                    _array_stats("wagers", wagers),
                    _array_stats("model_logits", model_logits),
                    first_bad,
                )
                raise ValueError("Invalid wagers (NaN or inf detected)")

            normalized_wagers = wagers / wager_sums
            
            # Softmax to get probabilities
            max_logits = np.max(model_logits, axis=2, keepdims=True)
            stabilized = model_logits - max_logits
            exp_stabilized = np.exp(stabilized)
            probs = exp_stabilized / exp_stabilized.sum(axis=2, keepdims=True)
            
            # Weighted average
            aggregated_probs = (normalized_wagers[:, :, None] * probs).sum(axis=1)
            
            if np.any(np.isnan(aggregated_probs)) or np.any(np.isinf(aggregated_probs)):
                bad_batch_idx = np.argwhere(~np.isfinite(aggregated_probs))
                first_bad = tuple(int(i) for i in bad_batch_idx[0]) if bad_batch_idx.size else None
                logger.error(
                    "Invalid aggregated probabilities in LinearPooling batch mode. %s | %s | %s | %s | first_bad_aggregated_index=%s",
                    _array_stats("model_logits", model_logits),
                    _array_stats("wagers", wagers),
                    _array_stats("normalized_wagers", normalized_wagers),
                    _array_stats("model_probs", probs),
                    first_bad,
                )
                raise ValueError(
                    "Invalid aggregated probabilities (NaN or inf detected); "
                    f"first_bad_index={first_bad}; "
                    f"model_logits_nan={int(np.isnan(model_logits).sum())}, "
                    f"model_logits_inf={int(np.isinf(model_logits).sum())}, "
                    f"wagers_nan={int(np.isnan(wagers).sum())}, wagers_inf={int(np.isinf(wagers).sum())}"
                )
            
            # Normalize
            probs_sum = aggregated_probs.sum(axis=1, keepdims=True)
            if np.any(probs_sum < 1e-10):
                raise ValueError("Aggregated probabilities sum to near-zero")
            
            aggregated_probs = aggregated_probs / probs_sum
            aggregated_probs = np.clip(aggregated_probs, 0.0, 1.0)
            aggregated_probs = aggregated_probs / aggregated_probs.sum(axis=1, keepdims=True)
            
            # Validate
            if not np.all(aggregated_probs >= 0):
                raise ValueError("Probabilities must be non-negative")
            if not np.allclose(aggregated_probs.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("Probabilities must sum to 1.0")
            
            # Log probabilities
            epsilon = 1e-10
            aggregated_log_probs = np.log(np.clip(aggregated_probs, epsilon, 1.0))
            
            return aggregated_log_probs, aggregated_probs
        
        # Single sample mode
        elif model_logits.ndim == 2 and wagers.ndim == 1:
            if wagers.shape[0] != model_logits.shape[0]:
                raise ValueError("Wagers shape must match number of models")

            if np.any(wagers < 0):
                raise ValueError("Wagers must be non-negative")

            wager_sum = np.sum(wagers)
            if wager_sum <= 1e-10:
                raise ValueError("Wagers must have positive sum")

            if np.any(~np.isfinite(model_logits)):
                bad_idx = np.argwhere(~np.isfinite(model_logits))
                first_bad = tuple(int(i) for i in bad_idx[0]) if bad_idx.size else None
                logger.error(
                    "LinearPooling received non-finite model logits in single mode. %s | %s | first_bad_logit_index=%s",
                    _array_stats("model_logits", model_logits),
                    _array_stats("wagers", wagers),
                    first_bad,
                )
                raise ValueError("Invalid model logits (NaN or inf detected)")

            if np.any(~np.isfinite(wagers)):
                bad_idx = np.argwhere(~np.isfinite(wagers))
                first_bad = tuple(int(i) for i in bad_idx[0]) if bad_idx.size else None
                logger.error(
                    "LinearPooling received non-finite wagers in single mode. %s | %s | first_bad_wager_index=%s",
                    _array_stats("wagers", wagers),
                    _array_stats("model_logits", model_logits),
                    first_bad,
                )
                raise ValueError("Invalid wagers (NaN or inf detected)")

            normalized_wagers = wagers / wager_sum
            
            # Softmax to get probabilities
            max_logits = np.max(model_logits, axis=1, keepdims=True)
            stabilized = model_logits - max_logits
            exp_stabilized = np.exp(stabilized)
            probs = exp_stabilized / exp_stabilized.sum(axis=1, keepdims=True)
            
            # Weighted average
            pooled_probs = (normalized_wagers[:, None] * probs).sum(axis=0)
            if np.any(~np.isfinite(pooled_probs)):
                bad_idx = np.argwhere(~np.isfinite(pooled_probs))
                first_bad = tuple(int(i) for i in bad_idx[0]) if bad_idx.size else None
                logger.error(
                    "Invalid pooled probabilities in LinearPooling single mode. %s | %s | %s | %s | first_bad_pooled_index=%s",
                    _array_stats("model_logits", model_logits),
                    _array_stats("wagers", wagers),
                    _array_stats("normalized_wagers", normalized_wagers),
                    _array_stats("model_probs", probs),
                    first_bad,
                )
                raise ValueError(
                    "Invalid pooled probabilities (NaN or inf detected); "
                    f"first_bad_index={first_bad}; "
                    f"model_logits_nan={int(np.isnan(model_logits).sum())}, "
                    f"model_logits_inf={int(np.isinf(model_logits).sum())}, "
                    f"wagers_nan={int(np.isnan(wagers).sum())}, wagers_inf={int(np.isinf(wagers).sum())}"
                )
            pooled_probs = pooled_probs / pooled_probs.sum()
            
            # Validate
            if not np.all(pooled_probs >= 0):
                raise ValueError("Probabilities must be non-negative")
            if not np.isclose(pooled_probs.sum(), 1.0, atol=1e-6):
                raise ValueError("Probabilities must sum to 1.0")
            
            # Log probabilities
            epsilon = 1e-10
            pooled_log_probs = np.log(np.clip(pooled_probs, epsilon, 1.0))
            
            return pooled_log_probs, pooled_probs
        
        else:
            raise ValueError(
                f"Invalid shapes: model_logits={model_logits.shape}, wagers={wagers.shape}"
            )
    
    @staticmethod
    def aggregate_torch(
        model_logits: torch.Tensor,
        wagers: torch.Tensor,
    ) -> torch.Tensor:
        """
        PyTorch version of linear pooling (supports gradients).
        
        Args:
            model_logits: Shape [batch_size, num_models, num_options] or [num_models, num_options]
            wagers: Shape [batch_size, num_models] or [num_models]
            
        Returns:
            aggregated_probs: Aggregated probabilities
        """
        # Batch mode
        if model_logits.ndim == 3 and wagers.ndim == 2:
            if torch.any(wagers < 0):
                raise ValueError("Wagers must be non-negative")

            wager_sums = wagers.sum(dim=1, keepdim=True)
            if torch.any(wager_sums <= 1e-10):
                raise ValueError("Wagers must have positive sum")

            normalized_wagers = wagers / wager_sums
            model_probs = torch.softmax(model_logits, dim=2)
            aggregated_probs = (normalized_wagers.unsqueeze(2) * model_probs).sum(dim=1)
            aggregated_probs = aggregated_probs / aggregated_probs.sum(dim=1, keepdim=True)
            return aggregated_probs
        
        # Single sample mode
        elif model_logits.ndim == 2 and wagers.ndim == 1:
            if torch.any(wagers < 0):
                raise ValueError("Wagers must be non-negative")

            wager_sum = wagers.sum()
            if wager_sum <= 1e-10:
                raise ValueError("Wagers must have positive sum")

            normalized_wagers = wagers / wager_sum
            model_probs = torch.softmax(model_logits, dim=1)
            aggregated_probs = (normalized_wagers.unsqueeze(1) * model_probs).sum(dim=0)
            aggregated_probs = aggregated_probs / aggregated_probs.sum()
            return aggregated_probs
        
        else:
            raise ValueError(
                f"Invalid shapes: model_logits={model_logits.shape}, wagers={wagers.shape}"
            )
