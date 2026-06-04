"""
Logarithmic pooling aggregation: averaging in log-probability space.
"""

import numpy as np
from typing import Tuple

from .base import AggregationFunction


class LogarithmicPooling(AggregationFunction):
    """
    Logarithmic pooling: averaging in log-probability space with internal
    normalization of non-negative wagers.
    """
    
    def aggregate(
        self,
        model_logits: np.ndarray,
        wagers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate using logarithmic pooling (weighted average in log-probability space).
        
        Args:
            model_logits: Shape [batch_size, num_models, num_options] or [num_models, num_options]
            wagers: Shape [batch_size, num_models] or [num_models]
            
        Returns:
            aggregated_log_probs: Log-probabilities after aggregation
            aggregated_probs: Normalized probabilities after aggregation
        """
        model_logits = np.asarray(model_logits, dtype=np.float32)
        wagers = np.asarray(wagers, dtype=np.float32)
        
        # Batch mode
        if model_logits.ndim == 3 and wagers.ndim == 2:
            batch_size, num_models, num_options = model_logits.shape
            
            if wagers.shape != (batch_size, num_models):
                raise ValueError(f"Wagers shape mismatch")

            if np.any(wagers < 0):
                raise ValueError("Wagers must be non-negative")

            wager_sums = np.sum(wagers, axis=1, keepdims=True)
            if np.any(wager_sums <= 1e-10):
                raise ValueError("Wagers must have positive sum")

            normalized_wagers = wagers / wager_sums
            
            # Convert to log-probabilities
            max_logits = np.max(model_logits, axis=2, keepdims=True)
            stabilized = model_logits - max_logits
            log_norm = max_logits + np.log(np.exp(stabilized).sum(axis=2, keepdims=True))
            log_probs = model_logits - log_norm
            
            # Weighted average in log space
            weighted_log = normalized_wagers[:, :, None] * log_probs
            pooled_log_unnorm = weighted_log.sum(axis=1)
            
            # Normalize
            max_pooled = np.max(pooled_log_unnorm, axis=1, keepdims=True)
            stabilized_pooled = pooled_log_unnorm - max_pooled
            log_z = max_pooled + np.log(np.exp(stabilized_pooled).sum(axis=1, keepdims=True))
            pooled_log_probs = pooled_log_unnorm - log_z
            pooled_probs = np.exp(pooled_log_probs)
            pooled_probs = pooled_probs / pooled_probs.sum(axis=1, keepdims=True)
            
            if not np.all(pooled_probs >= 0):
                raise ValueError("Probabilities must be non-negative")
            if not np.allclose(pooled_probs.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("Probabilities must sum to 1.0")
            
            return pooled_log_probs, pooled_probs
        
        # Single sample mode
        elif model_logits.ndim == 2 and wagers.ndim == 1:
            if wagers.shape[0] != model_logits.shape[0]:
                raise ValueError("Wagers shape must match number of models")

            if np.any(wagers < 0):
                raise ValueError("Wagers must be non-negative")

            wager_sum = np.sum(wagers)
            if wager_sum <= 1e-10:
                raise ValueError("Wagers must have positive sum")

            normalized_wagers = wagers / wager_sum
            
            # Convert to log-probabilities
            max_logits = np.max(model_logits, axis=1, keepdims=True)
            stabilized = model_logits - max_logits
            log_norm = max_logits + np.log(np.exp(stabilized).sum(axis=1, keepdims=True))
            log_probs = model_logits - log_norm
            
            # Weighted average in log space
            weighted_log = normalized_wagers[:, None] * log_probs
            pooled_log_unnorm = weighted_log.sum(axis=0)
            
            # Normalize
            max_pooled = np.max(pooled_log_unnorm)
            stabilized_pooled = pooled_log_unnorm - max_pooled
            log_z = max_pooled + np.log(np.exp(stabilized_pooled).sum())
            pooled_log_probs = pooled_log_unnorm - log_z
            pooled_probs = np.exp(pooled_log_probs)
            pooled_probs = pooled_probs / pooled_probs.sum()
            
            if not np.all(pooled_probs >= 0):
                raise ValueError("Probabilities must be non-negative")
            if not np.isclose(pooled_probs.sum(), 1.0, atol=1e-6):
                raise ValueError("Probabilities must sum to 1.0")
            
            return pooled_log_probs, pooled_probs
        
        else:
            raise ValueError(
                f"Invalid shapes: model_logits={model_logits.shape}, wagers={wagers.shape}"
            )
