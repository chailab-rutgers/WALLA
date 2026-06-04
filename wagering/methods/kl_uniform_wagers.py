"""
Inference-only wagers based on KL divergence from the uniform distribution.

Inspired by uncertainty-based routing in arXiv:2502.18581, this method uses
confidence estimated from each model's predictive distribution: models farther
from uniform get larger wagers.
"""

from typing import Optional, Dict, Any, List

import numpy as np

from .base import WageringMethod
from wagering.core.model import WhiteboxModel


class KLUniformWagers(WageringMethod):
    """
        Inference-only wagering method using KL(U || p) confidence.

    For each model and sample:
            confidence = KL(U || p), where U is uniform over answer options.

    Wagers are proportional to confidence and normalized to sum to 1 over models.
    """

    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(num_models=num_models, config=config)
        # This method only needs logits and should not force hidden-state loading.
        self.requires_hidden_states = False
        # Small smoothing term for all-zero confidence cases.
        self.confidence_epsilon = float(self.config.get("confidence_epsilon", 1e-8))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        max_logits = np.max(logits, axis=-1, keepdims=True)
        stabilized = logits - max_logits
        exp_vals = np.exp(stabilized)
        denom = np.sum(exp_vals, axis=-1, keepdims=True)
        return exp_vals / np.clip(denom, 1e-20, None)

    def _compute_from_logits(self, logits_3d: np.ndarray) -> np.ndarray:
        """Compute normalized wagers from logits with shape [B, M, O]."""
        batch_size, num_models, num_options = logits_3d.shape
        if num_models != self.num_models:
            raise ValueError(
                f"Expected {self.num_models} models in logits, got {num_models}."
            )

        probs = self._softmax(logits_3d)
        safe_probs = np.clip(probs, 1e-12, 1.0)

        # Match Self-Certainty direction: KL(U || p) = -log(O) - (1/O) * sum_i log p_i
        kl_to_uniform = -np.log(float(num_options)) - np.mean(np.log(safe_probs), axis=-1)
        confidences = np.maximum(kl_to_uniform, 0.0)

        # Ensure strictly positive mass for normalization stability.
        confidences = confidences + self.confidence_epsilon
        row_sums = np.sum(confidences, axis=1, keepdims=True)
        wagers = confidences / np.clip(row_sums, 1e-20, None)

        if not np.all(np.isfinite(wagers)):
            return np.ones((batch_size, self.num_models), dtype=np.float32) / self.num_models

        return wagers.astype(np.float32, copy=False)

    def compute_wagers(
        self,
        question: Optional[str] = None,
        models: Optional[List[WhiteboxModel]] = None,
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute wagers from model logits.

        Args:
            question: Unused; kept for interface compatibility.
            models: Unused; kept for interface compatibility.
            model_logits: Logits with shape [B, M, O] or [M, O].
            gold_label: Unused; kept for interface compatibility.

        Returns:
            Dict with key "wagers":
              - [B, M] for batch mode
              - [M] for single-sample mode
        """
        del question, models, gold_label

        if model_logits is None:
            # Fallback behavior aligned with equal wagers when logits are absent.
            wagers = np.ones(self.num_models, dtype=np.float32) / self.num_models
            return {"wagers": wagers}

        logits = np.asarray(model_logits, dtype=np.float64)

        if logits.ndim == 2:
            wagers = self._compute_from_logits(logits[None, :, :])[0]
            return {"wagers": wagers}

        if logits.ndim == 3:
            wagers = self._compute_from_logits(logits)
            return {"wagers": wagers}

        raise ValueError(
            "model_logits must have shape [num_models, num_options] or "
            "[batch_size, num_models, num_options]."
        )

    def update(
        self,
        aggregated_probs: np.ndarray,
        aggregated_pred: np.ndarray,
        gold_label: np.ndarray,
        model_probs: np.ndarray,
        model_logits: np.ndarray,
        question: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """No-op update: this method is inference-only and has no trainable state."""
        del aggregated_probs, aggregated_pred, gold_label, model_probs, model_logits, question, kwargs
        return {}
