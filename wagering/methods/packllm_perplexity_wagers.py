"""
Inference-only wagers based on inverse perplexity, inspired by PackLLM (arXiv:2404.11531).

PackLLM-sim uses weights proportional to exp(-loss / tau), where loss is token-level
cross-entropy on the prompt. Since perplexity = exp(loss), this is equivalent to
inverse-perplexity weighting.
"""

from typing import Optional, Dict, Any, List

import numpy as np

from .base import WageringMethod
from wagering.core.model import WhiteboxModel


class PackLLMPerplexityWagers(WageringMethod):
    """
    Inference-only wagering method with PackLLM-style inverse-perplexity weighting.

    Weighting rule:
      w_i = softmax(-log(ppl_i) / tau)

    Inputs:
            ``model_perplexities`` in kwargs with shape [B, M] or [M].
    """

    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(num_models=num_models, config=config)
        self.requires_hidden_states = False
        self.requires_model_perplexities = True
        self.tau = float(self.config.get("tau", 1.0))
        self.epsilon = float(self.config.get("epsilon", 1e-12))

        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")

    def _weights_from_perplexities(self, perplexities_2d: np.ndarray) -> np.ndarray:
        """Compute normalized wagers from perplexities with shape [B, M]."""
        batch_size, num_models = perplexities_2d.shape
        if num_models != self.num_models:
            raise ValueError(
                f"Expected {self.num_models} models in perplexities, got {num_models}."
            )

        safe_ppl = np.clip(perplexities_2d.astype(np.float64, copy=False), self.epsilon, None)
        scores = -np.log(safe_ppl) / self.tau

        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        denom = np.sum(exp_scores, axis=1, keepdims=True)
        wagers = exp_scores / np.clip(denom, 1e-20, None)

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
        Compute wagers from explicit model perplexities.

        Args:
            question: Unused; kept for interface compatibility.
            models: Unused; kept for interface compatibility.
            model_logits: Unused; kept for interface compatibility.
            gold_label: Unused; kept for interface compatibility.
            **kwargs: Optional ``model_perplexities`` with shape [B, M] or [M].

        Returns:
            Dict with key "wagers":
              - [B, M] for batch mode
              - [M] for single-sample mode
        """
        del question, models, model_logits, gold_label

        model_perplexities = kwargs.get("model_perplexities", None)
        if model_perplexities is not None:
            perplexities = np.asarray(model_perplexities, dtype=np.float64)
            if perplexities.ndim == 1:
                wagers = self._weights_from_perplexities(perplexities[None, :])[0]
                return {"wagers": wagers}
            if perplexities.ndim == 2:
                wagers = self._weights_from_perplexities(perplexities)
                return {"wagers": wagers}
            raise ValueError(
                "model_perplexities must have shape [num_models] or [batch_size, num_models]."
            )

        raise ValueError(
            "PackLLMPerplexityWagers requires `model_perplexities` in kwargs with "
            "shape [num_models] or [batch_size, num_models]."
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
