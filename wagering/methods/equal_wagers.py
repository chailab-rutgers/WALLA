"""
Equal wagers implementation: assigns equal weights to all LLMs.
"""

import numpy as np
from typing import Optional, Dict, Any, List

from .base import WageringMethod
from wagering.core.model import WhiteboxModel


class EqualWagers(WageringMethod):
    """
    Simple wagering method that assigns equal weights to all LLMs.
    
    This is the baseline method with no trainable parameters.
    """
    
    def compute_wagers(
        self,
        question: Optional[str] = None,
        models: Optional[List[WhiteboxModel]] = None,
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Return equal wagers for all models.
        Supports both batch and single-sample modes.
        
        Args:
            question: Ignored (kept for interface compatibility)
            models: Ignored (kept for interface compatibility)
            model_logits: np.ndarray of shape [batch_size, num_models, num_options] (batch)
                or [num_models, num_options] (single sample)
            gold_label: Ignored (kept for interface compatibility)
            **kwargs: Ignored
            
        Returns:
            Dictionary with:
                "wagers": np.ndarray of shape [batch_size, num_models] (batch)
                    or [num_models] (single sample)
        """
        # Detect batch mode from model_logits if provided
        if model_logits is not None and model_logits.ndim == 3:
            batch_size = model_logits.shape[0]
            # Return [batch_size, num_models] of equal weights
            wagers = np.ones((batch_size, self.num_models), dtype=np.float32) / self.num_models
        else:
            # Single sample mode: return [num_models]
            wagers = np.ones(self.num_models, dtype=np.float32) / self.num_models
        
        return {"wagers": wagers}
    
    def update(
        self,
        aggregated_probs: np.ndarray,
        aggregated_pred: np.ndarray,
        gold_label: np.ndarray,
        model_probs: np.ndarray,
        model_logits: np.ndarray,
        question: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        No-op update (equal wagers don't change).
        Supports both batch and single-sample modes.
        
        Args:
            aggregated_probs: Ignored
            aggregated_pred: Ignored (can be int or array)
            gold_label: Ignored (can be int or array)
            model_probs: Ignored
            model_logits: Ignored
            question: Ignored
            **kwargs: Ignored
            
        Returns:
            Empty dictionary
        """
        return {}


