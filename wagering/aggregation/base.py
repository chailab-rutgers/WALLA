"""
Base class for aggregation functions that combine predictions from multiple LLMs.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class AggregationFunction(ABC):
    """
    Base class for aggregation functions that combine logits/probabilities from multiple LLMs.
    """
    
    @abstractmethod
    def aggregate(
        self,
        model_logits: np.ndarray,
        wagers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate logits from multiple models using the provided wagers.
        
        Args:
            model_logits: np.ndarray of shape [batch_size, num_models, num_options] (batch mode)
                or [num_models, num_options] (single sample, backwards compatibility)
                Logits from each model for each option
            wagers: np.ndarray of shape [batch_size, num_models] (batch mode)
                or [num_models] (single sample)
                Weights/wagers for each model (non-negative, can be unnormalized)
                
        Returns:
            aggregated_log_probs: np.ndarray of shape [batch_size, num_options] or [num_options]
                Log-probabilities after aggregation
            aggregated_probs: np.ndarray of shape [batch_size, num_options] or [num_options]
                Normalized probabilities after aggregation
        """
        pass
