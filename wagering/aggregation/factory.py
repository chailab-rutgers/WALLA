"""
Factory for loading aggregation functions.
"""

import logging
from .base import AggregationFunction
from .linear_pooling import LinearPooling
from .logarithmic_pooling import LogarithmicPooling

log = logging.getLogger("wagering")


def load_aggregation_function(method_name: str) -> AggregationFunction:
    """
    Load an aggregation function by name.
    
    Args:
        method_name: Name of aggregation function
        
    Returns:
        AggregationFunction instance
        
    Raises:
        ValueError: If method_name is unknown
    """
    methods = {
        "linear_pooling": LinearPooling,
        "log_pooling": LogarithmicPooling,
    }
    
    if method_name in methods:
        return methods[method_name]()
    
    raise ValueError(
        f"Unknown aggregation function: {method_name}. "
        f"Available methods: {list(methods.keys())}"
    )
