"""Aggregation methods for wagering package."""

from .base import AggregationFunction
from .factory import load_aggregation_function
from .linear_pooling import LinearPooling
from .logarithmic_pooling import LogarithmicPooling

__all__ = [
    "AggregationFunction",
    "load_aggregation_function",
    "LinearPooling",
    "LogarithmicPooling",
]
