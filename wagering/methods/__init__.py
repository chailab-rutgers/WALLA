"""
Wagering methods for multi-LLM ensemble learning.

This module provides base classes and implementations for generating weights/wagers
for multiple LLMs based on questions, models, or both.
"""

from .base import WageringMethod
from .factory import load_wagering_method
from .equal_wagers import EqualWagers
from .stacked_generalization import StackedGeneralization
from .walla_v1_augmented import WallaV1Augmented
from .route_llm_bert import RouteLLMBertWagers
from .router_dc import RouterDCWagers
from .packllm_perplexity_wagers import PackLLMPerplexityWagers
from .kl_uniform_wagers import KLUniformWagers
from .nirt_router import NIRTRouterWagers

__all__ = [
    "WageringMethod",
    "load_wagering_method",
    "EqualWagers",
    "StackedGeneralization",
    "WallaV1Augmented",
    "RouteLLMBertWagers",
    "RouterDCWagers",
    "PackLLMPerplexityWagers",
    "KLUniformWagers",
    "NIRTRouterWagers",
]
