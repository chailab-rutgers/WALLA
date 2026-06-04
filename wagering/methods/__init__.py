"""
Wagering methods for multi-LLM ensemble learning.

This module provides base classes and implementations for generating weights/wagers
for multiple LLMs based on questions, models, or both.
"""

from .base import WageringMethod
from .factory import load_wagering_method
from .equal_wagers import EqualWagers
from .centralized_wagers import CentralizedWagers
from .mse_br_wagers_v2_augmented import MSEBrWagersV2Augmented
from .route_llm_bert import RouteLLMBertWagers
from .router_dc import RouterDCWagers
from .packllm_perplexity_wagers import PackLLMPerplexityWagers
from .kl_uniform_wagers import KLUniformWagers
from .nirt_router import NIRTRouterWagers

__all__ = [
    "WageringMethod",
    "load_wagering_method",
    "EqualWagers",
    "CentralizedWagers",
    "MSEBrWagersV2Augmented",
    "RouteLLMBertWagers",
    "RouterDCWagers",
    "PackLLMPerplexityWagers",
    "KLUniformWagers",
    "NIRTRouterWagers",
]
