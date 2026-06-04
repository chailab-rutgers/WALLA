"""
Factory for loading wagering methods from configuration.
"""

import logging
from typing import Dict, Any, Optional


log = logging.getLogger("wagering")


def _maybe_log_method_param_dtypes(method: Any, config: Dict[str, Any]) -> None:
    """
    Debug-only: log dtypes of common trainable submodules right after construction.

    Enable by setting `debug_param_dtypes: true` under wagering_method.config in YAML.
    """
    if not bool((config or {}).get("debug_param_dtypes", False)):
        return

    try:
        import torch  # local import to avoid hard dependency at import time
    except Exception:
        return

    def _first_param_dtype(module: Any) -> Optional[str]:
        try:
            params = list(module.parameters())
        except Exception:
            return None
        if not params:
            return None
        return str(params[0].dtype)

    dtype_report: Dict[str, Optional[str]] = {}
    for attr in ("encoder", "bert", "expert_embeddings", "router_head", "router"):
        if hasattr(method, attr):
            dtype_report[attr] = _first_param_dtype(getattr(method, attr))

    log.warning(
        "[debug_param_dtypes] method=%s default_dtype=%s dtypes=%s",
        method.__class__.__name__,
        str(torch.get_default_dtype()),
        dtype_report,
    )


def load_wagering_method(
    method_name: str,
    num_models: int,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Load a wagering method by name.
    
    Args:
        method_name: Name of the wagering method (e.g., "equal_wagers")
        num_models: Number of models in the ensemble
        config: Optional configuration dictionary
        
    Returns:
        WageringMethod instance
        
    Raises:
        ValueError: If method_name is unknown
    """
    config = config or {}
    
    # Import methods locally to avoid circular imports
    from .equal_wagers import EqualWagers
    from .centralized_wagers import CentralizedWagers
    from .mse_br_wagers import MSEBrWagers
    from .mse_br_wagers_v2 import MSEBrWagersV2
    from .mse_br_wagers_v3 import MSEBrWagersV3
    from .mse_br_wagers_v2_augmented import MSEBrWagersV2Augmented
    from .mse_br_wagers_v3_augmented import MSEBrWagersV3Augmented
    from .route_llm_bert import RouteLLMBertWagers
    from .router_dc import RouterDCWagers
    from .packllm_perplexity_wagers import PackLLMPerplexityWagers
    from .kl_uniform_wagers import KLUniformWagers
    from .nirt_router import NIRTRouterWagers
    
    # Built-in methods mapping
    methods = {
        "equal_wagers": EqualWagers,
        "equal": EqualWagers,
        "centralized_wagers": CentralizedWagers,
        "mse_br_wagers": MSEBrWagers,
        "mse_br_wagers_v2": MSEBrWagersV2,
        "mse_br_wagers_v3": MSEBrWagersV3,
        "mse_br_wagers_v2_augmented": MSEBrWagersV2Augmented,
        "mse_br_wagers_v3_augmented": MSEBrWagersV3Augmented,
        "route_llm_bert": RouteLLMBertWagers,
        "route_llm": RouteLLMBertWagers,
        "router_dc": RouterDCWagers,
        "routerDC": RouterDCWagers,
        "routerdc": RouterDCWagers,
        "packllm_perplexity_wagers": PackLLMPerplexityWagers,
        "kl_uniform_wagers": KLUniformWagers,
        "nirt_router": NIRTRouterWagers,
    }
    
    if method_name in methods:
        method = methods[method_name](num_models=num_models, config=config)
        _maybe_log_method_param_dtypes(method, config or {})
        return method
    
    raise ValueError(
        f"Unknown wagering method: {method_name}. "
        f"Available methods: {list(methods.keys())}"
    )

