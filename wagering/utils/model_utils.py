"""
Model loading utilities.

Simplified version with strict error handling.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch

# Ensure local project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wagering.core.model import WhiteboxModel
from wagering.core.common import load_external_module
from wagering.core.generation_parameters import GenerationParametersFactory

log = logging.getLogger(__name__)


def should_load_prompt_perplexity_models_sequentially(num_models: int) -> bool:
    """
    Return True when there are more ensemble models than visible CUDA devices.

    In that situation, ``device_map`` round-robin still places multiple large
    models on at least one GPU (or subprocesses pin ``CUDA_VISIBLE_DEVICES`` to
    a single device), which commonly causes OOM. Callers should load one model
    at a time when computing prompt perplexities instead of loading the full
    ensemble into VRAM together.
    """
    if num_models <= 1:
        return False
    if not torch.cuda.is_available():
        return False
    visible = int(torch.cuda.device_count())
    if visible <= 0:
        return False
    return num_models > visible


def _resolve_device_map_for_model(
    model_cfg: Dict[str, Any],
    model_index: int,
    total_models: int,
) -> Any:
    """
    Resolve a model's device map.

    For multi-model runs, default `device_map: auto` can place every model on the
    same GPU. To avoid this, distribute models round-robin across visible GPUs
    when there are enough devices for one model per visible index.

    When ``torch.cuda.device_count()`` is smaller than the ensemble size (for
    example a driver sets ``CUDA_VISIBLE_DEVICES`` to one id), every model maps
    to ``cuda:0``; use :func:`should_load_prompt_perplexity_models_sequentially`
    and sequential perplexity loading instead of loading the full ensemble.
    """
    load_model_args = model_cfg.get("load_model_args", {}) or {}
    requested_device_map = load_model_args.get("device_map", "auto")

    if (
        requested_device_map == "auto"
        and total_models > 1
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 0
    ):
        return f"cuda:{model_index % torch.cuda.device_count()}"

    return requested_device_map


def _align_max_memory_with_device_map(load_model_args: Dict[str, Any], device_map: Any) -> Dict[str, Any]:
    """Align max_memory keys with explicit single-device CUDA mapping."""
    if not isinstance(device_map, str) or not device_map.startswith("cuda:"):
        return load_model_args

    max_memory = load_model_args.get("max_memory")
    if not isinstance(max_memory, dict) or not max_memory:
        return load_model_args

    try:
        target_gpu = int(device_map.split(":", 1)[1])
    except (TypeError, ValueError, IndexError):
        return load_model_args

    # Accept both int and string keys from YAML parsing.
    if target_gpu in max_memory:
        return load_model_args
    if str(target_gpu) in max_memory:
        return load_model_args

    if 0 in max_memory:
        source_value = max_memory[0]
    elif "0" in max_memory:
        source_value = max_memory["0"]
    else:
        # If no GPU-0 entry exists, use first entry as fallback.
        source_value = next(iter(max_memory.values()))

    load_model_args = dict(load_model_args)
    load_model_args["max_memory"] = {target_gpu: source_value}
    return load_model_args


def load_api_keys() -> Dict[str, str]:
    """
    Load API keys from .api_keys.yaml file if it exists.
    
    Returns:
        Dictionary of API keys
    """
    api_keys_path = PROJECT_ROOT / ".api_keys.yaml"
    if not api_keys_path.exists():
        return {}
    
    try:
        import yaml
        with open(api_keys_path, "r") as f:
            config = yaml.safe_load(f)
            if not config:
                return {}
            
            # Filter out invalid values
            filtered = {}
            for k, v in config.items():
                if v is None or v == "null" or v == "":
                    continue
                if isinstance(v, str) and ("your-" in v.lower() and "-here" in v.lower()):
                    continue
                filtered[k] = v
            return filtered
    except Exception as e:
        log.warning(f"Could not load API keys: {e}")
        return {}


def load_models_from_config(
    model_configs: List[Dict[str, Any]],
    cache_kwargs: Optional[Dict[str, Any]] = None,
    share_identical_models: bool = True,
) -> Tuple[List[WhiteboxModel], List[str]]:
    """
    Load multiple whitebox models from configuration.
    
    Args:
        model_configs: List of model configuration dictionaries, each containing:
            - path: Model path (REQUIRED)
            - path_to_load_script: Path to load script (optional, for custom loading)
            - load_model_args: Arguments for model loading (optional)
            - load_tokenizer_args: Arguments for tokenizer loading (optional)
            - instruct: Whether model is instruction-tuned (optional, default False)
            - generation_params: Generation parameters (optional)
        cache_kwargs: Optional cache kwargs for model loading
        share_identical_models: If True, identical model configs are loaded once
            and the same WhiteboxModel instance is reused across ensemble slots
        
    Returns:
        Tuple of (list of WhiteboxModel instances, list of model names)
        
    Raises:
        ValueError: If model config is invalid
        FileNotFoundError: If load script not found
    """
    if not model_configs:
        raise ValueError("Must provide at least one model config")
    
    cache_kwargs = cache_kwargs or {}
    models = []
    model_names = []
    shared_model_cache: Dict[str, WhiteboxModel] = {}
    shared_model_first_slot: Dict[str, int] = {}
    
    # Load API keys for HF token
    api_keys = load_api_keys()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token is None:
        hf_token = api_keys.get("hf_token") or api_keys.get("huggingface_token")
    
    total_models = len(model_configs)

    for i, model_cfg in enumerate(model_configs):
        if "path" not in model_cfg:
            raise ValueError(f"Model config {i} missing required 'path' field: {model_cfg}")
        
        model_path = model_cfg["path"]
        model_names.append(model_path.replace("/", "_"))

        shared_model_key: Optional[str] = None
        if share_identical_models:
            try:
                shared_model_key = json.dumps(model_cfg, sort_keys=True, default=str)
            except TypeError:
                shared_model_key = repr(model_cfg)

            if shared_model_key in shared_model_cache:
                models.append(shared_model_cache[shared_model_key])
                first_slot = shared_model_first_slot[shared_model_key]
                log.info(
                    "Reusing loaded model for slot %s/%s: %s (shared with slot %s)",
                    i + 1,
                    total_models,
                    model_path,
                    first_slot + 1,
                )
                continue
        
        resolved_device_map = _resolve_device_map_for_model(
            model_cfg=model_cfg,
            model_index=i,
            total_models=total_models,
        )

        # Use custom load script if provided
        if model_cfg.get("path_to_load_script"):
            load_script_path = model_cfg["path_to_load_script"]
            
            # Resolve load script path
            if not os.path.isabs(load_script_path):
                # Try relative to examples/configs/
                examples_path = PROJECT_ROOT / "examples" / "configs" / load_script_path
                if examples_path.exists():
                    load_script_path = str(examples_path)
                else:
                    raise FileNotFoundError(
                        f"Load script not found: {load_script_path} "
                        f"(tried {examples_path})"
                    )
            
            if not Path(load_script_path).exists():
                raise FileNotFoundError(f"Load script not found: {load_script_path}")
            
            # Load the module
            load_module = load_external_module(str(load_script_path))
            
            # Load model
            load_model_args = {"model_path": model_path}
            load_model_args.update(model_cfg.get("load_model_args", {}))
            load_model_args["device_map"] = resolved_device_map
            load_model_args = _align_max_memory_with_device_map(load_model_args, resolved_device_map)
            if hf_token is not None:
                load_model_args["token"] = hf_token
            
            base_model = load_module.load_model(**load_model_args)
            
            # Load tokenizer
            load_tok_args = {"model_path": model_path}
            load_tok_args.update(model_cfg.get("load_tokenizer_args", {}))
            if hf_token is not None:
                load_tok_args["token"] = hf_token
            
            tokenizer = load_module.load_tokenizer(**load_tok_args)
            
            # Set pad_token_id
            if tokenizer.pad_token_id is not None:
                if hasattr(base_model, 'generation_config'):
                    base_model.generation_config.pad_token_id = tokenizer.pad_token_id
                if hasattr(base_model.config, 'pad_token_id'):
                    base_model.config.pad_token_id = tokenizer.pad_token_id
            
            # Generation params
            generation_params = GenerationParametersFactory.from_params(
                yaml_config=model_cfg.get("generation_params", {}),
                native_config=base_model.generation_config.to_dict()
            )
            
            # Create WhiteboxModel
            instruct = model_cfg.get("instruct", False)
            model = WhiteboxModel(
                base_model,
                tokenizer,
                model_path,
                model_cfg.get("type", "CausalLM"),
                generation_params,
                instruct=instruct,
            )
        else:
            # Use default loading
            instruct = model_cfg.get("instruct", False)
            
            model = WhiteboxModel.from_pretrained(
                model_path,
                generation_params=model_cfg.get("generation_params", {}),
                device_map=resolved_device_map,
                add_bos_token=model_cfg.get("add_bos_token", True),
                instruct=instruct,
                **cache_kwargs,
            )
        
        models.append(model)
        if shared_model_key is not None:
            shared_model_cache[shared_model_key] = model
            shared_model_first_slot[shared_model_key] = i
        log.info(
            "Loaded model %s/%s: %s (device_map=%s)",
            i + 1,
            total_models,
            model_path,
            resolved_device_map,
        )
    
    return models, model_names
