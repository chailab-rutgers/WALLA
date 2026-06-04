"""
Checkpoint directory utilities.

Simplified version with direct, predictable naming.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from wagering.utils.dataset_utils import (
    calibration_dataset_configs_include_pubmedqa,
    datasets_for_checkpoint_hash,
)


def sanitize_name(name: str, max_length: int = 20) -> str:
    """
    Sanitize a name for use in file paths.
    
    Args:
        name: Name to sanitize
        max_length: Maximum length of the sanitized name
        
    Returns:
        Sanitized name safe for file paths
    """
    # Remove special characters, keep alphanumeric, dash, underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Replace multiple underscores with single
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    if not sanitized:
        raise ValueError(f"Cannot sanitize name '{name}' - results in empty string")
    
    return sanitized


def get_model_name(model_path: str) -> str:
    """
    Extract a short model name from a model path.
    
    Args:
        model_path: Model path (e.g., "meta-llama/Llama-3.2-1B")
        
    Returns:
        Sanitized short name
    """
    parts = model_path.replace("/", "_").split("_")
    # Take last 2-3 parts for brevity
    if len(parts) >= 2:
        return sanitize_name("_".join(parts[-2:]), max_length=25)
    return sanitize_name(model_path, max_length=25)


def get_dataset_name(dataset_config: Dict[str, Any]) -> str:
    """
    Extract a short dataset name from dataset config.
    
    Args:
        dataset_config: Dataset configuration dict
        
    Returns:
        Sanitized short name
        
    Raises:
        ValueError: If dataset name cannot be determined
    """
    name = dataset_config.get("display_name") or dataset_config.get("name")
    
    if not name:
        raise ValueError(f"Dataset config missing 'name' or 'display_name': {dataset_config}")
    
    if isinstance(name, list):
        # Handle ['org/dataset', 'config'] format
        name = name[0] if name else None
    
    if not name:
        raise ValueError(f"Invalid dataset name in config: {dataset_config}")
    
    # Extract short name
    parts = str(name).replace("/", "_").split("_")
    if len(parts) >= 1:
        return sanitize_name(parts[-1], max_length=20)
    return sanitize_name(str(name), max_length=20)


def _stable_json_value(value: Any) -> Any:
    """Normalize values into deterministic JSON-compatible structures."""
    if isinstance(value, dict):
        return {str(k): _stable_json_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(v) for v in value]
    return value


def _strip_shuffle_seed_fields(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sweep/runtime seed fields that should not partition calibration artifacts."""
    excluded = {"shuffle_seed", "seed"}
    return {k: v for k, v in config.items() if k not in excluded}


def generate_checkpoint_dir(
    base_dir: Path,
    models: List[Dict[str, Any]],
    datasets: List[Dict[str, Any]],
    wagering_method: Dict[str, Any],
    aggregation: Dict[str, Any],
    create_hash: bool = True,
    calibration: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Generate a unique checkpoint directory name based on configuration.
    
    Args:
        base_dir: Base directory for checkpoints
        models: List of model configs
        datasets: List of dataset configs
        wagering_method: Wagering method config
        aggregation: Aggregation function config
        create_hash: If True, append a hash for uniqueness
        
    Returns:
        Path to checkpoint directory
        
    Raises:
        ValueError: If any config is invalid
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not models:
        raise ValueError("Must provide at least one model")
    if not datasets:
        raise ValueError("Must provide at least one dataset")
    if not wagering_method or "name" not in wagering_method:
        raise ValueError("wagering_method must have 'name'")
    if not aggregation or "name" not in aggregation:
        raise ValueError("aggregation must have 'name'")
    
    # Extract components
    model_names = [get_model_name(m["path"]) for m in models if "path" in m]
    if not model_names:
        raise ValueError("No valid model paths found in model configs")
    
    dataset_names = [get_dataset_name(d) for d in datasets]
    if not dataset_names:
        raise ValueError("No valid dataset names found in dataset configs")
    
    wagering_name = sanitize_name(wagering_method["name"], max_length=20)
    aggregation_name = sanitize_name(aggregation["name"], max_length=20)
    
    # Build directory name components
    components = []
    
    # Models (sorted for consistency)
    models_str = "_".join(sorted(set(model_names)))
    components.append(f"models_{models_str}")
    
    # Datasets (sorted for consistency)
    datasets_str = "_".join(sorted(set(dataset_names)))
    components.append(f"datasets_{datasets_str}")
    
    # Wagering method
    components.append(f"wagering_{wagering_name}")
    
    # Aggregation
    components.append(f"agg_{aggregation_name}")

    if calibration:
        calibration_name = sanitize_name(calibration.get("name", "calibrated"), max_length=20)
        components.append(f"cal_{calibration_name}")
    
    # Join components
    dir_name = "_".join(components)
    
    # Add hash for uniqueness if requested
    if create_hash:
        # Create hash from full config for uniqueness (omit training-only dataset fields).
        datasets_hashed = datasets_for_checkpoint_hash(datasets)
        config_str = f"{models}_{datasets_hashed}_{wagering_method}_{aggregation}_{calibration}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        dir_name = f"{dir_name}_{config_hash}"
    
    return base_dir / dir_name


def generate_per_model_calibration_dir(
    base_dir: Path,
    model_path: str,
    datasets: List[Dict[str, Any]],
    calibration_config: Dict[str, Any],
    create_hash: bool = True,
) -> Path:
    """Directory for one HF model's temperature head (non-PubMedQA only).

    Keyed by that model path plus calibration datasets and hyperparameters — not by
    ensemble size or order — so any subset of models can reuse previously fitted heads.
    """
    base_dir = Path(base_dir)
    per_model_root = base_dir / "per_model"
    per_model_root.mkdir(parents=True, exist_ok=True)

    calibration_name = sanitize_name(
        calibration_config.get("name", "adaptive_temperature_scaling"), max_length=25
    )
    normalized_payload = {
        "model_path": model_path,
        "datasets": _stable_json_value(datasets),
        "calibration": _stable_json_value(_strip_shuffle_seed_fields(calibration_config)),
    }
    serialized = json.dumps(
        normalized_payload, sort_keys=True, separators=(",", ":"), default=str
    )
    config_hash = hashlib.md5(serialized.encode("utf-8")).hexdigest()[:8]
    model_digest = hashlib.md5(model_path.encode("utf-8")).hexdigest()[:12]
    dir_name = f"{calibration_name}_{model_digest}_{config_hash}"
    if not create_hash:
        dir_name = f"{calibration_name}_{model_digest}"
    return per_model_root / dir_name


def get_checkpoint_metadata(
    models: List[Dict[str, Any]],
    datasets: List[Dict[str, Any]],
    wagering_method: Dict[str, Any],
    aggregation: Dict[str, Any],
    calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get metadata dictionary for logging/analytics.
    
    Args:
        models: List of model configs
        datasets: List of dataset configs
        wagering_method: Wagering method config
        aggregation: Aggregation function config
        
    Returns:
        Dictionary with metadata for analytics
        
    Raises:
        ValueError: If any config is invalid
    """
    if not models:
        raise ValueError("Must provide at least one model")
    if not datasets:
        raise ValueError("Must provide at least one dataset")
    
    model_names = [m.get("path", "unknown") for m in models]
    dataset_names = [get_dataset_name(d) for d in datasets]
    
    metadata = {
        "models": model_names,
        "model_count": len(models),
        "datasets": dataset_names,
        "dataset_count": len(datasets),
        "wagering_method": wagering_method.get("name", "unknown"),
        "wagering_config": wagering_method.get("config", {}),
        "aggregation_method": aggregation.get("name", "unknown"),
        "aggregation_config": aggregation.get("config", {}),
    }

    if calibration:
        metadata["calibrated"] = True
        metadata["calibration_method"] = calibration.get("name", "adaptive_temperature_scaling")
        metadata["calibration_config"] = calibration
    else:
        metadata["calibrated"] = False

    return metadata


def generate_calibration_dir(
    base_dir: Path,
    models: List[Dict[str, Any]],
    datasets: List[Dict[str, Any]],
    calibration_config: Dict[str, Any],
    create_hash: bool = True,
) -> Path:
    """Generate a unique artifact directory for cached-logit temperature calibration.

    When **no** calibration dataset config is PubMedQA, the directory name and content hash
    use **sorted unique** model paths only: ensemble order and duplicate slots do not create
    a separate artifact, as long as the calibration dataset configs match.

    When **any** calibration dataset is PubMedQA, the full ordered model list is used (mixed-
    context routing depends on slot index and count).
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if not models:
        raise ValueError("Must provide at least one model")
    if not datasets:
        raise ValueError("Must provide at least one dataset")

    ordered_model_paths = [m["path"] for m in models if "path" in m]
    if not ordered_model_paths:
        raise ValueError("No valid model paths found in model configs")

    # PubMedQA: ensemble order and duplicate slots matter (mixed-context routing).
    # Non-PubMedQA: reuse calibration across order/duplicates; key by unique paths only.
    pubmedqa_calibration = calibration_dataset_configs_include_pubmedqa(datasets)
    if pubmedqa_calibration:
        models_for_hash = ordered_model_paths
        model_names = [get_model_name(path) for path in ordered_model_paths]
        indexed_model_names = [f"m{idx}_{name}" for idx, name in enumerate(model_names)]
    else:
        models_for_hash = sorted(set(ordered_model_paths))
        model_names = [get_model_name(path) for path in models_for_hash]
        indexed_model_names = [f"u{idx}_{name}" for idx, name in enumerate(model_names)]

    dataset_names = [get_dataset_name(d) for d in datasets]
    calibration_name = sanitize_name(calibration_config.get("name", "adaptive_temperature_scaling"), max_length=25)

    components = [
        f"models_{'_'.join(indexed_model_names)}",
        f"datasets_{'_'.join(sorted(set(dataset_names)))}",
        f"calibration_{calibration_name}",
    ]

    normalized_payload = {
        "models": models_for_hash,
        "datasets": _stable_json_value(datasets_for_checkpoint_hash(datasets)),
        "calibration": _stable_json_value(_strip_shuffle_seed_fields(calibration_config)),
    }
    serialized = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.md5(serialized.encode("utf-8")).hexdigest()[:8]

    dir_name = "_".join(components)
    if create_hash:
        dir_name = f"{dir_name}_{config_hash}"

    # Avoid filesystem component limits (typically 255 chars) when many models are listed.
    if len(dir_name) > 220:
        models_digest = hashlib.md5("|".join(models_for_hash).encode("utf-8")).hexdigest()[:8]
        datasets_digest = hashlib.md5("|".join(sorted(set(dataset_names))).encode("utf-8")).hexdigest()[:8]
        compact_dir_name = (
            f"models_{len(models_for_hash)}m_{models_digest}_"
            f"datasets_{len(set(dataset_names))}d_{datasets_digest}_"
            f"calibration_{calibration_name}"
        )
        dir_name = f"{compact_dir_name}_{config_hash}" if create_hash else compact_dir_name

    return base_dir / dir_name
