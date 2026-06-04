"""
Configuration loading utilities.

Simplified version that loads configs without complex fallback logic.
All paths should be explicit and errors are raised immediately if files are not found.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


def _normalize_option_tokens(config: Dict[str, Any]) -> None:
    """Normalize option_tokens so YAML booleans like YES/NO don't break downstream code."""
    raw_tokens = config.get("option_tokens")
    if not isinstance(raw_tokens, list):
        return

    normalized = []
    for token in raw_tokens:
        if isinstance(token, bool):
            normalized.append("YES" if token else "NO")
        else:
            normalized.append(str(token))

    config["option_tokens"] = normalized


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    
    Raises:
        FileNotFoundError: If file does not exist
        yaml.YAMLError: If file is not valid YAML
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"Config file is empty: {file_path}")
        return config


def resolve_config_path(path: str | Path, base_dir: Path) -> Path:
    """
    Resolve a config path (can be relative or absolute).
    
    Args:
        path: Path to config file (relative or absolute)
        base_dir: Base directory for resolving relative paths
        
    Returns:
        Resolved Path object
        
    Raises:
        FileNotFoundError: If resolved path does not exist
    """
    path = Path(path)
    
    if path.is_absolute():
        resolved = path
    else:
        resolved = base_dir / path
    
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved} (base_dir: {base_dir})")
    
    return resolved


def _merge_model_includes(config: Dict[str, Any], base_dir: Path) -> None:
    """Resolve model include directives in-place."""
    if "_include_models" not in config:
        return

    if not isinstance(config["_include_models"], list):
        raise ValueError(f"_include_models must be a list, got {type(config['_include_models'])}")

    model_configs = []
    for model_path in config["_include_models"]:
        model_file = resolve_config_path(model_path, base_dir)
        model_config = load_yaml_file(model_file)
        model_configs.append(model_config)

    config["models"] = model_configs
    del config["_include_models"]


def _merge_dataset_includes(
    config: Dict[str, Any],
    base_dir: Path,
    include_key: str,
    target_key: str,
) -> None:
    """Resolve dataset include directives in-place."""
    if include_key not in config:
        return

    if not isinstance(config[include_key], list):
        raise ValueError(f"{include_key} must be a list, got {type(config[include_key])}")

    override_configs = config.get(target_key, [])
    dataset_configs = []
    for idx, dataset_path in enumerate(config[include_key]):
        dataset_file = resolve_config_path(dataset_path, base_dir)
        dataset_config = load_yaml_file(dataset_file)

        if idx < len(override_configs) and isinstance(override_configs[idx], dict):
            dataset_config.update(override_configs[idx])

        dataset_configs.append(dataset_config)

    config[target_key] = dataset_configs
    del config[include_key]


def _merge_ood_include(config: Dict[str, Any], base_dir: Path) -> None:
    """Resolve OOD dataset include directives in-place."""
    include_single = config.get("_include_ood_dataset")
    include_multi = config.get("_include_ood_datasets")

    if include_single is None and include_multi is None:
        return

    if include_single is not None and include_multi is not None:
        raise ValueError("Use either _include_ood_dataset or _include_ood_datasets, not both")

    if include_multi is not None:
        if not isinstance(include_multi, list):
            raise ValueError(f"_include_ood_datasets must be a list, got {type(include_multi)}")

        override_configs = config.get("ood_datasets", [])
        ood_configs: List[Dict[str, Any]] = []
        for idx, dataset_path in enumerate(include_multi):
            dataset_file = resolve_config_path(dataset_path, base_dir)
            dataset_config = load_yaml_file(dataset_file)

            if idx < len(override_configs) and isinstance(override_configs[idx], dict):
                dataset_config.update(override_configs[idx])

            ood_configs.append(dataset_config)

        config["ood_datasets"] = ood_configs
        del config["_include_ood_datasets"]
        return

    # Backward compatibility: allow a single include path or a list under _include_ood_dataset.
    if isinstance(include_single, list):
        include_paths = include_single
        override_configs = config.get("ood_datasets", [])
        ood_configs: List[Dict[str, Any]] = []
        for idx, dataset_path in enumerate(include_paths):
            dataset_file = resolve_config_path(dataset_path, base_dir)
            dataset_config = load_yaml_file(dataset_file)

            if idx < len(override_configs) and isinstance(override_configs[idx], dict):
                dataset_config.update(override_configs[idx])

            ood_configs.append(dataset_config)

        config["ood_datasets"] = ood_configs
        del config["_include_ood_dataset"]
        return

    dataset_file = resolve_config_path(include_single, base_dir)
    ood_config = load_yaml_file(dataset_file)

    if "ood_dataset" in config and isinstance(config["ood_dataset"], dict):
        ood_config.update(config["ood_dataset"])

    config["ood_dataset"] = ood_config
    del config["_include_ood_dataset"]


def _load_config_with_includes(
    config_path: Path,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load a config file and resolve any supported include directives."""
    config_path = Path(config_path)
    if base_dir is None:
        base_dir = config_path.parent

    config = load_yaml_file(config_path)

    _merge_model_includes(config, base_dir)
    _merge_dataset_includes(config, base_dir, "_include_datasets", "datasets")
    _merge_dataset_includes(config, base_dir, "_include_test_datasets", "test_datasets")
    _merge_ood_include(config, base_dir)

    if "_include_calibration" in config:
        calibration_file = resolve_config_path(config["_include_calibration"], base_dir)
        calibration_config = _load_config_with_includes(
            calibration_file,
            base_dir=calibration_file.parent,
        )
        override_config = config.get("calibration")
        if isinstance(override_config, dict):
            calibration_config.update(override_config)
        config["calibration"] = calibration_config
        del config["_include_calibration"]

    return config


def load_and_merge_configs(
    main_config_path: Path,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load main config and merge referenced config files.
    
    Supports:
    - `_include_models`: List of model config file paths (relative to base_dir)
    - `_include_datasets`: List of dataset config file paths (relative to base_dir)
    - `_include_test_datasets`: List of test dataset config file paths
    
    Args:
        main_config_path: Path to main config file
        base_dir: Base directory for resolving relative paths (defaults to main_config_path.parent)
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        FileNotFoundError: If any referenced config file is not found
        ValueError: If config is invalid
    """
    main_config_path = Path(main_config_path)
    if base_dir is None:
        base_dir = main_config_path.parent

    config = _load_config_with_includes(main_config_path, base_dir=base_dir)
    
    # Validate required keys
    if "models" not in config or not config["models"]:
        raise ValueError("Config must specify models")
    has_single_method = "wagering_method" in config
    phase_shift_cfg = config.get("phase_shift")
    has_phase_methods = isinstance(phase_shift_cfg, dict) and bool(phase_shift_cfg.get("methods"))
    if not has_single_method and not has_phase_methods:
        raise ValueError(
            "Config must specify wagering_method or phase_shift.methods"
        )
    if "aggregation" not in config:
        raise ValueError("Config must specify aggregation")

    _normalize_option_tokens(config)
    
    return config
