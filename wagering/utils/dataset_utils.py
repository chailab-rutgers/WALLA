"""
Dataset loading utilities.

Simplified version with strict error handling.
"""

import logging
import sys
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Sequence

import numpy as np

# Ensure local project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wagering.core.dataset import Dataset

log = logging.getLogger(__name__)

# Dataset YAML keys that tune training/optimization but do not change loaded data /
# prompt construction / logits-cache identity.
#
# These are sometimes (accidentally) placed under dataset blocks in YAML; we still
# want cached logits/hidden-states to be reusable across such changes.
_DATASET_CONFIG_EPHEMERAL_KEYS = frozenset(
    {
        "max_batches",
        "max_training_batches",
        "frozen_model_indices",
        "inactive_model_indices",
    }
)

# Strips from cache signatures for shared-source 6:2:2 view so on-disk cache matches
# a plain load of the same HF split (no tripartition metadata in the key).
_TRIPARTITION_EXCLUDED_CACHE_KEYS = frozenset(
    {
        "source_tripartition_ratios",
    }
)


def datasets_for_checkpoint_hash(dataset_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shallow copies of dataset dicts with training-only keys removed for stable directory hashes."""
    out: List[Dict[str, Any]] = []
    for cfg in dataset_configs:
        if not isinstance(cfg, dict):
            out.append(cfg)
            continue
        out.append({k: v for k, v in cfg.items() if k not in _DATASET_CONFIG_EPHEMERAL_KEYS})
    return out


def _stable_cache_value(value: Any) -> Any:
    """Normalize nested values to JSON-stable structures for cache signatures."""
    if isinstance(value, dict):
        return {str(k): _stable_cache_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_cache_value(v) for v in value]
    return value


def _build_dataset_cache_config_signature(
    dataset_cfg: Dict[str, Any],
    *,
    dataset_name: str,
    load_split: str,
    resolved_path: Any,
    resolved_split: str,
    resolved_config_name: Optional[str],
    dataset_target_split: Optional[str],
    random_seed: Optional[int],
) -> Dict[str, Any]:
    """Build a deterministic dataset configuration signature for cache keys."""
    dataset_for_sig = {k: v for k, v in dataset_cfg.items() if k not in _DATASET_CONFIG_EPHEMERAL_KEYS}
    signature_payload: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "load_split": load_split,
        "resolved_path": resolved_path,
        "resolved_split": resolved_split,
        "resolved_config_name": resolved_config_name,
        "dataset_target_split": dataset_target_split,
        # Keep random seed in signature only if explicitly pinned in dataset config.
        "split_seed": dataset_cfg.get("split_seed") if "split_seed" in dataset_cfg else None,
        "dataset_config": dict(dataset_for_sig),
    }
    if "split_seed" not in dataset_cfg:
        # Ignore global run-time seed to maximize cache reuse across shuffle sweeps.
        signature_payload["runtime_random_seed"] = None
    else:
        signature_payload["runtime_random_seed"] = random_seed

    normalized_payload = _stable_cache_value(signature_payload)
    serialized = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "schema_version": 2,
        "payload": normalized_payload,
        "signature": hashlib.md5(serialized.encode("utf-8")).hexdigest(),
    }


def _resolve_hf_identity_and_split(dataset_cfg: Dict[str, Any], *, load_split: str) -> Tuple[str, str]:
    """
    Resolve a stable HF identity + the exact HF split string used by ``Dataset.load`` for this cfg.

    This mirrors the resolution logic in ``load_datasets_from_config`` (config_name +
    actual_split selection) but does not load any data.
    """
    dataset_path = dataset_cfg.get("name")
    if dataset_path is None:
        return "", ""

    is_pubmedqa = _is_pubmedqa_dataset_config(dataset_cfg)
    is_race = _is_race_dataset_config(dataset_cfg)

    if is_pubmedqa:
        config_name = dataset_cfg.get(
            "pubmedqa_source_config_name",
            dataset_cfg.get(
                "train_config_name",
                dataset_cfg.get(
                    "eval_config_name",
                    dataset_cfg.get("config_name", "pqa_artificial"),
                ),
            ),
        )
        actual_split = dataset_cfg.get("pubmedqa_source_split", "train")
    elif is_race:
        config_name = dataset_cfg.get(
            "race_source_config_name",
            dataset_cfg.get(
                "train_config_name",
                dataset_cfg.get(
                    "eval_config_name",
                    dataset_cfg.get("config_name"),
                ),
            ),
        )
        actual_split = dataset_cfg.get(
            "race_source_split",
            dataset_cfg.get("train_split", dataset_cfg.get("eval_split", "test")),
        )
    elif str(load_split) == "train":
        config_name = dataset_cfg.get("train_config_name", dataset_cfg.get("config_name"))
        actual_split = dataset_cfg.get("train_split", "train")
    else:
        config_name = dataset_cfg.get(
            "eval_config_name",
            dataset_cfg.get("test_config_name", dataset_cfg.get("config_name")),
        )
        actual_split = dataset_cfg.get("eval_split", "test")

    if isinstance(dataset_path, (list, tuple)) and dataset_path:
        path0 = str(dataset_path[0])
        cfg0 = str(dataset_path[1]) if len(dataset_path) > 1 else ""
        hf_id = f"{path0},{cfg0}"
    else:
        hf_id = str(dataset_path)
        if config_name:
            hf_id = f"{hf_id},{str(config_name)}"

    return hf_id, str(actual_split)


def _shared_source_tripartition_peer_matched(
    dataset_cfg: Dict[str, Any],
    *,
    load_split: str,
    tripartition_peer_dataset_configs: Optional[Sequence[Dict[str, Any]]],
    infer_eval_split_train_without_peer: bool,
) -> bool:
    """
    True when shared-source 8:1:1 should apply.

    Either an explicit training peer matches (same resolved HF identity and aligned split strings),
    or (eval-only) ``infer_eval_split_train_without_peer`` is True, ``load_split`` is
    ``test``, and ``eval_split`` is ``train`` (common when the official test split has
    no labels and eval reuses HF ``train``).
    """
    h1, s1 = _resolve_hf_identity_and_split(dataset_cfg, load_split=str(load_split))
    if not h1:
        return False
    if tripartition_peer_dataset_configs:
        peer_split = "test" if str(load_split) == "train" else "train"
        for peer in tripartition_peer_dataset_configs:
            if not isinstance(peer, dict):
                continue
            h2, s2 = _resolve_hf_identity_and_split(peer, load_split=peer_split)
            if not h2 or h1 != h2:
                continue
            if str(s1) == str(s2):
                return True
        return False
    if infer_eval_split_train_without_peer and str(load_split) == "test":
        return str(dataset_cfg.get("eval_split", "test")).lower() == "train"
    return False


def _build_tripartition_full_source_cache_config(
    dataset_cfg: Dict[str, Any],
    *,
    cache_dataset_name: str,
    resolved_path: Any,
    resolved_huggingface_split: str,
    resolved_config_name: Optional[str],
    random_seed: Optional[int],
) -> Dict[str, Any]:
    """
    Logits/hidden-states cache identity for the *full* HF source split, aligned with
    a normal (non-tripartition) training load of that split in ``load_split: train`` mode.
    """
    for_sig: Dict[str, Any] = {}
    for k, v in dataset_cfg.items():
        if k in _DATASET_CONFIG_EPHEMERAL_KEYS or k in _TRIPARTITION_EXCLUDED_CACHE_KEYS:
            continue
        for_sig[k] = v
    # Align with a normal training-stub config: a single train_split for the source HF split.
    for_sig.pop("train_split", None)
    for_sig.pop("eval_split", None)
    for_sig["train_split"] = str(resolved_huggingface_split)
    for_sig["display_name"] = str(cache_dataset_name)

    signature_payload: Dict[str, Any] = {
        "dataset_name": cache_dataset_name,
        "load_split": "train",
        "resolved_path": resolved_path,
        "resolved_split": resolved_huggingface_split,
        "resolved_config_name": resolved_config_name,
        "dataset_target_split": None,
        "split_seed": dataset_cfg.get("split_seed") if "split_seed" in dataset_cfg else None,
        "dataset_config": for_sig,
    }
    if "split_seed" not in dataset_cfg:
        signature_payload["runtime_random_seed"] = None
    else:
        signature_payload["runtime_random_seed"] = random_seed

    normalized_payload = _stable_cache_value(signature_payload)
    serialized = json.dumps(
        normalized_payload, sort_keys=True, separators=(",", ":"), default=str
    )
    return {
        "schema_version": 2,
        "payload": normalized_payload,
        "signature": hashlib.md5(serialized.encode("utf-8")).hexdigest(),
    }


def _is_pubmedqa_dataset_config(dataset_cfg: Dict[str, Any]) -> bool:
    """Return True when a dataset config targets PubMedQA."""
    fields = [
        dataset_cfg.get("name", ""),
        dataset_cfg.get("display_name", ""),
        dataset_cfg.get("config_name", ""),
        dataset_cfg.get("train_config_name", ""),
        dataset_cfg.get("eval_config_name", ""),
        dataset_cfg.get("test_config_name", ""),
        dataset_cfg.get("pubmedqa_source_config_name", ""),
    ]
    normalized = " ".join(str(field).lower() for field in fields if field is not None)
    return "pubmedqa" in normalized or "pubmed_qa" in normalized


def _is_race_dataset_config(dataset_cfg: Dict[str, Any]) -> bool:
    """Return True when a dataset config targets RACE."""
    fields = [
        dataset_cfg.get("name", ""),
        dataset_cfg.get("display_name", ""),
        dataset_cfg.get("config_name", ""),
        dataset_cfg.get("train_config_name", ""),
        dataset_cfg.get("eval_config_name", ""),
        dataset_cfg.get("test_config_name", ""),
    ]
    normalized = " ".join(str(field).lower() for field in fields if field is not None)
    if "eleutherai/race" in normalized:
        return True
    padded = f" {normalized} "
    return " race " in padded


def calibration_dataset_configs_include_pubmedqa(dataset_configs: Sequence[Dict[str, Any]]) -> bool:
    """True if any calibration dataset uses mixed-context routing (PubMedQA or RACE)."""
    return any(
        _is_pubmedqa_dataset_config(cfg) or _is_race_dataset_config(cfg)
        for cfg in dataset_configs
    )


def _normalize_pubmedqa_split_ratios(raw_ratios: Any) -> Tuple[float, float, float]:
    """Normalize PubMedQA split ratios to a valid (train, val, test) tuple."""
    default_ratios = (0.8, 0.1, 0.1)
    if raw_ratios is None:
        return default_ratios

    if not isinstance(raw_ratios, Sequence) or len(raw_ratios) != 3:
        log.warning(
            "Invalid pubmedqa_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    try:
        ratio_array = np.array([float(v) for v in raw_ratios], dtype=np.float64)
    except (TypeError, ValueError):
        log.warning(
            "Could not parse pubmedqa_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    if np.any(ratio_array < 0) or not np.any(ratio_array > 0):
        log.warning(
            "Non-positive pubmedqa_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    ratio_array = ratio_array / ratio_array.sum()
    return tuple(float(v) for v in ratio_array.tolist())


def _normalize_race_split_ratios(raw_ratios: Any) -> Tuple[float, float, float]:
    """Normalize RACE split ratios to a valid (train, val, test) tuple."""
    default_ratios = (0.6, 0.2, 0.2)
    if raw_ratios is None:
        return default_ratios

    if not isinstance(raw_ratios, Sequence) or len(raw_ratios) != 3:
        log.warning(
            "Invalid race_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    try:
        ratio_array = np.array([float(v) for v in raw_ratios], dtype=np.float64)
    except (TypeError, ValueError):
        log.warning(
            "Could not parse race_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    if np.any(ratio_array < 0) or not np.any(ratio_array > 0):
        log.warning(
            "Non-positive race_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    ratio_array = ratio_array / ratio_array.sum()
    return tuple(float(v) for v in ratio_array.tolist())


def _normalize_shared_source_tripartition_ratios(raw_ratios: Any) -> Tuple[float, float, float]:
    """Normalize shared-source tripartition ratios to a valid (train, val, test) tuple."""
    default_ratios = (0.8, 0.1, 0.1)
    if raw_ratios is None:
        return default_ratios

    if not isinstance(raw_ratios, Sequence) or len(raw_ratios) != 3:
        log.warning(
            "Invalid source_tripartition_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    try:
        ratio_array = np.array([float(v) for v in raw_ratios], dtype=np.float64)
    except (TypeError, ValueError):
        log.warning(
            "Could not parse source_tripartition_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    if np.any(ratio_array < 0) or not np.any(ratio_array > 0):
        log.warning(
            "Non-positive source_tripartition_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    ratio_array = ratio_array / ratio_array.sum()
    return tuple(float(v) for v in ratio_array.tolist())


def _subset_pubmedqa_dataset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    """Apply an index subset while keeping PubMedQA prompt variants aligned."""
    index_list = [int(i) for i in indices.tolist()]
    with_context_prompts = getattr(dataset, "pubmedqa_with_context_x", None)
    without_context_prompts = getattr(dataset, "pubmedqa_without_context_x", None)

    dataset.select(index_list)

    if isinstance(with_context_prompts, list):
        dataset.pubmedqa_with_context_x = [with_context_prompts[i] for i in index_list]
    if isinstance(without_context_prompts, list):
        dataset.pubmedqa_without_context_x = [without_context_prompts[i] for i in index_list]

    return dataset


def _subset_race_dataset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    """Apply an index subset while keeping RACE prompt variants aligned."""
    index_list = [int(i) for i in indices.tolist()]
    with_context_prompts = getattr(dataset, "race_with_context_x", None)
    without_context_prompts = getattr(dataset, "race_without_context_x", None)

    dataset.select(index_list)

    if isinstance(with_context_prompts, list):
        dataset.race_with_context_x = [with_context_prompts[i] for i in index_list]
    if isinstance(without_context_prompts, list):
        dataset.race_without_context_x = [without_context_prompts[i] for i in index_list]

    return dataset


def _apply_race_split(
    dataset: Dataset,
    dataset_name: str,
    target_split: str,
    split_seed: int,
    split_ratios: Tuple[float, float, float],
    requested_size: Optional[int],
) -> Dataset:
    """Deterministically split a single-source RACE split into train/val/test partitions."""
    split_aliases = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "train_val": "train_val",
        "train+val": "train_val",
    }
    normalized_target = split_aliases.get(str(target_split).strip().lower())
    if normalized_target is None:
        raise ValueError(
            f"Unsupported RACE target split '{target_split}'. "
            "Use one of: train, validation, test, train_val."
        )

    total_examples = len(dataset.x)
    if total_examples <= 0:
        raise ValueError(f"RACE dataset '{dataset_name}' is empty before splitting")

    rng = np.random.RandomState(int(split_seed))
    all_indices = np.arange(total_examples, dtype=np.int64)
    rng.shuffle(all_indices)

    ratio_array = np.array(split_ratios, dtype=np.float64)
    raw_counts = ratio_array * float(total_examples)
    split_counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(total_examples - split_counts.sum())
    if remainder > 0:
        residual_order = np.argsort(-(raw_counts - split_counts))
        for idx in residual_order[:remainder]:
            split_counts[idx] += 1

    train_count, val_count, test_count = [int(v) for v in split_counts.tolist()]

    train_indices = np.array(all_indices[:train_count], copy=True)
    val_start = train_count
    val_end = train_count + val_count
    val_indices = np.array(all_indices[val_start:val_end], copy=True)
    test_indices = np.array(all_indices[val_end:], copy=True)

    split_indices_map = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
        "train_val": np.concatenate([train_indices, val_indices]),
    }
    selected_indices = np.array(split_indices_map[normalized_target], copy=True)
    rng.shuffle(selected_indices)

    if requested_size is not None:
        requested_size_int = int(requested_size)
        if requested_size_int <= 0:
            raise ValueError(
                f"Invalid size={requested_size_int} for RACE dataset '{dataset_name}'. "
                "Expected a positive integer."
            )
        if requested_size_int < selected_indices.shape[0]:
            selected_indices = selected_indices[:requested_size_int]

    dataset = _subset_race_dataset(dataset, selected_indices)

    dataset.race_split_source = "single_source_split"
    dataset.race_balanced_split = normalized_target
    dataset.race_split_seed = int(split_seed)
    dataset.race_split_ratios = tuple(float(v) for v in ratio_array.tolist())
    dataset.race_split_counts = {
        "source_examples": int(total_examples),
        "train_examples": train_count,
        "validation_examples": val_count,
        "test_examples": test_count,
        "selected_examples": int(len(dataset.x)),
    }

    log.info(
        "RACE deterministic split for %s: split=%s, seed=%d, source=%d, "
        "counts(train/val/test)=(%d/%d/%d), selected=%d",
        dataset_name,
        normalized_target,
        int(split_seed),
        int(total_examples),
        train_count,
        val_count,
        test_count,
        int(len(dataset.x)),
    )

    return dataset


def _apply_shared_source_tripartition(
    dataset: Dataset,
    dataset_name: str,
    target_split: str,
    split_seed: int,
    split_ratios: Tuple[float, float, float],
    requested_size: Optional[int],
) -> Dataset:
    """
    Deterministically partition a single loaded HF split into train/val/test (default 8:1:1),
    with ``cache_source_row_indices`` mapping each view row to a row in the *full* source
    (so option-logit caches can stay keyed as the unpartitioned split).
    """
    split_aliases = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "train_val": "train_val",
        "train+val": "train_val",
    }
    normalized_target = split_aliases.get(str(target_split).strip().lower())
    if normalized_target is None:
        raise ValueError(
            f"Unsupported shared-source tripartition target '{target_split}'. "
            "Use one of: train, validation, test, train_val."
        )

    total_examples = len(dataset.x)
    if total_examples <= 0:
        raise ValueError(f"Dataset '{dataset_name}' is empty before tripartition")

    rng = np.random.RandomState(int(split_seed))
    all_indices = np.arange(total_examples, dtype=np.int64)
    rng.shuffle(all_indices)

    ratio_array = np.array(split_ratios, dtype=np.float64)
    raw_counts = ratio_array * float(total_examples)
    split_counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(total_examples - split_counts.sum())
    if remainder > 0:
        residual_order = np.argsort(-(raw_counts - split_counts))
        for idx in residual_order[:remainder]:
            split_counts[idx] += 1

    train_count, val_count, test_count = [int(v) for v in split_counts.tolist()]

    train_indices = np.array(all_indices[:train_count], copy=True)
    val_start = train_count
    val_end = train_count + val_count
    val_indices = np.array(all_indices[val_start:val_end], copy=True)
    test_indices = np.array(all_indices[val_end:], copy=True)

    split_indices_map = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
        "train_val": np.concatenate([train_indices, val_indices]),
    }
    selected_indices = np.array(split_indices_map[normalized_target], copy=True).astype(np.int64, copy=True)

    preserve_contiguous = normalized_target == "train_val"
    if not preserve_contiguous:
        rng.shuffle(selected_indices)

    if requested_size is not None:
        requested_size_int = int(requested_size)
        if requested_size_int <= 0:
            raise ValueError(
                f"Invalid size={requested_size_int} for shared-source tripartition on '{dataset_name}'."
            )
        if requested_size_int < selected_indices.shape[0]:
            if preserve_contiguous:
                log.warning(
                    "Ignoring size=%d for '%s' train_val to preserve train|val contiguity.",
                    requested_size_int,
                    dataset_name,
                )
            else:
                selected_indices = selected_indices[:requested_size_int].astype(
                    np.int64, copy=True
                )

    as_list = [int(i) for i in selected_indices.tolist()]
    dataset.select(as_list)
    # Row i of the view reads/writes full-split cache at global row selected_indices[i].
    dataset.cache_source_num_examples = int(total_examples)
    dataset.cache_source_row_indices = selected_indices

    dataset.source_tripartition_target = normalized_target
    dataset.source_tripartition_split_seed = int(split_seed)
    dataset.source_tripartition_split_ratios = tuple(float(v) for v in ratio_array.tolist())
    dataset.source_tripartition_counts = {
        "source_examples": int(total_examples),
        "train_examples": train_count,
        "validation_examples": val_count,
        "test_examples": test_count,
        "selected_examples": int(len(dataset.x)),
    }

    if preserve_contiguous:
        dataset.source_tripartition_contiguous_train_val = True
        dataset.source_tripartition_train_val_boundary = int(train_indices.shape[0])
        tv = float(train_indices.shape[0] + val_indices.shape[0])
        dataset.source_tripartition_val_ratio = (
            float(val_indices.shape[0]) / tv if tv > 0 else 0.0
        )
    else:
        dataset.source_tripartition_contiguous_train_val = False
        dataset.source_tripartition_train_val_boundary = 0
        dataset.source_tripartition_val_ratio = None

    log.info(
        "Shared-source 8:1:1 for %s: target=%s, seed=%d, source=%d, "
        "counts(train/val/test)=(%d/%d/%d), view=%d",
        dataset_name,
        normalized_target,
        int(split_seed),
        int(total_examples),
        train_count,
        val_count,
        test_count,
        int(len(dataset.x)),
    )

    return dataset


def _apply_pubmedqa_balanced_split(
    dataset: Dataset,
    dataset_name: str,
    target_split: str,
    split_seed: int,
    split_ratios: Tuple[float, float, float],
    requested_size: Optional[int],
) -> Dataset:
    """
    Balance YES/NO labels, perform deterministic 6:2:2-style splitting, then subset.

    The split is performed on balanced labels first, then the target partition is selected.
    """
    split_aliases = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "train_val": "train_val",
        "train+val": "train_val",
    }
    normalized_target = split_aliases.get(str(target_split).strip().lower())
    if normalized_target is None:
        raise ValueError(
            f"Unsupported PubMedQA target split '{target_split}'. "
            "Use one of: train, validation, test, train_val."
        )

    labels = np.array([str(label).strip().upper() for label in dataset.y], dtype=object)
    yes_indices = np.where(labels == "YES")[0]
    no_indices = np.where(labels == "NO")[0]

    if yes_indices.size == 0 or no_indices.size == 0:
        raise ValueError(
            f"PubMedQA dataset '{dataset_name}' must contain both YES and NO labels "
            f"(found YES={yes_indices.size}, NO={no_indices.size})."
        )

    min_class_count = int(min(yes_indices.size, no_indices.size))
    rng = np.random.RandomState(int(split_seed))

    yes_pool = np.array(yes_indices, copy=True)
    no_pool = np.array(no_indices, copy=True)
    rng.shuffle(yes_pool)
    rng.shuffle(no_pool)
    yes_pool = yes_pool[:min_class_count]
    no_pool = no_pool[:min_class_count]

    ratio_array = np.array(split_ratios, dtype=np.float64)
    raw_counts = ratio_array * float(min_class_count)
    split_counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(min_class_count - split_counts.sum())
    if remainder > 0:
        residual_order = np.argsort(-(raw_counts - split_counts))
        for idx in residual_order[:remainder]:
            split_counts[idx] += 1

    train_count, val_count, test_count = [int(v) for v in split_counts.tolist()]

    train_yes = yes_pool[:train_count]
    val_yes = yes_pool[train_count: train_count + val_count]
    test_yes = yes_pool[train_count + val_count:]

    train_no = no_pool[:train_count]
    val_no = no_pool[train_count: train_count + val_count]
    test_no = no_pool[train_count + val_count:]

    train_indices = np.concatenate([train_yes, train_no])
    val_indices = np.concatenate([val_yes, val_no])
    test_indices = np.concatenate([test_yes, test_no])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    split_indices_map = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
        "train_val": np.concatenate([train_indices, val_indices]),
    }
    selected_indices = np.array(split_indices_map[normalized_target], copy=True)
    rng.shuffle(selected_indices)

    if requested_size is not None:
        requested_size_int = int(requested_size)
        if requested_size_int <= 0:
            raise ValueError(
                f"Invalid size={requested_size_int} for PubMedQA dataset '{dataset_name}'. "
                "Expected a positive integer."
            )

        if requested_size_int < selected_indices.shape[0]:
            selected_labels = labels[selected_indices]
            split_yes_indices = selected_indices[selected_labels == "YES"]
            split_no_indices = selected_indices[selected_labels == "NO"]
            per_class_limit = min(
                split_yes_indices.shape[0],
                split_no_indices.shape[0],
                requested_size_int // 2,
            )

            if per_class_limit <= 0:
                raise ValueError(
                    f"Requested size={requested_size_int} is too small to keep YES/NO balance "
                    f"for PubMedQA dataset '{dataset_name}'."
                )

            rng.shuffle(split_yes_indices)
            rng.shuffle(split_no_indices)

            selected_indices = np.concatenate(
                [split_yes_indices[:per_class_limit], split_no_indices[:per_class_limit]]
            )
            rng.shuffle(selected_indices)

            balanced_size = int(selected_indices.shape[0])
            if balanced_size < requested_size_int:
                log.warning(
                    "Requested size=%d for PubMedQA dataset '%s' adjusted to %d to preserve YES/NO balance.",
                    requested_size_int,
                    dataset_name,
                    balanced_size,
                )

    dataset = _subset_pubmedqa_dataset(dataset, selected_indices)

    dataset.pubmedqa_balanced_source = "pqa_artificial"
    dataset.pubmedqa_balanced_split = normalized_target
    dataset.pubmedqa_split_seed = int(split_seed)
    dataset.pubmedqa_split_ratios = tuple(float(v) for v in ratio_array.tolist())
    dataset.pubmedqa_balanced_counts = {
        "source_yes": int(yes_indices.size),
        "source_no": int(no_indices.size),
        "balanced_per_label": int(min_class_count),
        "train_per_label": train_count,
        "validation_per_label": val_count,
        "test_per_label": test_count,
        "selected_examples": int(len(dataset.x)),
    }

    if normalized_target == "train_val":
        train_val_size = train_count + val_count
        if train_val_size > 0:
            dataset.pubmedqa_train_val_split_ratio = float(val_count / train_val_size)

    log.info(
        "PubMedQA balanced split for %s: split=%s, seed=%d, source_yes=%d, source_no=%d, "
        "balanced_per_label=%d, per_label_counts(train/val/test)=(%d/%d/%d), selected=%d",
        dataset_name,
        normalized_target,
        int(split_seed),
        int(yes_indices.size),
        int(no_indices.size),
        int(min_class_count),
        train_count,
        val_count,
        test_count,
        int(len(dataset.x)),
    )

    return dataset


def _apply_balanced_binary_split(
    dataset: Dataset,
    dataset_name: str,
    target_split: str,
    split_seed: int,
    split_ratios: Tuple[float, float, float],
    requested_size: Optional[int],
    positive_label: str,
) -> Dataset:
    """Class-balanced train/val/test partition for binary CSV labels (e.g. cluster_saturation_bayes).

    Mirrors :func:`_apply_pubmedqa_balanced_split` but uses ``positive_label`` vs the other class
    (labels compared as ``str(label).strip()``).
    """
    split_aliases = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "train_val": "train_val",
        "train+val": "train_val",
    }
    normalized_target = split_aliases.get(str(target_split).strip().lower())
    if normalized_target is None:
        raise ValueError(
            f"Unsupported balanced-binary target split '{target_split}'. "
            "Use one of: train, validation, test, train_val."
        )

    pos_marker = str(positive_label).strip()
    labels = np.array([str(label).strip() for label in dataset.y], dtype=object)
    uniq = np.unique(labels)
    if uniq.size != 2:
        raise ValueError(
            f"Balanced binary dataset '{dataset_name}' must contain exactly two label values "
            f"(found {uniq.tolist()})."
        )
    if pos_marker not in set(uniq.tolist()):
        raise ValueError(
            f"Balanced binary dataset '{dataset_name}': positive_label={positive_label!r} "
            f"not found among labels {uniq.tolist()}."
        )

    pos_indices = np.where(labels == pos_marker)[0]
    neg_indices = np.where(labels != pos_marker)[0]

    if pos_indices.size == 0 or neg_indices.size == 0:
        raise ValueError(
            f"Balanced binary dataset '{dataset_name}' must contain both classes "
            f"(found positive={pos_indices.size}, negative={neg_indices.size})."
        )

    min_class_count = int(min(pos_indices.size, neg_indices.size))
    rng = np.random.RandomState(int(split_seed))

    pos_pool = np.array(pos_indices, copy=True)
    neg_pool = np.array(neg_indices, copy=True)
    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)
    pos_pool = pos_pool[:min_class_count]
    neg_pool = neg_pool[:min_class_count]

    ratio_array = np.array(split_ratios, dtype=np.float64)
    raw_counts = ratio_array * float(min_class_count)
    split_counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(min_class_count - split_counts.sum())
    if remainder > 0:
        residual_order = np.argsort(-(raw_counts - split_counts))
        for idx in residual_order[:remainder]:
            split_counts[idx] += 1

    train_count, val_count, test_count = [int(v) for v in split_counts.tolist()]

    train_pos = pos_pool[:train_count]
    val_pos = pos_pool[train_count : train_count + val_count]
    test_pos = pos_pool[train_count + val_count :]

    train_neg = neg_pool[:train_count]
    val_neg = neg_pool[train_count : train_count + val_count]
    test_neg = neg_pool[train_count + val_count :]

    train_indices = np.concatenate([train_pos, train_neg])
    val_indices = np.concatenate([val_pos, val_neg])
    test_indices = np.concatenate([test_pos, test_neg])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    split_indices_map = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
        "train_val": np.concatenate([train_indices, val_indices]),
    }
    selected_indices = np.array(split_indices_map[normalized_target], copy=True)
    rng.shuffle(selected_indices)

    if requested_size is not None:
        requested_size_int = int(requested_size)
        if requested_size_int <= 0:
            raise ValueError(
                f"Invalid size={requested_size_int} for balanced binary dataset '{dataset_name}'. "
                "Expected a positive integer."
            )

        if requested_size_int < selected_indices.shape[0]:
            selected_labels = labels[selected_indices]
            split_pos_sel = selected_indices[selected_labels == pos_marker]
            split_neg_sel = selected_indices[selected_labels != pos_marker]
            per_class_limit = min(
                split_pos_sel.shape[0],
                split_neg_sel.shape[0],
                requested_size_int // 2,
            )

            if per_class_limit <= 0:
                raise ValueError(
                    f"Requested size={requested_size_int} is too small to keep class balance "
                    f"for balanced binary dataset '{dataset_name}'."
                )

            rng.shuffle(split_pos_sel)
            rng.shuffle(split_neg_sel)

            selected_indices = np.concatenate(
                [split_pos_sel[:per_class_limit], split_neg_sel[:per_class_limit]]
            )
            rng.shuffle(selected_indices)

            balanced_size = int(selected_indices.shape[0])
            if balanced_size < requested_size_int:
                log.warning(
                    "Requested size=%d for balanced binary dataset '%s' adjusted to %d to preserve balance.",
                    requested_size_int,
                    dataset_name,
                    balanced_size,
                )

    dataset = _subset_pubmedqa_dataset(dataset, selected_indices)

    dataset.balanced_binary_split_source = "csv"
    dataset.balanced_binary_target_split = normalized_target
    dataset.balanced_binary_split_seed = int(split_seed)
    dataset.balanced_binary_split_ratios = tuple(float(v) for v in ratio_array.tolist())
    dataset.balanced_binary_split_counts = {
        "source_positive": int(pos_indices.size),
        "source_negative": int(neg_indices.size),
        "balanced_per_label": int(min_class_count),
        "train_per_label": train_count,
        "validation_per_label": val_count,
        "test_per_label": test_count,
        "selected_examples": int(len(dataset.x)),
    }

    if normalized_target == "train_val":
        train_val_size = train_count + val_count
        if train_val_size > 0:
            dataset.balanced_binary_train_val_ratio = float(val_count / train_val_size)

    log.info(
        "Balanced binary split for %s: split=%s, seed=%d, source_pos=%d, source_neg=%d, "
        "balanced_per_label=%d, per_label_counts(train/val/test)=(%d/%d/%d), selected=%d",
        dataset_name,
        normalized_target,
        int(split_seed),
        int(pos_indices.size),
        int(neg_indices.size),
        int(min_class_count),
        train_count,
        val_count,
        test_count,
        int(len(dataset.x)),
    )

    return dataset


def load_datasets_from_config(
    dataset_configs: List[Dict[str, Any]],
    split: str = "train",
    random_seed: Optional[int] = None,
    shared_source_tripartition: bool = False,
    tripartition_peer_dataset_configs: Optional[Sequence[Dict[str, Any]]] = None,
    infer_eval_split_train_without_peer: bool = False,
    force_shared_source_tripartition: bool = False,
) -> Tuple[List[Dataset], List[str]]:
    """
    Load multiple datasets from configuration.
    
    Args:
        dataset_configs: List of dataset configuration dictionaries, each containing:
            - name: Dataset name (REQUIRED)
            - display_name: Display name for logging (optional, defaults to name)
            - text_column: Text column name (optional, default "input")
            - label_column: Label column name (optional, default "output")
            - batch_size: Batch size (optional, default 8)
            - prompt: Prompt template (optional, default "")
            - description: Dataset description (optional, default "")
            - n_shot: Number of few-shot examples (optional, default 0)
            - few_shot_split: Split to use for few-shot (optional, default "train")
            - few_shot_prompt: Few-shot prompt template (optional)
            - instruct: Whether to use instruct format (optional, default False)
            - train_split: Split name for training (optional, default "train")
            - eval_split: Split name for evaluation (optional, default "test")
            - train_config_name: HF subset/config for training mode (optional)
            - eval_config_name: HF subset/config for evaluation mode (optional)
            - test_config_name: Alias for eval_config_name (optional)
            - config_name: HF subset/config fallback for all modes (optional)
                        - pubmedqa_source_config_name: PubMedQA source subset (default "pqa_artificial")
                        - pubmedqa_source_split: PubMedQA source split (default "train")
                        - pubmedqa_train_target_split: PubMedQA target split for training phase
                            (default "train_val")
                        - pubmedqa_eval_target_split: PubMedQA target split for evaluation phase
                            (default "test")
                        - pubmedqa_split_ratios: PubMedQA train/val/test ratios (default [0.8, 0.1, 0.1])
                        - split_seed: Seed for deterministic PubMedQA splitting (optional)
            - source_tripartition_ratios: optional (train, val, test) for shared-source 8:1:1
            - size: Number of examples to load (optional, loads all if None)
            - source_size: Source cap before PubMedQA split (optional)
            - load_from_disk: Whether to load from disk (optional, default False)
            - trust_remote_code: Whether to trust remote code (optional, default False)
        shared_source_tripartition: When True and a training row matches a test row
            (same ``name`` and the same train/eval split string), the HF source split is
            partitioned 8:1:1. Logits cache keys follow the *unpartitioned* full split.
        tripartition_peer_dataset_configs: The other side's dataset list (``test_datasets`` when
            ``split`` is train, and ``datasets`` when ``split`` is test) for pair detection.
        infer_eval_split_train_without_peer: When True and the peer list is empty, still apply
            shared-source tripartition on ``split=test`` rows with ``eval_split: train`` so
            eval-only configs align with training caches (OOD loads should pass False).
        force_shared_source_tripartition: When True, apply shared-source 8:1:1 whenever
            ``shared_source_tripartition`` is enabled (except PubMedQA/RACE). This is intended
            for in-distribution evaluation configs that want test_datasets to always be the
            held-out slice of an 8:1:1 partition, even without a training peer.
        split: Split to load ("train" or "test")
        random_seed: Optional global seed used by split-sensitive dataset loaders.
        
    Returns:
        Tuple of (list of Dataset instances, list of dataset names)
        
    Raises:
        ValueError: If dataset config is invalid
    """
    if not dataset_configs:
        raise ValueError("Must provide at least one dataset config")
    
    datasets = []
    dataset_names = []
    
    for i, dataset_cfg in enumerate(dataset_configs):
        if "name" not in dataset_cfg:
            raise ValueError(f"Dataset config {i} missing required 'name' field: {dataset_cfg}")
        
        dataset_path = dataset_cfg["name"]
        is_pubmedqa = _is_pubmedqa_dataset_config(dataset_cfg)
        is_race = _is_race_dataset_config(dataset_cfg)
        requested_size = dataset_cfg.get("size", None)
        is_shared_tripartition = (
            bool(shared_source_tripartition)
            and (
                bool(force_shared_source_tripartition)
                or _shared_source_tripartition_peer_matched(
                    dataset_cfg,
                    load_split=split,
                    tripartition_peer_dataset_configs=tripartition_peer_dataset_configs,
                    infer_eval_split_train_without_peer=bool(infer_eval_split_train_without_peer),
                )
            )
            and not is_pubmedqa
            and not is_race
        )

        if is_pubmedqa:
            config_name = dataset_cfg.get(
                "pubmedqa_source_config_name",
                dataset_cfg.get(
                    "train_config_name",
                    dataset_cfg.get(
                        "eval_config_name",
                        dataset_cfg.get("config_name", "pqa_artificial"),
                    ),
                ),
            )
            actual_split = dataset_cfg.get("pubmedqa_source_split", "train")
            if split == "train":
                dataset_target_split = dataset_cfg.get("pubmedqa_train_target_split", "train_val")
            elif split == "test":
                dataset_target_split = dataset_cfg.get("pubmedqa_eval_target_split", "test")
            else:
                dataset_target_split = dataset_cfg.get("pubmedqa_validation_target_split", split)
        elif is_race:
            config_name = dataset_cfg.get(
                "race_source_config_name",
                dataset_cfg.get(
                    "train_config_name",
                    dataset_cfg.get(
                        "eval_config_name",
                        dataset_cfg.get("config_name"),
                    ),
                ),
            )
            actual_split = dataset_cfg.get(
                "race_source_split",
                dataset_cfg.get("train_split", dataset_cfg.get("eval_split", "test")),
            )
            if split == "train":
                dataset_target_split = dataset_cfg.get("race_train_target_split", "train")
            elif split == "test":
                dataset_target_split = dataset_cfg.get("race_eval_target_split", "test")
            else:
                dataset_target_split = dataset_cfg.get("race_validation_target_split", split)
        elif split == "train":
            config_name = dataset_cfg.get("train_config_name", dataset_cfg.get("config_name"))
            split_key = "train_split"
            actual_split = dataset_cfg.get(split_key, split)
            dataset_target_split = None
        else:
            config_name = dataset_cfg.get(
                "eval_config_name",
                dataset_cfg.get("test_config_name", dataset_cfg.get("config_name")),
            )
            split_key = "eval_split"
            actual_split = dataset_cfg.get(split_key, split)
            dataset_target_split = None

        shared_tripartition_target: Optional[str] = None
        if is_shared_tripartition and split == "train":
            shared_tripartition_target = "train_val"
        elif is_shared_tripartition and split == "test":
            shared_tripartition_target = "test"
        else:
            shared_tripartition_target = None

        if config_name and isinstance(dataset_path, str):
            dataset_path = [dataset_path, config_name]
        dataset_name = dataset_cfg.get(
            "display_name",
            str(dataset_path).replace("/", "_").replace("[", "").replace("]", "")
        )

        source_size = dataset_cfg.get("source_size", None) if (is_pubmedqa or is_race) else requested_size
        
        # Load dataset
        try:
            dataset_load_kwargs = {
                "batch_size": dataset_cfg.get("batch_size", 8),
                "prompt": dataset_cfg.get("prompt", ""),
                "description": dataset_cfg.get("description", ""),
                "n_shot": dataset_cfg.get("n_shot", 0),
                "few_shot_split": dataset_cfg.get("few_shot_split", "train"),
                "few_shot_prompt": dataset_cfg.get("few_shot_prompt", None),
                "instruct": dataset_cfg.get("instruct", False),
                "split": actual_split,
                "size": source_size,
                "load_from_disk": dataset_cfg.get("load_from_disk", False),
                "trust_remote_code": dataset_cfg.get("trust_remote_code", False),
            }
            # Forward optional Hugging Face JSON loader keys used by local/custom datasets
            # such as ForecastQA (name: json + data_files + field).
            if "data_files" in dataset_cfg:
                dataset_load_kwargs["data_files"] = dataset_cfg["data_files"]
            if "field" in dataset_cfg:
                dataset_load_kwargs["field"] = dataset_cfg["field"]
            if "dataset_format" in dataset_cfg:
                dataset_load_kwargs["dataset_format"] = dataset_cfg["dataset_format"]
            if "prompt_without_context" in dataset_cfg:
                dataset_load_kwargs["prompt_without_context"] = dataset_cfg["prompt_without_context"]
            if "pubmedqa_context_model_path" in dataset_cfg:
                dataset_load_kwargs["pubmedqa_context_model_path"] = dataset_cfg["pubmedqa_context_model_path"]
            # CSV helpers / mixed-context (cluster_saturation_bayes, etc.): must reach Dataset.from_csv.
            if "prompt_helper" in dataset_cfg:
                dataset_load_kwargs["prompt_helper"] = dataset_cfg["prompt_helper"]
            if "prompt_without_context_helper" in dataset_cfg:
                dataset_load_kwargs["prompt_without_context_helper"] = dataset_cfg[
                    "prompt_without_context_helper"
                ]
            if "mixed_context_routing" in dataset_cfg:
                dataset_load_kwargs["mixed_context_routing"] = dataset_cfg["mixed_context_routing"]
            if "probability_label_column" in dataset_cfg:
                dataset_load_kwargs["probability_label_column"] = dataset_cfg["probability_label_column"]
            if "positive_label" in dataset_cfg:
                dataset_load_kwargs["positive_label"] = dataset_cfg["positive_label"]

            dataset = Dataset.load(
                dataset_path,
                dataset_cfg.get("text_column", "input"),
                dataset_cfg.get("label_column", "output"),
                **dataset_load_kwargs,
            )

            if is_pubmedqa:
                # Opt-in routing mode: one model gets correct context, one model gets wrong context,
                # remaining models get the without-context prompt variant. This should be a no-op
                # unless explicitly enabled in YAML.
                dataset.pubmedqa_wrong_context_routing = bool(
                    dataset_cfg.get("pubmedqa_wrong_context_routing", False)
                )
                # Optional extension: instead of selecting only one "wrong-context" model per example,
                # assign wrong context to *all* non-selected models.
                dataset.pubmedqa_wrong_context_all_others = bool(
                    dataset_cfg.get("pubmedqa_wrong_context_all_others", False)
                )
                seed_candidate = dataset_cfg.get("split_seed", random_seed if random_seed is not None else 42)
                try:
                    split_seed = int(seed_candidate)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid split_seed for PubMedQA dataset '{dataset_name}': {seed_candidate}"
                    ) from e

                split_ratios = _normalize_pubmedqa_split_ratios(dataset_cfg.get("pubmedqa_split_ratios"))
                dataset = _apply_pubmedqa_balanced_split(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    target_split=dataset_target_split,
                    split_seed=split_seed,
                    split_ratios=split_ratios,
                    requested_size=requested_size,
                )
            elif is_race:
                seed_candidate = dataset_cfg.get("split_seed", random_seed if random_seed is not None else 42)
                try:
                    split_seed = int(seed_candidate)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid split_seed for RACE dataset '{dataset_name}': {seed_candidate}"
                    ) from e

                split_ratios = _normalize_race_split_ratios(dataset_cfg.get("race_split_ratios"))
                dataset = _apply_race_split(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    target_split=dataset_target_split,
                    split_seed=split_seed,
                    split_ratios=split_ratios,
                    requested_size=requested_size,
                )
            elif bool(dataset_cfg.get("balanced_binary_split")) and not is_pubmedqa and not is_race:
                seed_candidate = dataset_cfg.get(
                    "split_seed", random_seed if random_seed is not None else 42
                )
                try:
                    split_seed = int(seed_candidate)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid split_seed for balanced_binary_split dataset '{dataset_name}': {seed_candidate}"
                    ) from e

                raw_ratios = dataset_cfg.get("split_ratios")
                if raw_ratios is None:
                    raw_ratios = dataset_cfg.get("pubmedqa_split_ratios")
                split_ratios = _normalize_pubmedqa_split_ratios(raw_ratios)

                if split == "train":
                    bb_target = dataset_cfg.get(
                        "train_target_split",
                        dataset_cfg.get("pubmedqa_train_target_split", "train_val"),
                    )
                elif split == "test":
                    bb_target = dataset_cfg.get(
                        "eval_target_split",
                        dataset_cfg.get(
                            "test_target_split",
                            dataset_cfg.get("pubmedqa_eval_target_split", "test"),
                        ),
                    )
                else:
                    bb_target = dataset_cfg.get(
                        "validation_target_split",
                        dataset_cfg.get("pubmedqa_validation_target_split", split),
                    )

                pos_lbl = dataset_cfg.get("positive_label", "1")
                dataset = _apply_balanced_binary_split(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    target_split=str(bb_target),
                    split_seed=split_seed,
                    split_ratios=split_ratios,
                    requested_size=requested_size,
                    positive_label=str(pos_lbl),
                )
            elif is_shared_tripartition and shared_tripartition_target is not None:
                seed_candidate = dataset_cfg.get(
                    "split_seed", random_seed if random_seed is not None else 42
                )
                try:
                    split_seed = int(seed_candidate)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid split_seed for shared-source tripartition on '{dataset_name}': {seed_candidate}"
                    ) from e
                raw_ratios = dataset_cfg.get("source_tripartition_ratios")
                split_ratios = _normalize_shared_source_tripartition_ratios(raw_ratios)
                dataset = _apply_shared_source_tripartition(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    target_split=str(shared_tripartition_target),
                    split_seed=split_seed,
                    split_ratios=split_ratios,
                    requested_size=requested_size,
                )

            if is_shared_tripartition and shared_tripartition_target is not None:
                def _display_or_slug(c: Dict[str, Any]) -> str:
                    d = c.get("display_name")
                    if d:
                        return str(d).replace("/", "_").replace("[", "").replace("]", "")
                    p = c.get("name", "")
                    return str(p).replace("/", "_").replace("[", "").replace("]", "")

                tripartition_cache_dataset_name = _display_or_slug(dataset_cfg)
                if str(split) == "test" and tripartition_peer_dataset_configs:
                    for peer in tripartition_peer_dataset_configs:
                        h_peer, _ = _resolve_hf_identity_and_split(peer, load_split="train")
                        h_self, _ = _resolve_hf_identity_and_split(dataset_cfg, load_split="test")
                        if (
                            isinstance(peer, dict)
                            and h_peer
                            and h_self
                            and h_peer == h_self
                        ):
                            tripartition_cache_dataset_name = _display_or_slug(peer)
                            break
                elif str(split) == "test" and not tripartition_peer_dataset_configs:
                    dn = dataset_cfg.get("display_name")
                    if isinstance(dn, str) and dn.endswith("_test"):
                        tripartition_cache_dataset_name = str(dn[:-5]).replace(
                            "/", "_"
                        ).replace("[", "").replace("]", "")
                    else:
                        tripartition_cache_dataset_name = _display_or_slug(
                            {**dataset_cfg, "display_name": None}
                        )
                dataset.cache_dataset_config = _build_tripartition_full_source_cache_config(
                    dataset_cfg,
                    cache_dataset_name=tripartition_cache_dataset_name,
                    resolved_path=dataset_path,
                    resolved_huggingface_split=str(actual_split),
                    resolved_config_name=config_name,
                    random_seed=random_seed,
                )
            else:
                dataset.cache_dataset_config = _build_dataset_cache_config_signature(
                    dataset_cfg=dataset_cfg,
                    dataset_name=dataset_name,
                    load_split=split,
                    resolved_path=dataset_path,
                    resolved_split=actual_split,
                    resolved_config_name=config_name,
                    dataset_target_split=dataset_target_split,
                    random_seed=random_seed,
                )
            dataset.cache_dataset_name = dataset_name
            dataset.cache_dataset_split = actual_split
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset {i} (name: {dataset_path}, split: {actual_split}): {e}"
            ) from e
        
        if len(dataset.x) == 0:
            raise ValueError(
                f"Dataset {dataset_name} loaded but has 0 examples "
                f"(split: {actual_split})"
            )
        
        datasets.append(dataset)
        dataset_names.append(dataset_name)
        if is_pubmedqa:
            log.info(
                f"Loaded dataset {i+1}/{len(dataset_configs)}: {dataset_name} "
                f"(source_split: {actual_split}, target_split: {dataset_target_split}, size: {len(dataset.x)})"
            )
        elif is_race:
            log.info(
                f"Loaded dataset {i+1}/{len(dataset_configs)}: {dataset_name} "
                f"(source_split: {actual_split}, target_split: {dataset_target_split}, size: {len(dataset.x)})"
            )
        elif is_shared_tripartition and shared_tripartition_target is not None:
            log.info(
                f"Loaded dataset {i+1}/{len(dataset_configs)}: {dataset_name} "
                f"(shared 8:1:1 of split {actual_split}, size: {len(dataset.x)})"
            )
        else:
            log.info(
                f"Loaded dataset {i+1}/{len(dataset_configs)}: {dataset_name} "
                f"(split: {actual_split}, size: {len(dataset.x)})"
            )
    
    return datasets, dataset_names
