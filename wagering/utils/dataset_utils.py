"""
Dataset loading utilities.

Simplified version with strict error handling.
"""

import logging
import sys
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Sequence, Union

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
    }
)

# Strips from cache signatures for shared-source 6:2:2 view so on-disk cache matches
# a plain load of the same HF split (no tripartition metadata in the key).
_PARTITION_EXCLUDED_CACHE_KEYS = frozenset({"partition"})

_PARTITION_TARGET_ALIASES = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
    "train_val": "train_val",
    "train+val": "train_val",
}


def dataset_for_checkpoint_hash(dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow copy of a dataset dict with training-only keys removed for stable directory hashes."""
    if not isinstance(dataset_config, dict):
        return dataset_config
    return {k: v for k, v in dataset_config.items() if k not in _DATASET_CONFIG_EPHEMERAL_KEYS}


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
        "split_seed": _partition_seed_from_cfg(dataset_cfg),
        "dataset_config": dict(dataset_for_sig),
    }
    if _partition_seed_from_cfg(dataset_cfg) is None:
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


def _partition_peer_matched(
    dataset_cfg: Dict[str, Any],
    *,
    load_split: str,
    partition_peer_dataset_configs: Optional[Sequence[Dict[str, Any]]],
    infer_eval_split_train_without_peer: bool,
) -> bool:
    """
    True when a test/eval load should use the same partitioned source as its train peer.

    Either an explicit training peer matches (same resolved HF identity and aligned split strings),
    or (eval-only) ``infer_eval_split_train_without_peer`` is True, ``load_split`` is
    ``test``, and ``eval_split`` is ``train`` (common when the official test split has
    no labels and eval reuses HF ``train``).
    """
    h1, s1 = _resolve_hf_identity_and_split(dataset_cfg, load_split=str(load_split))
    if not h1:
        return False
    if partition_peer_dataset_configs:
        peer_split = "test" if str(load_split) == "train" else "train"
        for peer in partition_peer_dataset_configs:
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
        if k in _DATASET_CONFIG_EPHEMERAL_KEYS or k in _PARTITION_EXCLUDED_CACHE_KEYS:
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
        "split_seed": _partition_seed_from_cfg(dataset_cfg),
        "dataset_config": for_sig,
    }
    if _partition_seed_from_cfg(dataset_cfg) is None:
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


def calibration_dataset_config_is_pubmedqa(dataset_config: Dict[str, Any]) -> bool:
    """True if the calibration dataset uses mixed-context routing (PubMedQA)."""
    return _is_pubmedqa_dataset_config(dataset_config)


def _normalize_partition_ratios(raw_ratios: Any) -> Tuple[float, float, float]:
    """Normalize partition ratios to a valid (train, val, test) tuple."""
    default_ratios = (0.8, 0.1, 0.1)
    if raw_ratios is None:
        return default_ratios

    if not isinstance(raw_ratios, Sequence) or len(raw_ratios) != 3:
        raise ValueError(
            f"partition.ratios must be a sequence of three numbers (got {raw_ratios!r})."
        )

    ratio_array = np.array([float(v) for v in raw_ratios], dtype=np.float64)
    if np.any(ratio_array < 0) or not np.any(ratio_array > 0):
        raise ValueError(f"partition.ratios must be positive (got {raw_ratios!r}).")

    ratio_array = ratio_array / ratio_array.sum()
    return tuple(float(v) for v in ratio_array.tolist())


def _normalize_partition_target(target_split: str) -> str:
    normalized_target = _PARTITION_TARGET_ALIASES.get(str(target_split).strip().lower())
    if normalized_target is None:
        raise ValueError(
            f"Unsupported partition target '{target_split}'. "
            "Use one of: train, validation, test, train_val."
        )
    return normalized_target


def _compute_split_counts(
    split_ratios: Tuple[float, float, float],
    pool_size: int,
) -> Tuple[int, int, int]:
    ratio_array = np.array(split_ratios, dtype=np.float64)
    raw_counts = ratio_array * float(pool_size)
    split_counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(pool_size - split_counts.sum())
    if remainder > 0:
        residual_order = np.argsort(-(raw_counts - split_counts))
        for idx in residual_order[:remainder]:
            split_counts[idx] += 1
    return tuple(int(v) for v in split_counts.tolist())


def _partition_class_pools(
    class_pools: Sequence[np.ndarray],
    split_ratios: Tuple[float, float, float],
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
    min_class_count = int(min(pool.shape[0] for pool in class_pools))
    rng = np.random.RandomState(int(split_seed))

    trimmed_pools = []
    for pool in class_pools:
        trimmed = np.array(pool, copy=True)
        rng.shuffle(trimmed)
        trimmed_pools.append(trimmed[:min_class_count])

    train_count, val_count, test_count = _compute_split_counts(split_ratios, min_class_count)

    train_parts = []
    val_parts = []
    test_parts = []
    for pool in trimmed_pools:
        train_parts.append(pool[:train_count])
        val_parts.append(pool[train_count : train_count + val_count])
        test_parts.append(pool[train_count + val_count :])

    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    test_indices = np.concatenate(test_parts)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return train_indices, val_indices, test_indices, (train_count, val_count, test_count)


def _resolve_partition_cfg(
    dataset_cfg: Dict[str, Any],
    random_seed: Optional[int],
) -> Dict[str, Any]:
    partition = dataset_cfg.get("partition")
    if not isinstance(partition, dict):
        raise ValueError(f"dataset config partition must be a mapping (got {partition!r}).")

    mode = str(partition.get("mode", "")).strip().lower()
    if mode not in {"uniform", "balanced_binary", "pubmedqa"}:
        raise ValueError(
            f"partition.mode must be one of uniform, balanced_binary, pubmedqa (got {mode!r})."
        )

    seed_candidate = partition.get("seed", random_seed if random_seed is not None else 42)
    return {
        "mode": mode,
        "ratios": _normalize_partition_ratios(partition.get("ratios")),
        "seed": int(seed_candidate),
        "train_target": partition.get("train_target", "train_val"),
        "eval_target": partition.get("eval_target", "test"),
        "validation_target": partition.get("validation_target"),
        "positive_label": str(partition.get("positive_label", dataset_cfg.get("positive_label", "1"))),
    }


def _should_apply_partition(
    *,
    load_split: str,
    partition_peer_dataset_configs: Optional[Sequence[Dict[str, Any]]],
    infer_eval_split_train_without_peer: bool,
    force_partition: bool,
    dataset_cfg: Dict[str, Any],
) -> bool:
    if str(load_split) == "train":
        return True
    partition = dataset_cfg.get("partition")
    if isinstance(partition, dict):
        mode = str(partition.get("mode", "")).strip().lower()
        if mode in {"balanced_binary", "pubmedqa"}:
            return True
    return bool(
        force_partition
        or _partition_peer_matched(
            dataset_cfg,
            load_split=load_split,
            partition_peer_dataset_configs=partition_peer_dataset_configs,
            infer_eval_split_train_without_peer=infer_eval_split_train_without_peer,
        )
    )


def _resolve_partition_target(partition_cfg: Dict[str, Any], load_split: str) -> str:
    if str(load_split) == "train":
        return str(partition_cfg["train_target"])
    if str(load_split) == "test":
        return str(partition_cfg["eval_target"])
    validation_target = partition_cfg.get("validation_target")
    if validation_target is None:
        return str(load_split)
    return str(validation_target)


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


def _apply_partition(
    dataset: Dataset,
    dataset_name: str,
    partition_cfg: Dict[str, Any],
    target_split: str,
    requested_size: Optional[int],
) -> Dataset:
    """Partition a loaded dataset into train/val/test views."""
    normalized_target = _normalize_partition_target(target_split)
    mode = partition_cfg["mode"]
    split_seed = int(partition_cfg["seed"])
    split_ratios = partition_cfg["ratios"]
    rng = np.random.RandomState(split_seed)

    if mode == "uniform":
        total_examples = len(dataset.x)
        if total_examples <= 0:
            raise ValueError(f"Dataset '{dataset_name}' is empty before partition")

        all_indices = np.arange(total_examples, dtype=np.int64)
        rng.shuffle(all_indices)
        train_count, val_count, test_count = _compute_split_counts(split_ratios, total_examples)

        train_indices = np.array(all_indices[:train_count], copy=True)
        val_indices = np.array(all_indices[train_count : train_count + val_count], copy=True)
        test_indices = np.array(all_indices[train_count + val_count :], copy=True)
        source_size = int(total_examples)
        partition_counts = {
            "source_examples": source_size,
            "train_examples": train_count,
            "validation_examples": val_count,
            "test_examples": test_count,
        }
    else:
        if mode == "pubmedqa":
            labels = np.array([str(label).strip().upper() for label in dataset.y], dtype=object)
            class_pools = [np.where(labels == "YES")[0], np.where(labels == "NO")[0]]
            class_names = ("YES", "NO")
        else:
            pos_marker = str(partition_cfg["positive_label"]).strip()
            labels = np.array([str(label).strip() for label in dataset.y], dtype=object)
            uniq = np.unique(labels)
            if uniq.size != 2:
                raise ValueError(
                    f"Balanced binary dataset '{dataset_name}' must contain exactly two label values "
                    f"(found {uniq.tolist()})."
                )
            if pos_marker not in set(uniq.tolist()):
                raise ValueError(
                    f"Balanced binary dataset '{dataset_name}': positive_label={pos_marker!r} "
                    f"not found among labels {uniq.tolist()}."
                )
            class_pools = [np.where(labels == pos_marker)[0], np.where(labels != pos_marker)[0]]
            class_names = (pos_marker, "other")

        if any(pool.size == 0 for pool in class_pools):
            raise ValueError(
                f"Partitioned dataset '{dataset_name}' must contain both classes "
                f"(mode={mode}, counts={[int(pool.size) for pool in class_pools]})."
            )

        train_indices, val_indices, test_indices, (train_count, val_count, test_count) = (
            _partition_class_pools(class_pools, split_ratios, split_seed)
        )
        source_size = int(sum(pool.size for pool in class_pools))
        partition_counts = {
            f"source_{class_names[0].lower()}": int(class_pools[0].size),
            f"source_{class_names[1].lower()}": int(class_pools[1].size),
            "balanced_per_label": int(min(pool.size for pool in class_pools)),
            "train_per_label": train_count,
            "validation_per_label": val_count,
            "test_per_label": test_count,
        }

    split_indices_map = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
        "train_val": np.concatenate([train_indices, val_indices]),
    }
    selected_indices = np.array(split_indices_map[normalized_target], copy=True).astype(
        np.int64, copy=True
    )

    preserve_contiguous = mode == "uniform" and normalized_target == "train_val"
    if not preserve_contiguous:
        rng.shuffle(selected_indices)

    if requested_size is not None:
        requested_size_int = int(requested_size)
        if requested_size_int <= 0:
            raise ValueError(
                f"Invalid size={requested_size_int} for partitioned dataset '{dataset_name}'."
            )
        if requested_size_int < selected_indices.shape[0]:
            if preserve_contiguous:
                raise ValueError(
                    f"Requested size={requested_size_int} for '{dataset_name}' train_val "
                    "conflicts with preserving train|val contiguity."
                )
            if mode in {"pubmedqa", "balanced_binary"}:
                selected_labels = labels[selected_indices]
                if mode == "pubmedqa":
                    pos_marker = "YES"
                else:
                    pos_marker = str(partition_cfg["positive_label"]).strip()
                split_pos = selected_indices[selected_labels == pos_marker]
                split_neg = selected_indices[selected_labels != pos_marker]
                per_class_limit = min(
                    split_pos.shape[0],
                    split_neg.shape[0],
                    requested_size_int // 2,
                )
                if per_class_limit <= 0:
                    raise ValueError(
                        f"Requested size={requested_size_int} is too small to keep class balance "
                        f"for partitioned dataset '{dataset_name}'."
                    )
                rng.shuffle(split_pos)
                rng.shuffle(split_neg)
                selected_indices = np.concatenate(
                    [split_pos[:per_class_limit], split_neg[:per_class_limit]]
                )
                rng.shuffle(selected_indices)
            else:
                selected_indices = selected_indices[:requested_size_int].astype(np.int64, copy=True)

    dataset = _subset_pubmedqa_dataset(dataset, selected_indices)

    dataset.partition_mode = mode
    dataset.partition_target = normalized_target
    dataset.partition_seed = split_seed
    dataset.partition_ratios = tuple(float(v) for v in split_ratios)
    dataset.partition_counts = {**partition_counts, "selected_examples": int(len(dataset.x))}

    if normalized_target == "train_val":
        train_val_size = int(train_indices.shape[0] + val_indices.shape[0])
        if train_val_size > 0:
            dataset.partition_val_ratio = float(val_indices.shape[0] / train_val_size)
        if preserve_contiguous:
            dataset.partition_contiguous_train_val = True
            dataset.partition_train_val_boundary = int(train_indices.shape[0])
        else:
            dataset.partition_contiguous_train_val = False
            dataset.partition_train_val_boundary = 0
    else:
        dataset.partition_val_ratio = None
        dataset.partition_contiguous_train_val = False
        dataset.partition_train_val_boundary = 0

    if mode == "uniform":
        dataset.cache_source_num_examples = source_size
        dataset.cache_source_row_indices = selected_indices

    log.info(
        "Partitioned %s: mode=%s, target=%s, seed=%d, counts(train/val/test)=(%d/%d/%d), view=%d",
        dataset_name,
        mode,
        normalized_target,
        split_seed,
        train_count,
        val_count,
        test_count,
        int(len(dataset.x)),
    )

    return dataset


def apply_shuffling(
    combined_dataset: Dataset,
    labels: np.ndarray,
    example_local_indices: np.ndarray,
    *,
    shuffle_data: bool,
    shuffle_seed: int,
    validation_split_ratio: float,
    dataset: Dataset,
    all_model_logits: Optional[np.ndarray] = None,
    all_hidden_states: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
) -> Tuple[
    Dataset,
    np.ndarray,
    np.ndarray,
    Optional[Dataset],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Union[np.ndarray, List[np.ndarray]]],
    Optional[Union[np.ndarray, List[np.ndarray]]],
]:
    """Shuffle cached arrays and create train/validation splits.

    Called after cache loading so cache keys are based on unshuffled data.
    Shuffles dataset (x, y, labels), all_model_logits, and all_hidden_states when present,
    then splits into train and validation sets.
    """
    contiguous_tri_split = bool(
        getattr(dataset, "partition_contiguous_train_val", False)
    )
    tri_boundary = int(
        getattr(dataset, "partition_train_val_boundary", 0) or 0
    )

    if not shuffle_data:
        log.debug("Shuffling disabled - using original order")
        indices = np.arange(len(combined_dataset.x))
    elif contiguous_tri_split and tri_boundary > 0:
        rng = np.random.RandomState(shuffle_seed)
        n = len(combined_dataset.x)
        idx_train = rng.permutation(tri_boundary)
        idx_val = tri_boundary + rng.permutation(max(n - tri_boundary, 0))
        indices = np.concatenate([idx_train, idx_val]) if idx_val.size else idx_train
        log.info(
            "Shuffling within shared-source train/val partitions only (boundary=%d of %d examples; seed=%d)",
            tri_boundary,
            n,
            int(shuffle_seed),
        )
    else:
        rng = np.random.RandomState(shuffle_seed)
        indices = np.arange(len(combined_dataset.x))
        rng.shuffle(indices)
        log.debug(f"Shuffled dataset with seed {shuffle_seed}")

    shuffled_x = [combined_dataset.x[i] for i in indices]
    shuffled_y = [combined_dataset.y[i] for i in indices]
    shuffled_labels = labels[indices]
    shuffled_example_local_indices = example_local_indices[indices]

    if all_model_logits is not None:
        all_model_logits = all_model_logits[:, indices, :]
        log.debug("Shuffled cached logits")

    if all_hidden_states is not None:
        if isinstance(all_hidden_states, list):
            all_hidden_states = [hs[indices, :] for hs in all_hidden_states]
        elif all_hidden_states.ndim == 3:
            all_hidden_states = all_hidden_states[:, indices, :]
        else:
            all_hidden_states = all_hidden_states[indices, :]
        log.debug("Shuffled cached hidden states")

    batch_size = combined_dataset.batch_size
    total_size = len(shuffled_x)

    log.debug(
        "Creating train/validation split: validation_split_ratio=%s, total_size=%d",
        validation_split_ratio,
        total_size,
    )

    validation_dataset = None
    validation_labels = None
    validation_example_local_indices = None
    all_model_val_logits = None
    all_val_hidden_states = None

    if validation_split_ratio > 0 and validation_split_ratio < 1:
        if contiguous_tri_split and tri_boundary > 0:
            train_size = tri_boundary
            val_size = total_size - train_size
        else:
            val_size = int(total_size * validation_split_ratio)
            train_size = total_size - val_size

        train_x = shuffled_x[:train_size]
        train_y = shuffled_y[:train_size]
        train_labels = shuffled_labels[:train_size]
        train_example_local_indices = shuffled_example_local_indices[:train_size]

        val_x = shuffled_x[train_size:]
        val_y = shuffled_y[train_size:]
        val_labels = shuffled_labels[train_size:]
        val_example_local_indices = shuffled_example_local_indices[train_size:]

        combined_dataset = Dataset(train_x, train_y, batch_size=batch_size)
        labels = np.array(train_labels, dtype=np.int32)
        example_local_indices = np.array(train_example_local_indices, dtype=np.int32)

        validation_dataset = Dataset(val_x, val_y, batch_size=batch_size)
        validation_labels = np.array(val_labels, dtype=np.int32)
        validation_example_local_indices = np.array(val_example_local_indices, dtype=np.int32)

        log.debug("Created validation_dataset with %d examples", len(validation_dataset.x))

        if all_model_logits is not None:
            all_model_val_logits = all_model_logits[:, train_size:, :]
            all_model_logits = all_model_logits[:, :train_size, :]
            log.debug(
                "Split logits: training=%s, validation=%s",
                all_model_logits.shape,
                all_model_val_logits.shape,
            )
        else:
            raise RuntimeError("No all_model_logits to split for validation")

        if all_hidden_states is not None:
            if isinstance(all_hidden_states, list):
                all_val_hidden_states = [hs[train_size:, :] for hs in all_hidden_states]
                all_hidden_states = [hs[:train_size, :] for hs in all_hidden_states]
            elif all_hidden_states.ndim == 3:
                all_val_hidden_states = all_hidden_states[:, train_size:, :]
                all_hidden_states = all_hidden_states[:, :train_size, :]
            else:
                all_val_hidden_states = all_hidden_states[train_size:, :]
                all_hidden_states = all_hidden_states[:train_size, :]

        log.debug(
            "Split dataset after shuffling: %d train, %d validation (%.1f%% validation)",
            train_size,
            val_size,
            validation_split_ratio * 100,
        )
    else:
        combined_dataset = Dataset(shuffled_x, shuffled_y, batch_size=batch_size)
        labels = shuffled_labels
        example_local_indices = shuffled_example_local_indices

        if all_model_logits is not None:
            all_model_val_logits = None

        log.debug(
            "Shuffled dataset: %d examples (no validation split)",
            len(combined_dataset.x),
        )

    return (
        combined_dataset,
        labels,
        example_local_indices,
        validation_dataset,
        validation_labels,
        validation_example_local_indices,
        all_model_logits,
        all_model_val_logits,
        all_hidden_states,
        all_val_hidden_states,
    )


def _partition_seed_from_cfg(dataset_cfg: Dict[str, Any]) -> Optional[int]:
    partition = dataset_cfg.get("partition")
    if isinstance(partition, dict) and "seed" in partition:
        return int(partition["seed"])
    if "split_seed" in dataset_cfg:
        return int(dataset_cfg["split_seed"])
    return None


def load_dataset_from_config(
    dataset_cfg: Dict[str, Any],
    split: str = "train",
    random_seed: Optional[int] = None,
    partition_peer_dataset_configs: Optional[Sequence[Dict[str, Any]]] = None,
    infer_eval_split_train_without_peer: bool = False,
    force_partition: bool = False,
) -> Tuple[Dataset, str]:
    """Load a single dataset from configuration."""
    if "name" not in dataset_cfg:
        raise ValueError(f"Dataset config missing required 'name' field: {dataset_cfg}")

    dataset_path = dataset_cfg["name"]
    is_pubmedqa = _is_pubmedqa_dataset_config(dataset_cfg)
    requested_size = dataset_cfg.get("size", None)
    partition_cfg_raw = dataset_cfg.get("partition")
    has_partition = isinstance(partition_cfg_raw, dict)
    partition_applied = False
    dataset_target_split: Optional[str] = None

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
    elif split == "train":
        config_name = dataset_cfg.get("train_config_name", dataset_cfg.get("config_name"))
        actual_split = dataset_cfg.get("train_split", split)
    else:
        config_name = dataset_cfg.get(
            "eval_config_name",
            dataset_cfg.get("test_config_name", dataset_cfg.get("config_name")),
        )
        actual_split = dataset_cfg.get("eval_split", split)

    if config_name and isinstance(dataset_path, str):
        dataset_path = [dataset_path, config_name]
    dataset_name = dataset_cfg.get(
        "display_name",
        str(dataset_path).replace("/", "_").replace("[", "").replace("]", ""),
    )

    source_size = dataset_cfg.get("source_size", None) if is_pubmedqa else requested_size

    dataset_load_kwargs = {
        "batch_size": dataset_cfg.get("batch_size", 8),
        "prompt": dataset_cfg.get("prompt", ""),
        "description": dataset_cfg.get("description", ""),
        "split": actual_split,
        "size": source_size,
        "trust_remote_code": dataset_cfg.get("trust_remote_code", False),
    }
    for optional_key in (
        "data_files",
        "field",
        "prompt_without_context",
        "prompt_helper",
        "prompt_without_context_helper",
        "mixed_context_routing",
        "probability_label_column",
        "positive_label",
    ):
        if optional_key in dataset_cfg:
            dataset_load_kwargs[optional_key] = dataset_cfg[optional_key]

    dataset = Dataset.load(
        dataset_path,
        dataset_cfg.get("text_column", "input"),
        dataset_cfg.get("label_column", "output"),
        **dataset_load_kwargs,
    )

    if is_pubmedqa:
        dataset.pubmedqa_wrong_context_routing = bool(
            dataset_cfg.get("pubmedqa_wrong_context_routing", False)
        )

    if has_partition and _should_apply_partition(
        load_split=split,
        partition_peer_dataset_configs=partition_peer_dataset_configs,
        infer_eval_split_train_without_peer=bool(infer_eval_split_train_without_peer),
        force_partition=bool(force_partition),
        dataset_cfg=dataset_cfg,
    ):
        partition_cfg = _resolve_partition_cfg(dataset_cfg, random_seed)
        dataset_target_split = _resolve_partition_target(partition_cfg, split)
        dataset = _apply_partition(
            dataset=dataset,
            dataset_name=dataset_name,
            partition_cfg=partition_cfg,
            target_split=dataset_target_split,
            requested_size=requested_size,
        )
        partition_applied = True

    use_uniform_cache = partition_applied and getattr(dataset, "partition_mode", None) == "uniform"

    if use_uniform_cache:
        def _display_or_slug(c: Dict[str, Any]) -> str:
            d = c.get("display_name")
            if d:
                return str(d).replace("/", "_").replace("[", "").replace("]", "")
            p = c.get("name", "")
            return str(p).replace("/", "_").replace("[", "").replace("]", "")

        partition_cache_dataset_name = _display_or_slug(dataset_cfg)
        if str(split) == "test" and partition_peer_dataset_configs:
            for peer in partition_peer_dataset_configs:
                h_peer, _ = _resolve_hf_identity_and_split(peer, load_split="train")
                h_self, _ = _resolve_hf_identity_and_split(dataset_cfg, load_split="test")
                if isinstance(peer, dict) and h_peer and h_self and h_peer == h_self:
                    partition_cache_dataset_name = _display_or_slug(peer)
                    break
        elif str(split) == "test" and not partition_peer_dataset_configs:
            dn = dataset_cfg.get("display_name")
            if isinstance(dn, str) and dn.endswith("_test"):
                partition_cache_dataset_name = str(dn[:-5]).replace(
                    "/", "_"
                ).replace("[", "").replace("]", "")
            else:
                partition_cache_dataset_name = _display_or_slug(
                    {**dataset_cfg, "display_name": None}
                )
        dataset.cache_dataset_config = _build_tripartition_full_source_cache_config(
            dataset_cfg,
            cache_dataset_name=partition_cache_dataset_name,
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

    if len(dataset.x) == 0:
        raise ValueError(
            f"Dataset {dataset_name} loaded but has 0 examples (split: {actual_split})"
        )

    if partition_applied:
        log.info(
            f"Loaded dataset: {dataset_name} "
            f"(partitioned from split {actual_split}, target={dataset_target_split}, size: {len(dataset.x)})"
        )
    else:
        log.info(
            f"Loaded dataset: {dataset_name} (split: {actual_split}, size: {len(dataset.x)})"
        )

    return dataset, dataset_name


def load_datasets_from_config(
    dataset_configs: List[Dict[str, Any]],
    split: str = "train",
    random_seed: Optional[int] = None,
    partition_peer_dataset_configs: Optional[Sequence[Dict[str, Any]]] = None,
    infer_eval_split_train_without_peer: bool = False,
    force_partition: bool = False,
) -> Tuple[List[Dataset], List[str]]:
    """Load multiple datasets (used for OOD eval lists)."""
    if not dataset_configs:
        raise ValueError("Must provide at least one dataset config")

    datasets: List[Dataset] = []
    dataset_names: List[str] = []
    for dataset_cfg in dataset_configs:
        dataset, name = load_dataset_from_config(
            dataset_cfg,
            split=split,
            random_seed=random_seed,
            partition_peer_dataset_configs=partition_peer_dataset_configs,
            infer_eval_split_train_without_peer=infer_eval_split_train_without_peer,
            force_partition=force_partition,
        )
        datasets.append(dataset)
        dataset_names.append(name)
    return datasets, dataset_names
