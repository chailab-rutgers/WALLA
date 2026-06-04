"""
Multi-LLM ensemble utilities for wagering package.

Contains functions for collecting logits and hidden states from models,
with disk-based caching for efficiency.

Disk cache (logits + hidden states + labels, ``wagering_model_logits_states_caches``):

- **Non-PubMedQA:** Cache key uses the Hugging Face ``model_path`` only (no slot index).
  Ensemble order does not matter; repeated copies of the same model reuse one cache file
  per (dataset signature, option tokens, prompt variant).

- **Mixed-context datasets (PubMedQA/RACE):** Keys include ``model_path::idx=<slot>`` and
    a dataset-specific namespace so routing and prompts stay aligned with the ensemble.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Import wagering-local model and dataset classes
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wagering.core.model import WhiteboxModel
from wagering.core.dataset import Dataset

log = logging.getLogger("wagering")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend by default
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    log.warning("matplotlib not available; plotting functions will be disabled")

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("sklearn not available; AUC calculation will be disabled")


@dataclass
class LogitCacheKey:
    """
    Lightweight identifier for a particular (dataset, split, model, size) logit run.
    """

    dataset_name: str
    split: str
    model_id: str
    num_examples: int | None = None

    def to_filename(self) -> str:
        safe_model = self.model_id.replace("/", "_")
        safe_dataset = self.dataset_name.replace("/", "_")
        safe_split = self.split.replace("/", "_")
        size_suffix = "" if self.num_examples is None else f"_{self.num_examples}"
        return f"{safe_dataset}{size_suffix}__{safe_split}__{safe_model}.npz"


class LogitCache:
    """
    Small utility for saving/loading per-option logits for multiple-choice QA.

    Stored format (npz):
        - logits: float32 array of shape [num_examples, num_options]
        - labels: int32 array of shape [num_examples]
        - meta:   UTF-8 encoded JSON string with any additional fields (optional)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: LogitCacheKey) -> Path:
        return self.cache_dir / key.to_filename()

    def save(
        self,
        key: LogitCacheKey,
        logits: np.ndarray,
        labels: np.ndarray,
        meta: Optional[Dict] = None,
    ) -> Path:
        path = self.path_for(key)
        meta = meta or {}
        np.savez_compressed(
            path,
            logits=logits.astype(np.float32),
            labels=labels.astype(np.int32),
            meta=np.string_(repr(meta)),
        )
        log.info(
            f"Saved logits cache for dataset={key.dataset_name}, "
            f"split={key.split}, model={key.model_id} to {path}"
        )
        return path

    def load(self, key: LogitCacheKey) -> Tuple[np.ndarray, np.ndarray]:
        path = self.path_for(key)
        if not path.exists():
            raise FileNotFoundError(f"Logit cache not found at {path}")
        data = np.load(path, allow_pickle=True)
        logits = data["logits"].astype(np.float32)
        labels = data["labels"].astype(np.int32)
        return logits, labels


# Disk-based cache directory for logits and hidden states.
#
# NOTE: This cache can be redirected at runtime by calling
# `configure_wagering_cache_dir(...)` (used by `scripts/wagering_pipeline.py`).
_WAGERING_CACHE_DIR = Path("/common/users/yl2310/MultiLLMs/wagering_model_logits_states_caches")
_WAGERING_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def configure_wagering_cache_dir(cache_path: Optional[str]) -> Path:
    """
    Configure the disk cache directory for logits/hidden-states caches.

    `cache_path` is expected to be a user-provided root cache directory (often set
    in YAML as `cache_path:`). We store wagering artifacts in a stable subfolder
    to avoid mixing with unrelated caches.
    """
    global _WAGERING_CACHE_DIR

    if cache_path is None:
        return _WAGERING_CACHE_DIR

    root = Path(str(cache_path)).expanduser()
    # Treat `cache_path` as a root; keep a stable subdir name for these artifacts.
    target = root / "wagering_model_logits_states_caches"
    target.mkdir(parents=True, exist_ok=True)
    _WAGERING_CACHE_DIR = target
    return _WAGERING_CACHE_DIR


def _get_model_path_key(model: WhiteboxModel) -> str:
    """Create a cache key from a single model path."""
    return model.model_path


def _get_dataset_signature(dataset: Dataset) -> Tuple:
    """Create a dataset signature for caching.

    Prefer explicit dataset-config signatures attached by dataset loaders so cache
    keys stay stable across run-level shuffle seeds.
    """
    dataset_cache_config = getattr(dataset, "cache_dataset_config", None)
    if isinstance(dataset_cache_config, dict):
        signature = dataset_cache_config.get("signature")
        schema_version = dataset_cache_config.get("schema_version", 0)
        if isinstance(signature, str) and signature:
            return ("cfg", int(schema_version), signature)

    return _get_legacy_dataset_signature(dataset)


def _get_legacy_dataset_signature(dataset: Dataset) -> Tuple:
    """Legacy cache signature based on content-derived heuristics.

    Older cache files were written before dataset loaders attached deterministic
    `cache_dataset_config` signatures. Keep this fallback so existing cache
    artifacts remain reusable.
    """
    dataset_size = len(dataset.x)
    # Create a hash from first 3 examples for uniqueness (coerce: CSV numeric text_column, etc.)
    sample_parts = dataset.x[: min(3, len(dataset.x))] if dataset.x else []
    sample_text = "\n".join(str(s) for s in sample_parts)
    content_hash = hashlib.md5(sample_text.encode('utf-8')).hexdigest()[:8]
    return (dataset_size, content_hash)


def _is_pubmedqa_dataset(dataset: Dataset) -> bool:
    """Return True when this dataset appears to be PubMedQA."""
    dataset_cache_config = getattr(dataset, "cache_dataset_config", None)
    if isinstance(dataset_cache_config, dict):
        payload = dataset_cache_config.get("payload")
        if isinstance(payload, dict):
            dataset_cfg = payload.get("dataset_config")
            if isinstance(dataset_cfg, dict):
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
                if "pubmedqa" in normalized or "pubmed_qa" in normalized:
                    return True

    source_name = str(getattr(dataset, "cache_dataset_name", "")).lower()
    return "pubmedqa" in source_name or "pubmed_qa" in source_name


def _is_race_dataset(dataset: Dataset) -> bool:
    """Return True when this dataset appears to be RACE."""
    dataset_cache_config = getattr(dataset, "cache_dataset_config", None)
    if isinstance(dataset_cache_config, dict):
        payload = dataset_cache_config.get("payload")
        if isinstance(payload, dict):
            dataset_cfg = payload.get("dataset_config")
            if isinstance(dataset_cfg, dict):
                fields = [
                    dataset_cfg.get("name", ""),
                    dataset_cfg.get("display_name", ""),
                    dataset_cfg.get("config_name", ""),
                    dataset_cfg.get("train_config_name", ""),
                    dataset_cfg.get("eval_config_name", ""),
                    dataset_cfg.get("test_config_name", ""),
                ]
                normalized = " ".join(str(field).lower() for field in fields if field is not None)
                if "eleutherai/race" in normalized or " race" in f" {normalized}" or "race " in f"{normalized} ":
                    return True

    source_name = str(getattr(dataset, "cache_dataset_name", "")).lower()
    return "eleutherai/race" in source_name or source_name == "race" or source_name.endswith("_race")


def _get_mixed_context_dataset_type(dataset: Dataset) -> Optional[str]:
    """Return mixed-context dataset type name when model-specific routing is enabled."""
    if getattr(dataset, "pubmedqa_prompt_strategy", None) == "mixed_context" or _is_pubmedqa_dataset(dataset):
        return "pubmedqa"
    if getattr(dataset, "race_prompt_strategy", None) == "article_context_mixed" or _is_race_dataset(dataset):
        return "race"
    return None


def _requires_slot_specific_cache(dataset: Dataset) -> bool:
    """Whether cache keys must include model slot index for this dataset."""
    return _get_mixed_context_dataset_type(dataset) in {"pubmedqa", "race"}


# Bump when PubMedQA logits/hidden-state cache semantics change (split policy, labels,
# mixed-context routing). Old hashed .npz files are then ignored without manual cleanup.
PUBMEDQA_LOGITS_CACHE_NAMESPACE = "pubmedqa_v2_stable_dataset_split_seed"
RACE_LOGITS_CACHE_NAMESPACE = "race_v1_first_question_article_context_routing"


def _wagering_logits_cache_key(
    model_key: str,
    dataset: Dataset,
    option_tokens: Sequence[str],
    prompt_variant: Optional[str],
    hidden_state_layers: Optional[Sequence[int]] = None,
) -> Tuple[Any, ...]:
    """Disk cache key for option logits + hidden states.

    Non-PubMedQA: ``model_key`` is the model path only — no per-slot suffix, so order/repeats
    share storage. PubMedQA: ``model_key`` includes ``::idx=`` and an extra namespace tuple field.
    """
    dataset_key = _get_dataset_signature(dataset)
    option_key = tuple(option_tokens)
    pv = prompt_variant or "default"

    mixed_context_type = _get_mixed_context_dataset_type(dataset)
    if mixed_context_type == "pubmedqa":
        base_key: Tuple[Any, ...] = (model_key, dataset_key, option_key, pv, PUBMEDQA_LOGITS_CACHE_NAMESPACE)
    elif mixed_context_type == "race":
        base_key = (model_key, dataset_key, option_key, pv, RACE_LOGITS_CACHE_NAMESPACE)
    else:
        base_key = (model_key, dataset_key, option_key, pv)

    # Cache key intentionally ignores hidden_state_layers. Layer compatibility
    # is validated at load-time using metadata saved with the cache artifact so
    # subset requests (e.g. cache built with [0,-2,-1], request [0,-1]) can
    # reuse existing cache files without recollection.
    _ = hidden_state_layers
    return base_key


def _normalize_hidden_state_layers(hidden_state_layers: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    """Normalize configured hidden-state layers to a stable tuple used in cache keys."""
    if hidden_state_layers is None:
        return None
    if not isinstance(hidden_state_layers, (list, tuple)):
        raise ValueError(
            "hidden_state_layers must be a list/tuple of integers, "
            f"got {type(hidden_state_layers).__name__}"
        )
    if len(hidden_state_layers) == 0:
        raise ValueError("hidden_state_layers cannot be empty")

    normalized: List[int] = []
    seen: set[int] = set()
    for value in hidden_state_layers:
        layer_idx = int(value)
        if layer_idx in seen:
            continue
        seen.add(layer_idx)
        normalized.append(layer_idx)
    return tuple(normalized)


def resolve_hidden_state_layers_for_model(
    hidden_state_layers: Optional[Sequence[int]],
    hidden_state_layers_per_model: Optional[Any],
    model_index: int,
    num_models: Optional[int] = None,
) -> Optional[List[int]]:
    """Resolve hidden-state layer selection for a specific model index."""
    selected: Any
    if hidden_state_layers_per_model is None:
        selected = hidden_state_layers
    elif isinstance(hidden_state_layers_per_model, dict):
        if model_index in hidden_state_layers_per_model:
            selected = hidden_state_layers_per_model[model_index]
        elif str(model_index) in hidden_state_layers_per_model:
            selected = hidden_state_layers_per_model[str(model_index)]
        else:
            raise ValueError(
                "hidden_state_layers_per_model is missing an entry for model index "
                f"{model_index}"
            )
    elif isinstance(hidden_state_layers_per_model, (list, tuple)):
        if num_models is not None and len(hidden_state_layers_per_model) != int(num_models):
            raise ValueError(
                "hidden_state_layers_per_model length must match number of models: "
                f"expected {int(num_models)}, got {len(hidden_state_layers_per_model)}"
            )
        if model_index < 0 or model_index >= len(hidden_state_layers_per_model):
            raise ValueError(
                "hidden_state_layers_per_model index out of range for model index "
                f"{model_index}"
            )
        selected = hidden_state_layers_per_model[model_index]
    else:
        raise ValueError(
            "hidden_state_layers_per_model must be a dict or list/tuple, "
            f"got {type(hidden_state_layers_per_model).__name__}"
        )

    if selected is None:
        return None
    if isinstance(selected, (int, np.integer)):
        return [int(selected)]
    if isinstance(selected, (list, tuple)):
        if len(selected) == 0:
            raise ValueError("Per-model hidden_state_layers entry cannot be empty")
        return [int(value) for value in selected]

    raise ValueError(
        "Per-model hidden_state_layers entry must be an int or list/tuple of ints, "
        f"got {type(selected).__name__} for model index {model_index}"
    )


def _resolve_transformer_layer_indices(
    requested_layers: Optional[Sequence[int]],
    num_transformer_layers: int,
) -> Tuple[int, ...]:
    """Resolve configured layer indices against a transformer-layer count."""
    if num_transformer_layers <= 0:
        raise ValueError("No transformer layers available to select from")

    normalized_request = _normalize_hidden_state_layers(requested_layers)
    if normalized_request is None:
        return (num_transformer_layers - 1,)

    resolved_indices: List[int] = []
    for layer in normalized_request:
        if layer >= 0:
            if layer >= num_transformer_layers:
                raise ValueError(
                    f"Requested hidden_state_layers entry {layer} is out of range for "
                    f"{num_transformer_layers} transformer layers"
                )
            resolved_indices.append(int(layer))
        else:
            if -layer > num_transformer_layers:
                raise ValueError(
                    f"Requested hidden_state_layers entry {layer} is out of range for "
                    f"{num_transformer_layers} transformer layers"
                )
            resolved_indices.append(int(num_transformer_layers + layer))

    return tuple(resolved_indices)


def extract_hidden_state_features(
    hidden_states: Optional[np.ndarray],
    hidden_state_layers: Optional[Sequence[int]] = None,
    cached_requested_hidden_state_layers: Optional[Sequence[int]] = None,
) -> Optional[np.ndarray]:
    """Extract selected layer features from cached hidden states.

    Supported cache formats:
      - all layers: [num_examples, num_layers, hidden_dim]
      - legacy selected features: [num_examples, hidden_dim]
    """
    if hidden_states is None:
        return None

    hidden_states_array = np.asarray(hidden_states)
    if hidden_states_array.ndim == 3:
        cached_requested = _normalize_hidden_state_layers(cached_requested_hidden_state_layers)
        if cached_requested is not None and len(cached_requested) == hidden_states_array.shape[1]:
            requested = _normalize_hidden_state_layers(hidden_state_layers)
            if requested is None:
                # Preserve prior default behavior: use the last available cached layer.
                selected = hidden_states_array[:, -1:, :]
            else:
                layer_to_cached_pos = {layer: pos for pos, layer in enumerate(cached_requested)}
                missing = [layer for layer in requested if layer not in layer_to_cached_pos]
                if missing:
                    raise RuntimeError(
                        "Cached hidden states do not include requested layers. "
                        f"requested={list(requested)}, cached={list(cached_requested)}"
                    )
                selected_positions = [layer_to_cached_pos[layer] for layer in requested]
                selected = hidden_states_array[:, selected_positions, :]

            if selected.shape[1] == 1:
                return selected[:, 0, :].astype(np.float32, copy=False)
            batch_size = selected.shape[0]
            return selected.reshape(batch_size, -1).astype(np.float32, copy=False)

        selected_layers = _resolve_transformer_layer_indices(
            hidden_state_layers,
            hidden_states_array.shape[1],
        )
        selected = hidden_states_array[:, selected_layers, :]
        if selected.shape[1] == 1:
            return selected[:, 0, :].astype(np.float32, copy=False)
        batch_size = selected.shape[0]
        return selected.reshape(batch_size, -1).astype(np.float32, copy=False)

    if hidden_states_array.ndim == 2:
        raise RuntimeError(
            "Detected legacy hidden-state cache format [num_examples, hidden_dim]. "
            "Please delete old cache files so hidden states can be recollected with all layers."
        )

    raise ValueError(f"Unexpected cached hidden_states shape: {hidden_states_array.shape}")


def _extract_hidden_state_features_from_layer_map(
    hidden_states_by_layer: Dict[int, np.ndarray],
    hidden_state_layers: Optional[Sequence[int]],
) -> Optional[np.ndarray]:
    """
    Extract selected hidden-state features from a per-layer mapping.

    Mapping keys are configured layer ids (e.g. -1, -12, 0) and values are arrays
    of shape [num_examples, hidden_dim]. Multiple requested layers are concatenated
    along the feature dimension.
    """
    if not hidden_states_by_layer:
        return None

    requested = _normalize_hidden_state_layers(hidden_state_layers)
    if requested is None:
        # Mirror the "default last layer" behavior. Prefer -1 when present.
        if -1 in hidden_states_by_layer:
            return np.asarray(hidden_states_by_layer[-1], dtype=np.float32)
        best_key = sorted(hidden_states_by_layer.keys())[-1]
        return np.asarray(hidden_states_by_layer[best_key], dtype=np.float32)

    missing = [layer for layer in requested if int(layer) not in hidden_states_by_layer]
    if missing:
        raise RuntimeError(
            "Cached hidden states do not include requested layers. "
            f"requested={list(requested)}, cached={sorted(hidden_states_by_layer.keys())}"
        )

    parts = [np.asarray(hidden_states_by_layer[int(layer)], dtype=np.float32) for layer in requested]
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)


def _cache_key_to_filename(cache_key: Tuple) -> str:
    """Convert a cache key to a filename-safe string using MD5 hash."""
    key_str = json.dumps(cache_key, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
    return f"{key_hash}.npz"


def _get_cache_path(cache_key: Tuple) -> Path:
    """Get the file path for a cache key."""
    filename = _cache_key_to_filename(cache_key)
    return _WAGERING_CACHE_DIR / filename


def _build_pubmedqa_balanced_assignments(
    num_examples: int,
    num_models: int,
    seed: int,
) -> np.ndarray:
    """Create balanced per-example model assignments with randomized ordering."""
    if num_models <= 0:
        raise ValueError(f"num_models must be positive, got {num_models}")
    if num_examples <= 0:
        return np.empty((0,), dtype=np.int32)

    rng = np.random.RandomState(seed)
    base_count = num_examples // num_models
    remainder = num_examples % num_models

    assignments = np.repeat(np.arange(num_models, dtype=np.int32), base_count)
    if remainder > 0:
        extra_models = rng.permutation(np.arange(num_models, dtype=np.int32))[:remainder]
        assignments = np.concatenate([assignments, extra_models.astype(np.int32)])

    rng.shuffle(assignments)
    return assignments.astype(np.int32, copy=False)


def _build_pubmedqa_wrong_context_assignments(
    *,
    num_examples: int,
    num_models: int,
    seed: int,
    right_context_assignments: np.ndarray,
) -> np.ndarray:
    """Choose a distinct 'wrong-context model' index per example.

    - For num_models==2: wrong assignment is always the other model.
    - For num_models>2: uniformly sample from the remaining models.
    """
    if num_models < 2:
        raise ValueError("pubmedqa_wrong_context_routing requires at least 2 models")
    if num_examples <= 0:
        return np.empty((0,), dtype=np.int32)

    assignments = np.asarray(right_context_assignments, dtype=np.int32)
    if assignments.shape != (num_examples,):
        raise ValueError("right_context_assignments must be 1D and match num_examples")
    if np.any(assignments < 0) or np.any(assignments >= num_models):
        raise ValueError("right_context_assignments contains out-of-range model indices")

    if num_models == 2:
        return (1 - assignments).astype(np.int32, copy=False)

    rng = np.random.RandomState(int(seed))
    wrong = np.empty((num_examples,), dtype=np.int32)
    for i in range(num_examples):
        right_idx = int(assignments[i])
        # Sample from [0..num_models-2] then shift to skip right_idx.
        r = int(rng.randint(0, num_models - 1))
        wrong[i] = r if r < right_idx else r + 1
    return wrong.astype(np.int32, copy=False)


def _build_pubmedqa_wrong_context_example_indices(
    *,
    num_examples: int,
    seed: int,
) -> np.ndarray:
    """Choose an alternate example index per example to source wrong context from."""
    if num_examples <= 0:
        return np.empty((0,), dtype=np.int32)
    if num_examples == 1:
        # Degenerate: no other example exists; fall back to self.
        return np.zeros((1,), dtype=np.int32)

    rng = np.random.RandomState(int(seed))
    out = np.empty((num_examples,), dtype=np.int32)
    for i in range(num_examples):
        # Sample from [0..num_examples-2], then shift to skip i.
        r = int(rng.randint(0, num_examples - 1))
        out[i] = r if r < i else r + 1
    return out.astype(np.int32, copy=False)


def _build_pubmedqa_wrong_context_example_indices_by_model(
    *,
    num_examples: int,
    num_models: int,
    seed: int,
) -> np.ndarray:
    """Choose wrong-context source rows per example *and* per model.

    For each example i, we pick a random permutation of candidate source rows
    (all rows except i). Each model m is assigned a (cyclic) choice from that
    permutation so different models tend to receive different wrong contexts.

    Returns shape [num_examples, num_models].
    """
    if num_models <= 0:
        raise ValueError(f"num_models must be positive, got {num_models}")
    if num_examples <= 0:
        return np.empty((0, num_models), dtype=np.int32)
    if num_examples == 1:
        return np.zeros((1, num_models), dtype=np.int32)

    rng = np.random.RandomState(int(seed))
    out = np.empty((num_examples, num_models), dtype=np.int32)
    for i in range(num_examples):
        candidates = np.arange(num_examples, dtype=np.int32)
        candidates = candidates[candidates != i]
        perm = rng.permutation(candidates)
        # Randomly rotate the permutation per example so model indices don't always map
        # to the same candidate position across examples.
        shift = int(rng.randint(0, len(perm)))
        for m in range(num_models):
            out[i, m] = int(perm[(m + shift) % len(perm)])
    return out.astype(np.int32, copy=False)


def assign_pubmedqa_context_model(
    dataset: Dataset,
    model_paths: Sequence[str],
    random_seed: Optional[int] = None,
    dataset_index: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    """
    Assign mixed-context prompts (PubMedQA/RACE) per-example via balanced randomized routing.

    Returns assignment metadata if the dataset uses mixed PubMedQA prompts,
    otherwise returns None.
    """
    dataset_type = _get_mixed_context_dataset_type(dataset)
    if dataset_type is None:
        return None

    paths = [str(path) for path in model_paths if path]
    if not paths:
        return None

    num_examples = len(dataset.x)
    num_models = len(paths)

    normalized_seed: Optional[int] = None if random_seed is None else int(random_seed)
    # Persist dataset_index so any later deterministic "rebuild" logic can exactly
    # mirror the seed construction used during assignment.
    if dataset_index is not None:
        try:
            dataset.pubmedqa_context_dataset_index = int(dataset_index)
        except Exception:
            pass

    assignment_attr = f"{dataset_type}_context_assignment_by_example"
    counts_attr = f"{dataset_type}_context_assignment_counts"
    hash_attr = f"{dataset_type}_context_assignment_hash"
    run_seed_attr = f"{dataset_type}_context_run_seed"

    assignments: Optional[np.ndarray] = None
    existing = getattr(dataset, assignment_attr, None)
    if isinstance(existing, list) and len(existing) == num_examples:
        try:
            existing_array = np.asarray(existing, dtype=np.int32)
            if np.all((existing_array >= 0) & (existing_array < num_models)):
                assignments = existing_array
        except Exception:
            assignments = None

    if assignments is None:
        dataset_signature = _get_dataset_signature(dataset)
        seed_components = [
            f"{dataset_type}_balanced_context",
            str(dataset_signature),
            "||".join(paths),
        ]
        if dataset_index is not None:
            seed_components.append(f"dataset_index={int(dataset_index)}")
        seed_input = "::".join(seed_components)
        seed = int(hashlib.md5(seed_input.encode("utf-8")).hexdigest()[:8], 16)
        assignments = _build_pubmedqa_balanced_assignments(
            num_examples=num_examples,
            num_models=num_models,
            seed=seed,
        )

    assignment_hash = hashlib.md5(assignments.tobytes()).hexdigest()[:12]
    context_counts = np.bincount(assignments, minlength=num_models).astype(np.int32).tolist()

    setattr(dataset, assignment_attr, assignments.tolist())
    setattr(dataset, counts_attr, context_counts)
    setattr(dataset, hash_attr, assignment_hash)
    setattr(dataset, run_seed_attr, normalized_seed)

    # Optional: assign an additional model per example to receive "wrong context" prompts.
    wrong_context_enabled = bool(
        dataset_type == "pubmedqa" and bool(getattr(dataset, "pubmedqa_wrong_context_routing", False))
    )
    wrong_context_all_others = bool(
        dataset_type == "pubmedqa" and bool(getattr(dataset, "pubmedqa_wrong_context_all_others", False))
    )
    wrong_assignment_hash = None
    wrong_counts: Optional[List[int]] = None
    if wrong_context_enabled:
        if not wrong_context_all_others:
            wrong_seed_components = [
                f"{dataset_type}_wrong_context_model",
                str(_get_dataset_signature(dataset)),
                "||".join(paths),
            ]
            if dataset_index is not None:
                wrong_seed_components.append(f"dataset_index={int(dataset_index)}")
            wrong_seed_input = "::".join(wrong_seed_components)
            wrong_seed = int(hashlib.md5(wrong_seed_input.encode("utf-8")).hexdigest()[:8], 16)
            wrong_assignments = _build_pubmedqa_wrong_context_assignments(
                num_examples=num_examples,
                num_models=num_models,
                seed=wrong_seed,
                right_context_assignments=assignments,
            )
            wrong_assignment_hash = hashlib.md5(wrong_assignments.tobytes()).hexdigest()[:12]
            wrong_counts = np.bincount(wrong_assignments, minlength=num_models).astype(np.int32).tolist()

            dataset.pubmedqa_wrong_context_assignment_by_example = wrong_assignments.tolist()
            dataset.pubmedqa_wrong_context_assignment_counts = wrong_counts
            dataset.pubmedqa_wrong_context_assignment_hash = wrong_assignment_hash
        else:
            # Still produce a stable hash for cache-keying even though we don't use a per-model
            # wrong-context assignment.
            wrong_assignment_hash = "all_others"
            wrong_counts = None

        # Also choose, per example, which *other* example to pull context from.
        example_seed_components = [
            f"{dataset_type}_wrong_context_source_row",
            str(_get_dataset_signature(dataset)),
        ]
        if dataset_index is not None:
            example_seed_components.append(f"dataset_index={int(dataset_index)}")
        example_seed_input = "::".join(example_seed_components)
        example_seed = int(hashlib.md5(example_seed_input.encode("utf-8")).hexdigest()[:8], 16)
        if wrong_context_all_others:
            by_model = _build_pubmedqa_wrong_context_example_indices_by_model(
                num_examples=num_examples,
                num_models=num_models,
                seed=example_seed,
            )
            dataset.pubmedqa_wrong_context_source_example_by_example_by_model = by_model.tolist()
            dataset.pubmedqa_wrong_context_source_hash_by_model = [
                hashlib.md5(by_model[:, m].tobytes()).hexdigest()[:12] for m in range(num_models)
            ]
        else:
            source_indices = _build_pubmedqa_wrong_context_example_indices(
                num_examples=num_examples,
                seed=example_seed,
            )
            dataset.pubmedqa_wrong_context_source_example_by_example = source_indices.tolist()
            dataset.pubmedqa_wrong_context_source_hash = hashlib.md5(source_indices.tobytes()).hexdigest()[:12]

    # Preserve existing PubMedQA fields for backwards compatibility.
    if dataset_type == "pubmedqa":
        dataset.pubmedqa_context_model_index = None
        dataset.pubmedqa_context_model_path = None

    return {
        "dataset_type": dataset_type,
        "assignment_hash": assignment_hash,
        "num_examples": int(num_examples),
        "model_context_counts": context_counts,
        "routing_seed": normalized_seed,
        "wrong_context_enabled": bool(wrong_context_enabled),
        "wrong_assignment_hash": wrong_assignment_hash,
        "wrong_context_counts": wrong_counts,
    }


def _ensure_pubmedqa_wrong_context_sources_materialized(dataset: Dataset, *, num_models: int) -> None:
    """
    Ensure PubMedQA wrong-context source indices + hashes exist on the dataset object.

    This is required for stable `prompt_variant` values (and therefore stable disk-cache keys)
    when `pubmedqa_wrong_context_all_others` is enabled.
    """
    if not bool(getattr(dataset, "pubmedqa_wrong_context_routing", False)):
        return
    if not bool(getattr(dataset, "pubmedqa_wrong_context_all_others", False)):
        # In the non-all-others mode, prompt_variant uses wrong_context_assignment_hash only.
        return

    n = len(getattr(dataset, "x", []) or [])
    if n <= 0:
        return

    existing = getattr(dataset, "pubmedqa_wrong_context_source_example_by_example_by_model", None)
    hashes = getattr(dataset, "pubmedqa_wrong_context_source_hash_by_model", None)
    if (
        isinstance(existing, list)
        and len(existing) == n
        and all(isinstance(row, list) and len(row) >= num_models for row in existing)
        and isinstance(hashes, list)
        and len(hashes) >= num_models
        and all(isinstance(h, str) and h for h in hashes[:num_models])
    ):
        return

    dataset_index = getattr(dataset, "pubmedqa_context_dataset_index", None)
    seed_components = [
        "pubmedqa_wrong_context_source_row",
        str(_get_dataset_signature(dataset)),
    ]
    if dataset_index is not None:
        try:
            seed_components.append(f"dataset_index={int(dataset_index)}")
        except Exception:
            pass
    seed_input = "::".join(seed_components)
    seed = int(hashlib.md5(seed_input.encode("utf-8")).hexdigest()[:8], 16)

    by_model = _build_pubmedqa_wrong_context_example_indices_by_model(
        num_examples=n,
        num_models=int(num_models),
        seed=seed,
    )
    dataset.pubmedqa_wrong_context_source_example_by_example_by_model = by_model.tolist()
    dataset.pubmedqa_wrong_context_source_hash_by_model = [
        hashlib.md5(by_model[:, m].tobytes()).hexdigest()[:12] for m in range(int(num_models))
    ]


def assign_pubmedqa_context_models(
    datasets: Sequence[Dataset],
    model_paths: Sequence[str],
    random_seed: Optional[int] = None,
) -> Dict[int, Dict[str, object]]:
    """Assign PubMedQA context routing metadata for all datasets that need it."""
    assignments: Dict[int, Dict[str, object]] = {}
    for idx, dataset in enumerate(datasets):
        selected = assign_pubmedqa_context_model(
            dataset,
            model_paths,
            random_seed=random_seed,
            dataset_index=idx,
        )
        if selected is not None:
            assignments[idx] = selected
    return assignments


def _get_pubmedqa_context_assignments(dataset: Dataset) -> Optional[np.ndarray]:
    """Return per-example context model indices for mixed-context datasets."""
    dataset_type = _get_mixed_context_dataset_type(dataset)
    if dataset_type is None:
        return None

    assignments = getattr(dataset, f"{dataset_type}_context_assignment_by_example", None)
    if not isinstance(assignments, list) or len(assignments) != len(dataset.x):
        return None
    try:
        return np.asarray(assignments, dtype=np.int32)
    except Exception:
        return None


def get_model_prompt_variant(
    dataset: Dataset,
    model_index: int,
) -> Optional[str]:
    """Return the prompt variant key used for a model on this dataset."""
    dataset_type = _get_mixed_context_dataset_type(dataset)
    if dataset_type is None:
        return None

    assignments = _get_pubmedqa_context_assignments(dataset)
    if assignments is None:
        raise RuntimeError(
            "Mixed-context dataset missing per-example assignments. "
            "Call assign_pubmedqa_context_models before cache checks/collection."
        )

    assignment_hash = getattr(dataset, f"{dataset_type}_context_assignment_hash", None)
    if not isinstance(assignment_hash, str) or not assignment_hash:
        assignment_hash = hashlib.md5(assignments.tobytes()).hexdigest()[:12]
        setattr(dataset, f"{dataset_type}_context_assignment_hash", assignment_hash)

    if dataset_type == "pubmedqa":
        wrong_hash = getattr(dataset, "pubmedqa_wrong_context_assignment_hash", None)
        all_others = bool(getattr(dataset, "pubmedqa_wrong_context_all_others", False))
        if all_others:
            # Ensure the per-model wrong-context source hashes exist so prompt_variant is stable.
            counts = getattr(dataset, "pubmedqa_context_assignment_counts", None)
            num_models = int(len(counts)) if isinstance(counts, list) and len(counts) > 0 else int(model_index) + 1
            _ensure_pubmedqa_wrong_context_sources_materialized(dataset, num_models=num_models)
            source_hashes = getattr(dataset, "pubmedqa_wrong_context_source_hash_by_model", None)
            if isinstance(source_hashes, list) and model_index < len(source_hashes):
                source_hash = source_hashes[model_index]
                if isinstance(source_hash, str) and source_hash:
                    return f"balanced_random_context_all_wrong_m{model_index}_{assignment_hash}_{source_hash}"
            return f"balanced_random_context_all_wrong_m{model_index}_{assignment_hash}"
        if isinstance(wrong_hash, str) and wrong_hash:
            return f"balanced_random_context_wrong_m{model_index}_{assignment_hash}_{wrong_hash}"
        return f"balanced_random_context_m{model_index}_{assignment_hash}"
    return f"article_random_context_m{model_index}_{assignment_hash}"


def get_model_specific_prompts(
    dataset: Dataset,
    model_index: int,
) -> List[str]:
    """Return prompt texts for this model, defaulting to dataset.x."""
    dataset_type = _get_mixed_context_dataset_type(dataset)
    if dataset_type is not None:
        with_context_attr = f"{dataset_type}_with_context_x"
        without_context_attr = f"{dataset_type}_without_context_x"
        with_context_prompts = getattr(dataset, with_context_attr, None)
        without_context_prompts = getattr(dataset, without_context_attr, None)
        assignments = _get_pubmedqa_context_assignments(dataset)

        if (
            isinstance(with_context_prompts, list)
            and isinstance(without_context_prompts, list)
            and assignments is not None
        ):
            if len(with_context_prompts) != len(without_context_prompts):
                raise ValueError(
                    f"{dataset_type} prompt variants have different lengths: "
                    f"with_context={len(with_context_prompts)}, "
                    f"without_context={len(without_context_prompts)}"
                )
            if len(with_context_prompts) != len(assignments):
                raise ValueError(
                    f"{dataset_type} assignment length does not match prompt length: "
                    f"assignments={len(assignments)}, prompts={len(with_context_prompts)}"
                )

            # Optional PubMedQA-only behavior: one model gets correct context, one model gets wrong context,
            # remaining models get the without-context prompt.
            if dataset_type == "pubmedqa" and bool(getattr(dataset, "pubmedqa_wrong_context_routing", False)):
                all_others = bool(getattr(dataset, "pubmedqa_wrong_context_all_others", False))
                if all_others:
                    wrong_by_model = getattr(dataset, "pubmedqa_wrong_context_x_by_model", None)
                    if not isinstance(wrong_by_model, list):
                        wrong_by_model = []
                    while len(wrong_by_model) <= model_index:
                        wrong_by_model.append(None)
                    if not isinstance(wrong_by_model[model_index], list) or len(wrong_by_model[model_index]) != len(assignments):
                        wrong_by_model[model_index] = _build_pubmedqa_wrong_context_prompts(
                            dataset, model_index=model_index
                        )
                        dataset.pubmedqa_wrong_context_x_by_model = wrong_by_model
                    wrong_prompts = wrong_by_model[model_index]

                    out: List[str] = []
                    for idx in range(len(assignments)):
                        if int(assignments[idx]) == model_index:
                            out.append(with_context_prompts[idx])
                        else:
                            out.append(wrong_prompts[idx])
                    return out

                wrong_prompts = getattr(dataset, "pubmedqa_wrong_context_x", None)
                if not isinstance(wrong_prompts, list) or len(wrong_prompts) != len(assignments):
                    wrong_prompts = _build_pubmedqa_wrong_context_prompts(dataset, model_index=None)
                    dataset.pubmedqa_wrong_context_x = wrong_prompts

                wrong_assignments = getattr(dataset, "pubmedqa_wrong_context_assignment_by_example", None)
                if not isinstance(wrong_assignments, list) or len(wrong_assignments) != len(assignments):
                    raise RuntimeError(
                        "pubmedqa_wrong_context_routing enabled but wrong-context assignments are missing. "
                        "Ensure assign_pubmedqa_context_models ran before cache checks/collection."
                    )

                out: List[str] = []
                for idx in range(len(assignments)):
                    if int(assignments[idx]) == model_index:
                        out.append(with_context_prompts[idx])
                    elif int(wrong_assignments[idx]) == model_index:
                        out.append(wrong_prompts[idx])
                    else:
                        out.append(without_context_prompts[idx])
                return out

            return [
                with_context_prompts[idx] if int(assignments[idx]) == model_index else without_context_prompts[idx]
                for idx in range(len(assignments))
            ]

    return dataset.x


def _build_pubmedqa_wrong_context_prompts(
    dataset: Dataset,
    model_index: Optional[int] = None,
) -> List[str]:
    """Render per-example 'wrong context' prompts for PubMedQA.

    Uses the same question/long_answer as the current example, but swaps in the context
    from a different randomly chosen example.
    """
    questions = getattr(dataset, "pubmedqa_questions", None)
    long_answers = getattr(dataset, "pubmedqa_long_answers", None)
    contexts = getattr(dataset, "pubmedqa_context_texts", None)
    template = getattr(dataset, "pubmedqa_prompt_template_with_context", None)
    if model_index is None:
        source_rows = getattr(dataset, "pubmedqa_wrong_context_source_example_by_example", None)
    else:
        source_rows = getattr(dataset, "pubmedqa_wrong_context_source_example_by_example_by_model", None)

    if not (isinstance(questions, list) and isinstance(long_answers, list) and isinstance(contexts, list)):
        raise RuntimeError(
            "PubMedQA wrong-context routing requires dataset to expose pubmedqa_questions, "
            "pubmedqa_long_answers, and pubmedqa_context_texts. Ensure you are using a standard "
            "PubMedQA dataset variant that stores these fields."
        )
    if not (len(questions) == len(long_answers) == len(contexts)):
        raise RuntimeError("PubMedQA raw field arrays must have identical lengths")

    n = len(questions)
    if model_index is None:
        if not isinstance(source_rows, list) or len(source_rows) != n:
            # Be robust to dataset transforms or older cached objects that may have dropped
            # the per-example wrong-context source indices. Rebuild deterministically.
            seed_components = [
                "pubmedqa_wrong_context_source_row",
                str(_get_dataset_signature(dataset)),
            ]
            dataset_index = getattr(dataset, "pubmedqa_context_dataset_index", None)
            if dataset_index is not None:
                try:
                    seed_components.append(f"dataset_index={int(dataset_index)}")
                except Exception:
                    pass
            seed_input = "::".join(seed_components)
            seed = int(hashlib.md5(seed_input.encode("utf-8")).hexdigest()[:8], 16)
            rebuilt = _build_pubmedqa_wrong_context_example_indices(num_examples=n, seed=seed).tolist()
            dataset.pubmedqa_wrong_context_source_example_by_example = rebuilt
            source_rows = rebuilt
    else:
        if (
            not isinstance(source_rows, list)
            or len(source_rows) != n
            or not all(isinstance(row, list) and len(row) > model_index for row in source_rows)
        ):
            # Be robust to older cached dataset objects that don't yet have per-model
            # wrong-context indices (e.g. after enabling pubmedqa_wrong_context_all_others).
            # IMPORTANT: this rebuild must use the *same seed construction* as the
            # initial assignment in `assign_pubmedqa_context_model`, otherwise the
            # derived per-model source hashes (used in prompt_variant -> cache keys)
            # can drift across runs even when the user changes nothing.
            counts = getattr(dataset, "pubmedqa_context_assignment_counts", None)
            if isinstance(counts, list) and len(counts) > 0:
                num_models = int(len(counts))
            else:
                assignments = getattr(dataset, "pubmedqa_context_assignment_by_example", None)
                if isinstance(assignments, list) and assignments:
                    num_models = int(max(int(a) for a in assignments) + 1)
                else:
                    num_models = int(model_index) + 1

            seed_components = [
                "pubmedqa_wrong_context_source_row",
                str(_get_dataset_signature(dataset)),
            ]
            seed_input = "::".join(seed_components)
            seed = int(hashlib.md5(seed_input.encode("utf-8")).hexdigest()[:8], 16)

            rebuilt = _build_pubmedqa_wrong_context_example_indices_by_model(
                num_examples=n,
                num_models=num_models,
                seed=seed,
            ).tolist()
            dataset.pubmedqa_wrong_context_source_example_by_example_by_model = rebuilt
            dataset.pubmedqa_wrong_context_source_hash_by_model = [
                hashlib.md5(np.asarray([row[m] for row in rebuilt], dtype=np.int32).tobytes()).hexdigest()[:12]
                for m in range(num_models)
            ]
            source_rows = rebuilt

    rendered: List[str] = []
    for i in range(n):
        src_idx = int(source_rows[i][model_index]) if model_index is not None else int(source_rows[i])
        if src_idx < 0 or src_idx >= n:
            raise RuntimeError("Wrong-context source index out of range")

        question = str(questions[i])
        long_answer = str(long_answers[i])
        wrong_context = str(contexts[src_idx])

        # Best-effort rendering: prefer the original prompt template if available.
        if isinstance(template, str) and template.strip():
            try:
                rendered.append(
                    template.format(
                        question=question,
                        context=wrong_context,
                        long_answer=long_answer,
                        text=question,
                        answer=long_answer,
                    )
                )
                continue
            except KeyError:
                # Fall back to default builder below.
                pass

        # Default fallback matches the built-in PubMedQA prompt.
        rendered.append(
            _build_pubmedqa_prompt_fallback(
                question=question,
                long_answer=long_answer,
                context_text=wrong_context,
            )
        )

    return rendered


def _build_pubmedqa_prompt_fallback(question: str, long_answer: str, context_text: str) -> str:
    """Local fallback prompt builder matching wagering.core.dataset defaults."""
    return (
        f"Question:\n{question}\n"
        f"Context:\n{context_text}\n"
        f"Long Answer:\n{long_answer}\n"
        "Is the long answer provided correct or incorrect? "
        "Answer with YES or NO. Answer:"
    )


def get_concatenated_router_prompts(
    dataset: Dataset,
    num_models: int,
    *,
    deduplicate: bool = True,
) -> List[str]:
    """
    Build router inputs by concatenating model-specific prompts for each example.

    For mixed-context datasets (PubMedQA/RACE), this exposes all prompt/context
    variants observed by the ensemble to the router while preserving per-model
    inference prompts for logit collection.
    """
    if num_models <= 0:
        return list(dataset.x)

    per_model_prompts: List[List[str]] = [
        get_model_specific_prompts(dataset, model_index=model_idx)
        for model_idx in range(num_models)
    ]

    if not per_model_prompts:
        return list(dataset.x)

    num_examples = len(per_model_prompts[0])
    for model_idx, prompts in enumerate(per_model_prompts):
        if len(prompts) != num_examples:
            raise ValueError(
                "Model-specific prompt length mismatch while building router prompts: "
                f"model_index={model_idx}, len={len(prompts)}, expected={num_examples}"
            )

    router_prompts: List[str] = []
    for example_idx in range(num_examples):
        pieces = [per_model_prompts[model_idx][example_idx] for model_idx in range(num_models)]
        if deduplicate:
            unique_pieces: List[str] = []
            seen = set()
            for piece in pieces:
                piece_text = str(piece)
                if piece_text in seen:
                    continue
                seen.add(piece_text)
                unique_pieces.append(piece_text)
            pieces = unique_pieces
        router_prompts.append("\n\n".join(str(piece) for piece in pieces))

    return router_prompts


def _resolve_label_to_index(
    label: object,
    option_tokens: List[str],
) -> int:
    """Resolve labels robustly, including case-insensitive YES/NO extraction."""
    option_lookup = {str(option).strip().lower(): idx for idx, option in enumerate(option_tokens)}

    if isinstance(label, bool):
        if "yes" in option_lookup and "no" in option_lookup:
            return option_lookup["yes"] if label else option_lookup["no"]
        return int(label)

    if isinstance(label, str):
        stripped = label.strip()
        if stripped in option_tokens:
            return option_tokens.index(stripped)

        lowered = stripped.lower()
        if lowered in option_lookup:
            return option_lookup[lowered]

        if lowered in {"y", "yes"} and "yes" in option_lookup:
            return option_lookup["yes"]
        if lowered in {"n", "no"} and "no" in option_lookup:
            return option_lookup["no"]

        raise ValueError(
            f"Could not map label '{label}' to option tokens {option_tokens}. "
            "For PubMedQA, expected yes/no labels."
        )

    return int(label)


def get_cached_logits_and_hidden_states_for_model(
    model_path: str,
    dataset: Dataset,
    option_tokens: List[str],
    prompt_variant: Optional[str] = None,
    model_index: Optional[int] = None,
    hidden_state_layers: Optional[Sequence[int]] = None,
) -> Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Get cached logits and hidden states for a model if available from disk.

    Args:
        model_path: Model path string
        dataset: Dataset instance
        option_tokens: List of option tokens (e.g., ['A', 'B', 'C', 'D'])

    Returns:
        Tuple of (logits, hidden_states, labels) if cached, else (None, None, None)
        logits shape: [num_examples, num_options]
        hidden_states shape: [num_examples, hidden_dim]
        labels shape: [num_examples]
    """
    model_key: str = model_path
    if _requires_slot_specific_cache(dataset):
        if model_index is None:
            raise ValueError(
                "Mixed-context cache lookups require model_index to disambiguate repeated model paths"
            )
        model_key = f"{model_path}::idx={int(model_index)}"
    cache_key = _wagering_logits_cache_key(
        model_key,
        dataset,
        option_tokens,
        prompt_variant,
        hidden_state_layers=hidden_state_layers,
    )
    cache_path = _get_cache_path(cache_key)

    if not cache_path.exists():
        # Backwards-compatible lookup: older cache artifacts were created before
        # deterministic dataset config signatures existed. If we have a config-based
        # signature but the corresponding file is missing, try the legacy heuristic
        # signature as a secondary key before declaring a miss.
        dataset_cache_config = getattr(dataset, "cache_dataset_config", None)
        if isinstance(dataset_cache_config, dict):
            legacy_dataset_key = _get_legacy_dataset_signature(dataset)
            option_key = tuple(option_tokens)
            pv = prompt_variant or "default"
            mixed_context_type = _get_mixed_context_dataset_type(dataset)
            if mixed_context_type == "pubmedqa":
                legacy_key: Tuple[Any, ...] = (
                    model_key,
                    legacy_dataset_key,
                    option_key,
                    pv,
                    PUBMEDQA_LOGITS_CACHE_NAMESPACE,
                )
            elif mixed_context_type == "race":
                legacy_key = (
                    model_key,
                    legacy_dataset_key,
                    option_key,
                    pv,
                    RACE_LOGITS_CACHE_NAMESPACE,
                )
            else:
                legacy_key = (model_key, legacy_dataset_key, option_key, pv)

            legacy_path = _get_cache_path(legacy_key)
            if legacy_path.exists():
                cache_path = legacy_path

    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            logits = data["logits"] if "logits" in data else None
            hidden_states = data["hidden_states"] if "hidden_states" in data else None
            labels = data["labels"] if "labels" in data else None
            cached_requested_hidden_state_layers = (
                data["requested_hidden_state_layers"].astype(np.int32).tolist()
                if "requested_hidden_state_layers" in data
                else None
            )

            # Handle hidden_states if it was pickled
            if "hidden_states_pickle" in data:
                hidden_states = pickle.loads(data["hidden_states_pickle"].item())

            hidden_states_by_layer: Optional[Dict[int, np.ndarray]] = None
            if "hidden_states_by_layer_pickle" in data:
                raw = pickle.loads(data["hidden_states_by_layer_pickle"].item())
                if isinstance(raw, dict):
                    parsed: Dict[int, np.ndarray] = {}
                    for k, v in raw.items():
                        try:
                            kk = int(k)
                        except Exception:
                            continue
                        if v is None:
                            continue
                        arr = np.asarray(v, dtype=np.float32)
                        if arr.ndim != 2:
                            continue
                        parsed[kk] = arr
                    hidden_states_by_layer = parsed

            try:
                if hidden_states_by_layer is not None:
                    hidden_states = _extract_hidden_state_features_from_layer_map(
                        hidden_states_by_layer,
                        hidden_state_layers,
                    )
                else:
                    hidden_states = extract_hidden_state_features(
                        hidden_states,
                        hidden_state_layers,
                        cached_requested_hidden_state_layers=cached_requested_hidden_state_layers,
                    )
            except RuntimeError as layer_err:
                log.debug(
                    "Cache layer mismatch for model %s (prompt_variant=%s): %s",
                    model_key,
                    prompt_variant or "default",
                    layer_err,
                )
                return None, None, None

            row_map = getattr(dataset, "cache_source_row_indices", None)
            n_src = int(getattr(dataset, "cache_source_num_examples", 0) or 0)
            if row_map is not None and n_src > 0 and logits is not None:
                if logits.shape[0] == n_src:
                    idx = np.asarray(row_map, dtype=np.int64)
                    logits = np.asarray(logits, dtype=np.float32)[idx, :]
                    if labels is not None and np.asarray(labels).shape[0] == n_src:
                        lab = np.asarray(labels)
                        labels = lab[idx, ...] if lab.ndim > 1 else lab[idx]
                    if hidden_states is not None:
                        if isinstance(hidden_states, list):
                            hidden_states = [
                                np.asarray(h, dtype=np.float32)[idx, :]
                                for h in hidden_states
                            ]
                        else:
                            hs = np.asarray(hidden_states, dtype=np.float32)
                            if hs.ndim == 3:
                                hidden_states = hs[:, idx, :]
                            else:
                                hidden_states = hs[idx, :]
                elif logits.shape[0] != len(dataset.x):
                    log.debug(
                        "Cache shape %s vs view %d / full %d; skipping source-row reindex (prompt_variant=%s)",
                        getattr(logits, "shape", None),
                        len(dataset.x),
                        n_src,
                        prompt_variant or "default",
                    )
                    return None, None, None

            log.debug(
                "Cache hit for model %s and dataset size %d (prompt_variant=%s)",
                model_key,
                len(dataset.x),
                prompt_variant or "default",
            )
            return logits, hidden_states, labels
        except Exception as e:
            raise Exception(f"Error loading cache from {cache_path}: {e}")
    return None, None, None


def set_cached_logits_and_hidden_states_for_model(
    model: WhiteboxModel,
    dataset: Dataset,
    option_tokens: List[str],
    logits: Optional[np.ndarray],
    hidden_states: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    prompt_variant: Optional[str] = None,
    model_index: Optional[int] = None,
    hidden_state_layers: Optional[Sequence[int]] = None,
):
    """
    Cache logits and hidden states for a single model on disk.
    
    Args:
        model: WhiteboxModel instance
        dataset: Dataset instance
        option_tokens: List of option tokens
        logits: Optional np.ndarray of shape [num_examples, num_options]
        hidden_states: Optional np.ndarray of shape [num_examples, hidden_dim]
        labels: Optional np.ndarray of shape [num_examples]
    """
    model_key = _get_model_path_key(model)
    if _requires_slot_specific_cache(dataset):
        if model_index is None:
            raise ValueError(
                "Mixed-context cache writes require model_index to disambiguate repeated model paths"
            )
        model_key = f"{model_key}::idx={int(model_index)}"
    cache_key = _wagering_logits_cache_key(
        model_key,
        dataset,
        option_tokens,
        prompt_variant,
        hidden_state_layers=hidden_state_layers,
    )
    cache_path = _get_cache_path(cache_key)
    
    # Load existing cache entry if present
    existing_data = {}
    existing_hidden_states_by_layer: Dict[int, np.ndarray] = {}
    if cache_path.exists():
        try:
            existing = np.load(cache_path, allow_pickle=True)
            if "logits" in existing:
                existing_data["logits"] = existing["logits"]
            if "hidden_states" in existing:
                existing_data["hidden_states"] = existing["hidden_states"]
            if "hidden_states_pickle" in existing:
                existing_data["hidden_states"] = pickle.loads(existing["hidden_states_pickle"].item())
            if "labels" in existing:
                existing_data["labels"] = existing["labels"]
            if "hidden_states_by_layer_pickle" in existing:
                raw = pickle.loads(existing["hidden_states_by_layer_pickle"].item())
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        try:
                            kk = int(k)
                        except Exception:
                            continue
                        if v is None:
                            continue
                        arr = np.asarray(v, dtype=np.float32)
                        if arr.ndim != 2:
                            continue
                        existing_hidden_states_by_layer[kk] = arr
        except Exception as e:
            log.warning(f"Corrupted cache at {cache_path}: {e}. Deleting and rebuilding.")
            try:
                cache_path.unlink(missing_ok=True)
            except Exception as delete_err:
                log.warning(f"Failed to delete corrupted cache file {cache_path}: {delete_err}")
            existing_data = {}
            existing_hidden_states_by_layer = {}

    row_map = getattr(dataset, "cache_source_row_indices", None)
    n_src = int(getattr(dataset, "cache_source_num_examples", 0) or 0)
    if row_map is not None and n_src > 0 and logits is not None:
        idx = np.asarray(row_map, dtype=np.int64)
        if logits.shape[0] == len(idx):
            n_opt = int(logits.shape[1])
            full_logits = np.zeros((n_src, n_opt), dtype=np.float32)
            ex_log = existing_data.get("logits")
            if isinstance(ex_log, np.ndarray) and ex_log.shape[0] == n_src:
                full_logits = np.asarray(ex_log, dtype=np.float32).copy()
            full_logits[idx, :] = np.asarray(logits, dtype=np.float32)
            logits = full_logits
    if row_map is not None and n_src > 0 and labels is not None:
        idx = np.asarray(row_map, dtype=np.int64)
        lab = np.asarray(labels)
        if lab.shape[0] == len(idx):
            full_labels = np.zeros(n_src, dtype=np.int32)
            ex_lb = existing_data.get("labels")
            if isinstance(ex_lb, np.ndarray) and ex_lb.shape[0] == n_src:
                full_labels = np.asarray(ex_lb, dtype=np.int32).copy()
            full_labels[idx] = lab.astype(np.int32, copy=False)
            labels = full_labels
    if row_map is not None and n_src > 0 and hidden_states is not None:
        idx = np.asarray(row_map, dtype=np.int64)
        ex_h = existing_data.get("hidden_states")
        if isinstance(hidden_states, np.ndarray) and hidden_states.shape[0] == len(idx):
            mat = np.asarray(hidden_states, dtype=np.float32)
            if mat.ndim == 2:
                _, d = mat.shape
                full_h = np.zeros((n_src, d), dtype=np.float32)
                if isinstance(ex_h, np.ndarray) and ex_h.shape[0] == n_src and ex_h.ndim == 2:
                    full_h = np.asarray(ex_h, dtype=np.float32).copy()
                full_h[idx, :] = mat
                hidden_states = full_h
        elif (
            isinstance(hidden_states, list)
            and ex_h
            and isinstance(ex_h, list)
            and len(hidden_states) == len(idx)
            and len(ex_h) == len(hidden_states)
        ):
            merged: List[np.ndarray] = []
            for h_new, h_prev in zip(hidden_states, ex_h):
                a = np.asarray(h_new, dtype=np.float32)
                b = np.asarray(h_prev, dtype=np.float32)
                if a.shape[0] != len(idx) or b.shape[0] != n_src or a.ndim != 2:
                    merged = []
                    break
                _, d = a.shape
                full = np.zeros((n_src, d), dtype=np.float32)
                full += b
                full[idx, :] = a
                merged.append(full)
            if merged:
                hidden_states = merged

    # Merge with existing cache entry
    cache_dict = {}
    if logits is not None:
        cache_dict["logits"] = logits.copy() if isinstance(logits, np.ndarray) else logits
    elif "logits" in existing_data:
        cache_dict["logits"] = existing_data["logits"]
    
    if hidden_states is not None:
        cache_dict["hidden_states"] = hidden_states
    elif "hidden_states" in existing_data:
        cache_dict["hidden_states"] = existing_data["hidden_states"]
    
    if labels is not None:
        cache_dict["labels"] = labels.copy() if isinstance(labels, np.ndarray) else labels
    elif "labels" in existing_data:
        cache_dict["labels"] = existing_data["labels"]
    
    # Save to disk
    try:
        save_dict = {}
        if "logits" in cache_dict and cache_dict["logits"] is not None:
            save_dict["logits"] = cache_dict["logits"].astype(np.float32) if isinstance(cache_dict["logits"], np.ndarray) else cache_dict["logits"]
        if "labels" in cache_dict and cache_dict["labels"] is not None:
            save_dict["labels"] = cache_dict["labels"].astype(np.int32) if isinstance(cache_dict["labels"], np.ndarray) else cache_dict["labels"]
        normalized_requested_layers = _normalize_hidden_state_layers(hidden_state_layers)

        # Incremental hidden-state caching: store per-layer arrays so different layer
        # requests (e.g., calibration [-1] vs wagering [-12]) can coexist.
        hidden_states_by_layer: Dict[int, np.ndarray] = dict(existing_hidden_states_by_layer)
        if (
            "hidden_states" in cache_dict
            and isinstance(cache_dict["hidden_states"], np.ndarray)
            and cache_dict["hidden_states"] is not None
            and normalized_requested_layers is not None
        ):
            hs = np.asarray(cache_dict["hidden_states"], dtype=np.float32)
            if hs.ndim == 3 and hs.shape[1] == len(normalized_requested_layers):
                for pos, layer_id in enumerate(normalized_requested_layers):
                    hidden_states_by_layer[int(layer_id)] = hs[:, pos, :].astype(np.float32, copy=False)
            elif hs.ndim == 2 and len(normalized_requested_layers) == 1:
                hidden_states_by_layer[int(normalized_requested_layers[0])] = hs.astype(np.float32, copy=False)

        if hidden_states_by_layer:
            save_dict["hidden_states_by_layer_pickle"] = np.void(pickle.dumps(hidden_states_by_layer))

        # Keep legacy fields for backward compatibility / inspection.
        if "hidden_states" in cache_dict and cache_dict["hidden_states"] is not None:
            # Handle different data types for hidden_states
            if isinstance(cache_dict["hidden_states"], list):
                save_dict["hidden_states_pickle"] = np.void(pickle.dumps(cache_dict["hidden_states"]))
            else:
                save_dict["hidden_states"] = cache_dict["hidden_states"].astype(np.float32) if isinstance(cache_dict["hidden_states"], np.ndarray) else cache_dict["hidden_states"]
        if normalized_requested_layers is not None:
            save_dict["requested_hidden_state_layers"] = np.asarray(normalized_requested_layers, dtype=np.int32)
        
        np.savez_compressed(cache_path, **save_dict)
        
        items_cached = []
        if logits is not None:
            items_cached.append("logits")
        if hidden_states is not None:
            items_cached.append("hidden_states")
        if labels is not None:
            items_cached.append("labels")
        log.info(
            "Cached %s for model %s and dataset size %d (prompt_variant=%s) to %s",
            ", ".join(items_cached) if items_cached else "data",
            model_key,
            len(dataset.x),
            prompt_variant or "default",
            cache_path,
        )
    except Exception as e:
        raise Exception(f"Error saving cache to {cache_path}: {e}", exc_info=True)


def shuffle_cached_arrays(
    array: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Shuffle a cached array using the provided indices.
    
    Args:
        array: np.ndarray to shuffle (any shape, first dimension is samples)
        indices: np.ndarray of indices to use for shuffling
        
    Returns:
        Shuffled array
    """
    return array[indices]


def filter_dataset_by_token_length(
    dataset: Dataset,
    max_prompt_tokens: int,
    tokenizer=None,
) -> Dataset:
    """
    Filter a Dataset to remove samples that exceed max_prompt_tokens.
    
    Args:
        dataset: Dataset to filter
        max_prompt_tokens: Maximum number of tokens allowed per prompt
        tokenizer: Optional tokenizer to use for counting tokens
        
    Returns:
        Filtered Dataset
    """
    if max_prompt_tokens is None or max_prompt_tokens <= 0:
        return dataset
    
    log.info(f"Filtering dataset: dropping samples longer than {max_prompt_tokens} tokens")
    kept_x, kept_y = [], []
    kept_images = [] if dataset.images is not None else None
    
    for idx, (text, target) in enumerate(zip(dataset.x, dataset.y)):
        if tokenizer is None:
            token_count = len(text.split())
        else:
            token_count = len(
                tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )
        
        if token_count <= max_prompt_tokens:
            kept_x.append(text)
            kept_y.append(target)
            if kept_images is not None:
                kept_images.append(dataset.images[idx])
    
    dropped = len(dataset.x) - len(kept_x)
    if dropped > 0:
        log.info(f"Dropped {dropped} samples due to length > {max_prompt_tokens} tokens")
    
    # Create new Dataset with filtered data
    filtered_dataset = Dataset(
        kept_x, 
        kept_y, 
        dataset.batch_size, 
        images=kept_images
    )
    
    return filtered_dataset


def _resolve_option_token_ids(
    model: WhiteboxModel, option_tokens: List[str], sample_prompt: str = None
) -> List[int]:
    """
    Resolve single-token IDs for answer option strings (e.g., 'A', 'B', 'C', 'D').

    Args:
        model: WhiteboxModel instance
        option_tokens: List of option strings (e.g., ['A', 'B', 'C', 'D'])
        sample_prompt: A sample prompt from the dataset (optional)

    Returns:
        List of token IDs, one per option token

    Raises:
        ValueError: If an option doesn't map to a single token in context
    """
    token_ids: List[int] = []
    
    # Determine prompt suffix
    if sample_prompt is None:
        prompt_suffix = "Answer: "
    else:
        lines = sample_prompt.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if ':' in last_line:
                prompt_suffix = last_line.rsplit(':', 1)[0] + ': '
            else:
                prompt_suffix = last_line + ' ' if last_line else "Answer: "
        else:
            prompt_suffix = "Answer: "
    
    base_prompt = sample_prompt if sample_prompt else prompt_suffix
    
    # Check if we need to handle chat template
    use_chat_template = (
        model.instruct
        and hasattr(model.tokenizer, 'chat_template')
        and model.tokenizer.chat_template is not None
    )
    
    if use_chat_template:
        try:
            chat = [{"role": "user", "content": base_prompt}]
            formatted_base = model.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
            base_ids = model.tokenizer.encode(formatted_base, add_special_tokens=False)
            
            for opt in option_tokens:
                formatted_with_opt = formatted_base + opt
                test_ids = model.tokenizer.encode(formatted_with_opt, add_special_tokens=False)
                opt_token_ids = test_ids[len(base_ids):]
                
                if len(opt_token_ids) == 1:
                    token_ids.append(opt_token_ids[0])
                elif len(opt_token_ids) > 1:
                    log.warning(
                        f"Option '{opt}' spans {len(opt_token_ids)} tokens. Using first token."
                    )
                    token_ids.append(opt_token_ids[0])
                else:
                    ids = model.tokenizer.encode(opt, add_special_tokens=False)
                    if len(ids) != 1:
                        raise ValueError(f"Option '{opt}' could not be resolved in context.")
                    token_ids.append(ids[0])
                    log.warning(f"Using standalone tokenization for '{opt}' as fallback.")
        except (ValueError, TypeError) as e:
            log.warning(f"Chat template failed: {e}. Falling back to plain tokenization.")
            use_chat_template = False
    
    if not use_chat_template:
        tokenized_base = model.tokenize([base_prompt])
        base_ids = tokenized_base['input_ids'][0].tolist()
        
        for opt in option_tokens:
            test_prompt = base_prompt + opt
            tokenized_test = model.tokenize([test_prompt])
            test_ids = tokenized_test['input_ids'][0].tolist()
            
            if len(test_ids) > len(base_ids) and test_ids[:len(base_ids)] == base_ids:
                opt_token_ids = test_ids[len(base_ids):]
                # opt_token_ids is guaranteed to have at least 1 element here
                if len(opt_token_ids) == 1:
                    token_ids.append(opt_token_ids[0])
                else:  # len(opt_token_ids) > 1
                    # We score only the first generation step, so use the first
                    # token when an option is split into multiple sub-tokens.
                    log.warning(
                        "Option '%s' spans %d tokens in context. "
                        "Using first token id=%s from %s.",
                        opt,
                        len(opt_token_ids),
                        opt_token_ids[0],
                        opt_token_ids,
                    )
                    token_ids.append(opt_token_ids[0])
            else:
                # Fallback: context extraction failed, use standalone tokenization
                ids = model.tokenizer.encode(opt, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"Option '{opt}' could not be resolved in context.")
                token_ids.append(ids[0])
                log.warning(f"Using standalone tokenization for '{opt}' as fallback.")
    log.info(
        f"Resolved option token IDs for {getattr(model.tokenizer, 'name_or_path', 'unknown')}: "
        f"{dict(zip(option_tokens, token_ids))}"
    )
    
    return token_ids


def collect_option_logits_and_hidden_states_for_model(
    model: WhiteboxModel,
    dataset: Dataset,
    option_tokens: List[str],
    max_new_tokens: int = 1,
    model_identifier: Optional[str] = None,
    model_index: int = 0,
    hidden_state_layers: Optional[Sequence[int]] = None,
    collect_hidden_states: bool = True,
    model_prompts: Optional[List[str]] = None,
    prompt_variant: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Collect per-option log-probabilities AND hidden states for a model in a single forward pass.
    
    Args:
        model: WhiteboxModel instance
        dataset: Dataset instance
        option_tokens: List of option tokens (e.g., ['A', 'B', 'C', 'D'])
        max_new_tokens: Maximum number of tokens to generate (default: 1)
        
    Returns:
        logits: np.ndarray, shape [num_examples, num_options]
        hidden_states: np.ndarray, shape [num_examples, num_selected_layers, hidden_dim] or None
        labels: np.ndarray, shape [num_examples]
    """
    model_device = model.device()
    model_path = str(model_identifier) if model_identifier is not None else str(model.model_path)

    if model_prompts is None:
        model_prompts = get_model_specific_prompts(dataset, model_index=model_index)
    if prompt_variant is None:
        prompt_variant = get_model_prompt_variant(dataset, model_index=model_index)

    if len(model_prompts) != len(dataset.y):
        raise ValueError(
            "Prompt/label length mismatch while collecting logits. "
            f"prompts={len(model_prompts)}, labels={len(dataset.y)}"
        )

    if prompt_variant is not None:
        log.info("Using prompt_variant='%s' for model %s", prompt_variant, model_path)

    if torch.cuda.is_available() and getattr(model_device, "type", None) == "cuda":
        try:
            torch.cuda.set_device(model_device)
        except Exception as e:
            raise RuntimeError(f"Could not set CUDA device to {model_device}: {e}") from e
    
    sample_prompt = model_prompts[0] if len(model_prompts) > 0 else None
    option_token_ids = _resolve_option_token_ids(model, option_tokens, sample_prompt=sample_prompt)
    
    all_log_probs: List[np.ndarray] = []
    all_hidden_states: List[np.ndarray] = []
    all_labels: List[int] = []
    selected_layer_indices: Optional[Tuple[int, ...]] = None
    
    if len(model_prompts) == 0:
        raise ValueError("Dataset is empty (0 examples).")

    for batch_start in range(0, len(model_prompts), dataset.batch_size):
        batch_end = min(batch_start + dataset.batch_size, len(model_prompts))
        batch_x = model_prompts[batch_start:batch_end]
        batch_y = dataset.y[batch_start:batch_end]

        batch = model.tokenize(batch_x)
        batch = {k: v.to(model_device) for k, v in batch.items()}
        
        generation = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=collect_hidden_states,
        )
        
        scores: List[torch.Tensor] = generation.scores
        if not scores:
            raise RuntimeError("Model.generate returned no scores.")
        first_step_scores = scores[0]
        
        with torch.no_grad():
            batch_log_probs = torch.stack(
                [first_step_scores[:, tid] for tid in option_token_ids],
                dim=-1,
            )
        
        # Extract hidden states when requested.
        if collect_hidden_states and hasattr(generation, 'hidden_states') and generation.hidden_states is not None:
            try:
                # Different HF model implementations populate `generation.hidden_states`
                # differently across steps. Some steps may only include a subset of
                # layers, so choose the step that exposes the *most* layers to keep
                # `hidden_state_layers` selection strict and meaningful.
                steps = list(generation.hidden_states)
                if len(steps) == 0:
                    raise ValueError("Model returned empty generation.hidden_states")

                def _layer_count(step_obj: object) -> int:
                    if isinstance(step_obj, tuple):
                        # Many models include an embedding slot; count transformer layers only.
                        return max(0, len(step_obj) - 1)
                    return 0

                best_step_idx = max(range(len(steps)), key=lambda i: _layer_count(steps[i]))
                selected_step_hidden = steps[best_step_idx]

                if isinstance(selected_step_hidden, tuple):
                    # Exclude embedding slot; keep only requested transformer layers.
                    transformer_hidden_states = selected_step_hidden[1:] if len(selected_step_hidden) > 1 else selected_step_hidden
                    if len(transformer_hidden_states) == 0:
                        raise ValueError(
                            "Model returned an empty transformer hidden-state tuple "
                            f"(selected_step={best_step_idx}, tuple_len={len(selected_step_hidden)})"
                        )

                    if selected_layer_indices is None:
                        selected_layer_indices = _resolve_transformer_layer_indices(
                            hidden_state_layers,
                            len(transformer_hidden_states),
                        )

                    per_layer_last_token = []
                    for layer_idx in selected_layer_indices:
                        layer_hidden = transformer_hidden_states[layer_idx]
                        if layer_hidden.dim() == 3:
                            per_layer_last_token.append(layer_hidden[:, -1, :])
                        elif layer_hidden.dim() == 2:
                            per_layer_last_token.append(layer_hidden)
                        else:
                            raise ValueError(
                                f"Unexpected hidden state shape: {layer_hidden.shape}"
                            )

                    all_layers_last_token_hidden = torch.stack(per_layer_last_token, dim=1)
                else:
                    if selected_step_hidden.dim() == 3:
                        last_token_hidden = selected_step_hidden[:, -1, :]
                    elif selected_step_hidden.dim() == 2:
                        last_token_hidden = selected_step_hidden
                    else:
                        raise ValueError(f"Unexpected hidden state shape: {selected_step_hidden.shape}")
                    all_layers_last_token_hidden = last_token_hidden.unsqueeze(1)
            except Exception as e:
                raise RuntimeError(
                    f"Could not extract hidden states from model {model.model_path}: {e}"
                ) from e
        elif collect_hidden_states:
            raise RuntimeError(
                f"Model {model.model_path} did not return hidden_states."
            )
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                log.warning(f"CUDA synchronization error: {e}")
                raise
        
        if len(all_log_probs) == 0:
            log.info(f"Extracting logits for option tokens: {dict(zip(option_tokens, option_token_ids))}")
            token_names = {
                opt: model.tokenizer.convert_ids_to_tokens([tid])[0]
                for opt, tid in zip(option_tokens, option_token_ids)
            }
            log.info(f"Token names: {token_names}")
        
        # NumPy cannot materialize bfloat16 tensors directly; normalize to float32 first.
        all_log_probs.append(batch_log_probs.detach().to(dtype=torch.float32).cpu().numpy())
        if collect_hidden_states:
            all_hidden_states.append(all_layers_last_token_hidden.detach().to(dtype=torch.float32).cpu().numpy())
        
        del generation
        del scores
        del first_step_scores
        del batch_log_probs
        if collect_hidden_states:
            del all_layers_last_token_hidden
        
        if len(all_log_probs) % 10 == 0 and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                log.warning(f"CUDA error during cache clearing: {e}")
                try:
                    torch.cuda.synchronize()
                except:
                    pass
        
        for y in batch_y:
            idx = _resolve_label_to_index(y, option_tokens)
            all_labels.append(idx)
    
    if len(all_log_probs) == 0:
        raise ValueError("No batches were processed.")
    
    logits = np.concatenate(all_log_probs, axis=0)
    hidden_states = np.concatenate(all_hidden_states, axis=0) if collect_hidden_states else None
    labels = np.asarray(all_labels, dtype=np.int32)
    return logits, hidden_states, labels


def aggregate_logits_log_pooling(
    llm_logits: np.ndarray,
    wagers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted log pooling across multiple LLMs.

    Args:
        llm_logits: np.ndarray of shape [num_llms, num_options]
        wagers: np.ndarray of shape [num_llms]

    Returns:
        pooled_log_probs: np.ndarray of shape [num_options]
        pooled_probs: np.ndarray of shape [num_options]
    """
    llm_logits = np.asarray(llm_logits, dtype=np.float32)
    wagers = np.asarray(wagers, dtype=np.float32)
    if llm_logits.ndim != 2:
        raise ValueError(f"llm_logits must have shape [num_llms, num_options], got {llm_logits.shape}")
    if wagers.ndim != 1 or wagers.shape[0] != llm_logits.shape[0]:
        raise ValueError(f"wagers shape mismatch")

    max_logits = np.max(llm_logits, axis=1, keepdims=True)
    stabilized = llm_logits - max_logits
    log_norm = max_logits + np.log(np.exp(stabilized).sum(axis=1, keepdims=True))
    log_probs = llm_logits - log_norm

    weighted_log = wagers[:, None] * log_probs
    pooled_log_unnorm = weighted_log.sum(axis=0)

    max_pooled = np.max(pooled_log_unnorm)
    stabilized_pooled = pooled_log_unnorm - max_pooled
    log_z = max_pooled + np.log(np.exp(stabilized_pooled).sum())
    pooled_log_probs = pooled_log_unnorm - log_z
    pooled_probs = np.exp(pooled_log_probs)
    return pooled_log_probs, pooled_probs


def update_wagers(
    llm_probs: np.ndarray,
    gold_answer: int,
    current_wagers: np.ndarray,
    state: Optional[Dict] = None,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Oracle-style wager update rule for multi-LLM ensembles.

    Args:
        llm_probs: np.ndarray, shape [num_llms, num_options]
        gold_answer: int
        current_wagers: np.ndarray, shape [num_llms]
        state: Optional user-defined dictionary

    Returns:
        new_wagers: np.ndarray
        new_state: Optional[Dict]
    """
    llm_probs = np.asarray(llm_probs, dtype=np.float64)
    current_wagers = np.asarray(current_wagers, dtype=np.float64)

    if llm_probs.ndim != 2:
        raise ValueError(f"llm_probs must have shape [num_llms, num_options], got {llm_probs.shape}")
    num_llms, num_options = llm_probs.shape
    if current_wagers.shape != (num_llms,):
        raise ValueError(f"current_wagers shape mismatch")
    if not (0 <= gold_answer < num_options):
        raise ValueError(f"gold_answer out of bounds")

    k = int(gold_answer)
    w_i = current_wagers
    p_i_k = llm_probs[:, k]
    w_i_k = w_i * p_i_k

    eps = 1e-12
    w_i_k_safe = np.clip(w_i_k, eps, None)
    p_i_k_safe = np.clip(p_i_k, eps, None)

    sum_w_j_k = np.sum(w_i_k_safe)
    if sum_w_j_k <= 0.0:
        return current_wagers.astype(np.float32), state

    log_p_i_k = np.log(p_i_k_safe)
    mean_log_term = float(np.sum(log_p_i_k * w_i_k_safe) / sum_w_j_k)

    delta = w_i_k_safe * (1.0 + log_p_i_k - mean_log_term)
    new_wagers = w_i + delta - w_i_k_safe

    new_wagers = np.maximum(new_wagers, 0.0)

    return new_wagers.astype(np.float32), state


def run_online_ensemble(
    all_model_logits: List[np.ndarray],
    labels: np.ndarray,
    initial_wagers: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Simple online replay engine over pre-computed logits.

    Returns:
        Dictionary with:
            - pooled_probs: [num_examples, num_options]
            - pooled_pred:  [num_examples]
            - labels:       [num_examples]
            - wagers_history: [num_examples + 1, num_models]
    """
    num_models = len(all_model_logits)
    if num_models == 0:
        raise ValueError("all_model_logits must contain at least one model.")
    num_examples, num_options = all_model_logits[0].shape
    for idx, arr in enumerate(all_model_logits):
        if arr.shape != (num_examples, num_options):
            raise ValueError(f"Model {idx} logits shape mismatch")

    if initial_wagers is None:
        wagers = np.ones(num_models, dtype=np.float32) / float(num_models)
    else:
        wagers = np.asarray(initial_wagers, dtype=np.float32)
        if wagers.shape != (num_models,):
            raise ValueError(f"initial_wagers shape mismatch")

    pooled_probs_all = np.zeros((num_examples, num_options), dtype=np.float32)
    pooled_pred_all = np.zeros((num_examples,), dtype=np.int32)
    wagers_history = np.zeros((num_examples + 1, num_models), dtype=np.float32)
    wagers_history[0] = wagers
    state: Optional[Dict] = None

    # Pre-compute per-model probabilities
    model_probs = [None] * num_models
    for i in range(num_models):
        logits_i = all_model_logits[i].astype(np.float32)
        max_i = np.max(logits_i, axis=1, keepdims=True)
        stabilized_i = logits_i - max_i
        log_z_i = max_i + np.log(np.exp(stabilized_i).sum(axis=1, keepdims=True))
        model_probs[i] = np.exp(logits_i - log_z_i)

    for t in range(num_examples):
        llm_logits_t = np.stack([all_model_logits[i][t] for i in range(num_models)], axis=0)
        pooled_log_t, pooled_probs_t = aggregate_logits_log_pooling(llm_logits_t, wagers)
        pooled_probs_all[t] = pooled_probs_t
        pooled_pred_all[t] = int(np.argmax(pooled_probs_t))

        llm_probs_t = np.stack([model_probs[i][t] for i in range(num_models)], axis=0)
        wagers, state = update_wagers(
            llm_probs=llm_probs_t,
            gold_answer=int(labels[t]),
            current_wagers=wagers,
            state=state,
        )
        wagers_history[t + 1] = wagers

    return {
        "pooled_probs": pooled_probs_all,
        "pooled_pred": pooled_pred_all,
        "labels": labels.astype(np.int32),
        "wagers_history": wagers_history,
    }


def _calculate_cumulative_accuracy(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Calculate cumulative accuracy over time."""
    num_examples = len(predictions)
    cumulative_accuracy = np.zeros(num_examples, dtype=np.float32)
    
    correct_count = 0
    for t in range(num_examples):
        if predictions[t] == labels[t]:
            correct_count += 1
        cumulative_accuracy[t] = correct_count / (t + 1)
    
    return cumulative_accuracy


def _calculate_cumulative_auc(
    probs: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Calculate cumulative AUC over time using max probability as confidence."""
    if not SKLEARN_AVAILABLE:
        log.warning("sklearn not available; AUC calculation skipped")
        return np.full(len(predictions), np.nan)
    
    num_examples = len(predictions)
    cumulative_auc = np.zeros(num_examples, dtype=np.float32)
    
    max_probs = probs.max(axis=1)
    correctness = (predictions == labels).astype(int)
    
    for t in range(num_examples):
        if t < 1:
            cumulative_auc[t] = np.nan
        else:
            try:
                correctness_t = correctness[:t+1]
                max_probs_t = max_probs[:t+1]
                
                if len(np.unique(correctness_t)) < 2:
                    cumulative_auc[t] = np.nan
                else:
                    auc_value = roc_auc_score(correctness_t, max_probs_t)
                    cumulative_auc[t] = auc_value
            except ValueError:
                cumulative_auc[t] = np.nan
    
    return cumulative_auc


def plot_accuracy_and_auc_over_time(
    all_model_logits: List[np.ndarray],
    ensemble_result: Dict[str, np.ndarray],
    model_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """Plot accuracy and AUC over time for individual models and ensemble."""
    if not MATPLOTLIB_AVAILABLE:
        log.warning("matplotlib not available; skipping plot generation")
        return
    
    required_keys = ["labels", "pooled_probs", "pooled_pred"]
    missing_keys = [key for key in required_keys if key not in ensemble_result]
    if missing_keys:
        raise ValueError(f"ensemble_result missing required keys: {missing_keys}")
    
    labels = ensemble_result["labels"]
    pooled_probs = ensemble_result["pooled_probs"]
    pooled_pred = ensemble_result["pooled_pred"]
    
    num_examples = len(labels)
    num_models = len(all_model_logits)
    
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(num_models)]
    
    # Calculate per-model predictions and probabilities
    model_predictions = []
    model_probs = []
    for i, logits_i in enumerate(all_model_logits):
        logits_i = logits_i.astype(np.float32)
        max_i = np.max(logits_i, axis=1, keepdims=True)
        stabilized_i = logits_i - max_i
        log_z_i = max_i + np.log(np.exp(stabilized_i).sum(axis=1, keepdims=True))
        probs_i = np.exp(logits_i - log_z_i)
        pred_i = probs_i.argmax(axis=1)
        model_probs.append(probs_i)
        model_predictions.append(pred_i)
    
    # Calculate cumulative metrics
    model_accuracies = []
    model_aucs = []
    for i in range(num_models):
        acc = _calculate_cumulative_accuracy(model_predictions[i], labels)
        auc = _calculate_cumulative_auc(model_probs[i], model_predictions[i], labels)
        model_accuracies.append(acc)
        model_aucs.append(auc)
    
    ensemble_accuracy = _calculate_cumulative_accuracy(pooled_pred, labels)
    ensemble_auc = _calculate_cumulative_auc(pooled_probs, pooled_pred, labels)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    time_steps = np.arange(1, num_examples + 1)
    
    # Plot accuracy
    for i, (acc, name) in enumerate(zip(model_accuracies, model_names)):
        ax1.plot(time_steps, acc, label=name, alpha=0.7, linewidth=1.5)
    ax1.plot(time_steps, ensemble_accuracy, label="Ensemble", linewidth=2, linestyle='--', color='black')
    ax1.set_xlabel("Number of examples seen", fontsize=11)
    ax1.set_ylabel("Cumulative Accuracy", fontsize=11)
    ax1.set_title("Accuracy Over Time", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot AUC
    for i, (auc, name) in enumerate(zip(model_aucs, model_names)):
        valid_mask = ~np.isnan(auc)
        if np.any(valid_mask):
            ax2.plot(time_steps[valid_mask], auc[valid_mask], label=name, alpha=0.7, linewidth=1.5)
    valid_ensemble_auc = ~np.isnan(ensemble_auc)
    if np.any(valid_ensemble_auc):
        ax2.plot(time_steps[valid_ensemble_auc], ensemble_auc[valid_ensemble_auc], 
                label="Ensemble", linewidth=2, linestyle='--', color='black')
    ax2.set_xlabel("Number of examples seen", fontsize=11)
    ax2.set_ylabel("Cumulative AUC", fontsize=11)
    ax2.set_title("AUC Over Time", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
