"""
Disk cache for model logits, hidden states, labels, and prompt perplexities.

Also runs batched forward passes to collect per-option log-probs and hidden states.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from wagering.core.dataset import Dataset
from wagering.core.model import WhiteboxModel
from wagering.utils.prompt_manager import (
    get_dataset_signature,
    get_mixed_context_dataset_type,
    get_model_prompt_variant,
    get_model_specific_prompts,
    requires_slot_specific_cache,
)

log = logging.getLogger("wagering")

_WAGERING_CACHE_DIR: Optional[Path] = None

PUBMEDQA_LOGITS_CACHE_NAMESPACE = "pubmedqa_v2_stable_dataset_split_seed"
PROMPT_PERPLEXITY_CACHE_NAMESPACE = "prompt_perplexity_v1"


def configure_wagering_cache_dir(cache_path: str) -> Path:
    global _WAGERING_CACHE_DIR

    root = Path(cache_path).expanduser()
    target = root / "wagering_model_logits_states_caches"
    target.mkdir(parents=True, exist_ok=True)
    _WAGERING_CACHE_DIR = target
    return _WAGERING_CACHE_DIR


def _get_model_path_key(model: WhiteboxModel) -> str:
    return model.model_path


def _wagering_logits_cache_key(
    model_key: str,
    dataset: Dataset,
    option_tokens: Sequence[str],
    prompt_variant: Optional[str],
) -> Tuple[Any, ...]:
    dataset_key = get_dataset_signature(dataset)
    option_key = tuple(option_tokens)
    pv = prompt_variant or "default"

    if get_mixed_context_dataset_type(dataset) == "pubmedqa":
        return (model_key, dataset_key, option_key, pv, PUBMEDQA_LOGITS_CACHE_NAMESPACE)
    return (model_key, dataset_key, option_key, pv)


def _wagering_perplexity_cache_key(
    model_key: str,
    dataset: Dataset,
    prompt_variant: Optional[str],
) -> Tuple[Any, ...]:
    dataset_key = get_dataset_signature(dataset)
    pv = prompt_variant or "default"
    if get_mixed_context_dataset_type(dataset) == "pubmedqa":
        return (
            model_key,
            dataset_key,
            pv,
            PROMPT_PERPLEXITY_CACHE_NAMESPACE,
            PUBMEDQA_LOGITS_CACHE_NAMESPACE,
        )
    return (model_key, dataset_key, pv, PROMPT_PERPLEXITY_CACHE_NAMESPACE)


def _load_cached_hidden_states(data: Any) -> Optional[np.ndarray]:
    if "hidden_states_by_layer_pickle" in data:
        raise RuntimeError(
            "Cached hidden states use deprecated per-layer format (hidden_states_by_layer_pickle). "
            "Delete cache files and recollect."
        )
    if "hidden_states_pickle" in data:
        raise RuntimeError(
            "Cached hidden states use deprecated pickle format (hidden_states_pickle). "
            "Delete cache files and recollect."
        )
    if "requested_hidden_state_layers" in data:
        raise RuntimeError(
            "Cached hidden states use deprecated layer metadata (requested_hidden_state_layers). "
            "Delete cache files and recollect."
        )
    if "hidden_states" not in data:
        return None
    hidden_states = np.asarray(data["hidden_states"], dtype=np.float32)
    if hidden_states.ndim != 2:
        raise RuntimeError(
            f"Cached hidden_states must be [num_examples, hidden_dim] (ndim=2), "
            f"got shape {hidden_states.shape}. Delete cache files and recollect."
        )
    return hidden_states


def _cache_key_hash(cache_key: Tuple) -> str:
    key_str = json.dumps(cache_key, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def _slug_component(value: str, *, max_len: int = 64) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))
    slug = slug.strip("_") or "unknown"
    return slug[:max_len]


def _model_slug_from_cache_key(model_key: str) -> str:
    path = model_key
    slot_suffix = ""
    if "::idx=" in model_key:
        path, slot = model_key.rsplit("::idx=", 1)
        slot_suffix = f"_slot{slot}"
    basename = path.rsplit("/", 1)[-1]
    return _slug_component(basename) + slot_suffix


def _dataset_slug(dataset: Dataset) -> str:
    name = getattr(dataset, "cache_dataset_name", None)
    if not name:
        raise ValueError("Dataset missing cache_dataset_name")
    return _slug_component(str(name))


def _prompt_variant_from_cache_key(cache_key: Tuple) -> str:
    if PROMPT_PERPLEXITY_CACHE_NAMESPACE in cache_key:
        return str(cache_key[2])
    return str(cache_key[3])


def _cache_filename(cache_key: Tuple, dataset: Dataset, key_hash: str) -> str:
    kind = "ppl" if PROMPT_PERPLEXITY_CACHE_NAMESPACE in cache_key else "logits"
    model_slug = _model_slug_from_cache_key(str(cache_key[0]))
    dataset_slug = _dataset_slug(dataset)
    variant_slug = _slug_component(_prompt_variant_from_cache_key(cache_key))
    return f"{kind}__{model_slug}__{dataset_slug}__{variant_slug}__{key_hash}.npz"


def _get_cache_path(cache_key: Tuple, *, dataset: Dataset) -> Path:
    if _WAGERING_CACHE_DIR is None:
        raise RuntimeError("configure_wagering_cache_dir must be called before using the cache")
    key_hash = _cache_key_hash(cache_key)
    return _WAGERING_CACHE_DIR / _cache_filename(cache_key, dataset, key_hash)


@dataclass
class CachedModelArtifacts:
    logits: Optional[np.ndarray] = None
    hidden_states: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None


def _resolve_model_cache_key(
    model_path: str,
    dataset: Dataset,
    model_index: Optional[int],
) -> str:
    model_key = model_path
    if requires_slot_specific_cache(dataset):
        if model_index is None:
            raise ValueError(
                "Mixed-context cache requires model_index to disambiguate repeated model paths"
            )
        model_key = f"{model_path}::idx={int(model_index)}"
    return model_key


def _resolve_model_cache_path(
    model_path: str,
    dataset: Dataset,
    option_tokens: Sequence[str],
    prompt_variant: Optional[str],
    model_index: Optional[int],
) -> Path:
    model_key = _resolve_model_cache_key(model_path, dataset, model_index)
    cache_key = _wagering_logits_cache_key(model_key, dataset, option_tokens, prompt_variant)
    return _get_cache_path(cache_key, dataset=dataset)


def _resolve_perplexity_cache_path(
    model_path: str,
    dataset: Dataset,
    prompt_variant: Optional[str],
    model_index: Optional[int],
) -> Path:
    model_key = _resolve_model_cache_key(model_path, dataset, model_index)
    cache_key = _wagering_perplexity_cache_key(model_key, dataset, prompt_variant)
    return _get_cache_path(cache_key, dataset=dataset)


def _source_row_map(dataset: Dataset) -> Tuple[Optional[np.ndarray], int]:
    row_map = getattr(dataset, "cache_source_row_indices", None)
    n_src = int(getattr(dataset, "cache_source_num_examples", 0) or 0)
    if row_map is None or n_src <= 0:
        return None, 0
    return np.asarray(row_map, dtype=np.int64), n_src


def _slice_source_cache_to_view(
    arr: np.ndarray,
    row_map: np.ndarray,
    n_src: int,
    view_len: int,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[0] == view_len:
        return arr
    if arr.shape[0] != n_src:
        raise ValueError(
            f"Cache array has {arr.shape[0]} rows; expected source size {n_src} or view {view_len}"
        )
    if arr.ndim > 1:
        return arr[row_map, ...]
    return arr[row_map]


def _scatter_view_to_source_cache(
    view_arr: np.ndarray,
    row_map: np.ndarray,
    n_src: int,
    existing: Optional[np.ndarray],
    dtype: np.dtype,
) -> np.ndarray:
    view_arr = np.asarray(view_arr)
    if view_arr.shape[0] != len(row_map):
        return view_arr
    if view_arr.ndim == 1:
        full = np.zeros(n_src, dtype=dtype)
        if existing is not None and np.asarray(existing).shape[0] == n_src:
            full = np.asarray(existing, dtype=dtype).copy()
        full[row_map] = view_arr.astype(dtype, copy=False)
        return full
    if view_arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D view array, got shape {view_arr.shape}")
    _, width = view_arr.shape
    full = np.zeros((n_src, width), dtype=dtype)
    if existing is not None:
        ex = np.asarray(existing)
        if ex.shape[0] == n_src and ex.ndim == 2:
            full = ex.astype(dtype, copy=True)
    full[row_map, :] = view_arr.astype(dtype, copy=False)
    return full


def _read_artifact_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    arrays: Dict[str, np.ndarray] = {}
    if "logits" in data:
        arrays["logits"] = np.asarray(data["logits"], dtype=np.float32)
    if "labels" in data:
        arrays["labels"] = np.asarray(data["labels"], dtype=np.int32)
    hidden_states = _load_cached_hidden_states(data)
    if hidden_states is not None:
        arrays["hidden_states"] = hidden_states
    return arrays


def _write_artifact_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    save_dict: Dict[str, np.ndarray] = {}
    if "logits" in arrays and arrays["logits"] is not None:
        save_dict["logits"] = np.asarray(arrays["logits"], dtype=np.float32)
    if "labels" in arrays and arrays["labels"] is not None:
        save_dict["labels"] = np.asarray(arrays["labels"], dtype=np.int32)
    if "hidden_states" in arrays and arrays["hidden_states"] is not None:
        hs = np.asarray(arrays["hidden_states"], dtype=np.float32)
        if hs.ndim != 2:
            raise ValueError(
                f"hidden_states must be [num_examples, hidden_dim] (ndim=2), got shape {hs.shape}"
            )
        save_dict["hidden_states"] = hs
    np.savez_compressed(path, **save_dict)


def _apply_view_slice_to_artifacts(
    artifacts: CachedModelArtifacts,
    dataset: Dataset,
) -> CachedModelArtifacts:
    row_map, n_src = _source_row_map(dataset)
    if row_map is None:
        return artifacts
    view_len = len(dataset.x)
    logits = artifacts.logits
    if logits is not None:
        logits = _slice_source_cache_to_view(
            np.asarray(logits, dtype=np.float32), row_map, n_src, view_len
        )
    labels = artifacts.labels
    if labels is not None:
        labels = _slice_source_cache_to_view(np.asarray(labels), row_map, n_src, view_len)
    hidden_states = artifacts.hidden_states
    if hidden_states is not None:
        hidden_states = _slice_source_cache_to_view(
            np.asarray(hidden_states, dtype=np.float32), row_map, n_src, view_len
        )
    return CachedModelArtifacts(logits=logits, hidden_states=hidden_states, labels=labels)


def _apply_view_scatter_for_save(
    artifacts: CachedModelArtifacts,
    dataset: Dataset,
    existing: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    row_map, n_src = _source_row_map(dataset)
    out = dict(existing)
    if row_map is None:
        if artifacts.logits is not None:
            out["logits"] = np.asarray(artifacts.logits, dtype=np.float32)
        if artifacts.labels is not None:
            out["labels"] = np.asarray(artifacts.labels, dtype=np.int32)
        if artifacts.hidden_states is not None:
            out["hidden_states"] = np.asarray(artifacts.hidden_states, dtype=np.float32)
        return out
    if artifacts.logits is not None:
        out["logits"] = _scatter_view_to_source_cache(
            artifacts.logits, row_map, n_src, existing.get("logits"), np.float32
        )
    if artifacts.labels is not None:
        out["labels"] = _scatter_view_to_source_cache(
            artifacts.labels, row_map, n_src, existing.get("labels"), np.int32
        )
    if artifacts.hidden_states is not None:
        mat = np.asarray(artifacts.hidden_states, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(
                f"hidden_states must be [num_examples, hidden_dim] (ndim=2), got shape {mat.shape}"
            )
        out["hidden_states"] = _scatter_view_to_source_cache(
            mat, row_map, n_src, existing.get("hidden_states"), np.float32
        )
    return out


def load_cached_model_artifacts(
    model_path: str,
    dataset: Dataset,
    option_tokens: List[str],
    prompt_variant: Optional[str] = None,
    model_index: Optional[int] = None,
) -> Optional[CachedModelArtifacts]:
    cache_path = _resolve_model_cache_path(
        model_path, dataset, option_tokens, prompt_variant, model_index
    )
    if not cache_path.exists():
        return None
    raw = _read_artifact_npz(cache_path)
    if not raw or "logits" not in raw:
        return None
    artifacts = CachedModelArtifacts(
        logits=raw.get("logits"),
        hidden_states=raw.get("hidden_states"),
        labels=raw.get("labels"),
    )
    artifacts = _apply_view_slice_to_artifacts(artifacts, dataset)
    model_key = _resolve_model_cache_key(model_path, dataset, model_index)
    log.debug(
        "Cache hit for model %s and dataset size %d (prompt_variant=%s)",
        model_key,
        len(dataset.x),
        prompt_variant or "default",
    )
    return artifacts


def save_cached_model_artifacts(
    model: WhiteboxModel,
    dataset: Dataset,
    option_tokens: List[str],
    artifacts: CachedModelArtifacts,
    prompt_variant: Optional[str] = None,
    model_index: Optional[int] = None,
) -> None:
    model_path = _get_model_path_key(model)
    cache_path = _resolve_model_cache_path(
        model_path, dataset, option_tokens, prompt_variant, model_index
    )
    existing: Dict[str, np.ndarray] = {}
    if cache_path.exists():
        existing = _read_artifact_npz(cache_path)
    merged = _apply_view_scatter_for_save(artifacts, dataset, existing)
    if artifacts.logits is None and "logits" in existing:
        merged.setdefault("logits", existing["logits"])
    if artifacts.hidden_states is None and "hidden_states" in existing:
        merged.setdefault("hidden_states", existing["hidden_states"])
    if artifacts.labels is None and "labels" in existing:
        merged.setdefault("labels", existing["labels"])
    _write_artifact_npz(cache_path, merged)
    model_key = _resolve_model_cache_key(model_path, dataset, model_index)
    items = [k for k in ("logits", "hidden_states", "labels") if getattr(artifacts, k) is not None]
    log.info(
        "Cached %s for model %s and dataset size %d (prompt_variant=%s) to %s",
        ", ".join(items) if items else "data",
        model_key,
        len(dataset.x),
        prompt_variant or "default",
        cache_path,
    )


def load_cached_prompt_perplexities(
    model_path: str,
    dataset: Dataset,
    prompt_variant: Optional[str] = None,
    model_index: Optional[int] = None,
) -> Optional[np.ndarray]:
    cache_path = _resolve_perplexity_cache_path(
        model_path, dataset, prompt_variant, model_index
    )
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    if "perplexities" not in data:
        return None
    arr = np.asarray(data["perplexities"], dtype=np.float32)
    row_map, n_src = _source_row_map(dataset)
    if row_map is not None:
        arr = _slice_source_cache_to_view(arr, row_map, n_src, len(dataset.x))
    return arr


def save_cached_prompt_perplexities(
    model_path: str,
    dataset: Dataset,
    perplexities: np.ndarray,
    prompt_variant: Optional[str] = None,
    model_index: Optional[int] = None,
) -> None:
    cache_path = _resolve_perplexity_cache_path(
        model_path, dataset, prompt_variant, model_index
    )
    view_ppl = np.asarray(perplexities, dtype=np.float32)
    existing: Dict[str, np.ndarray] = {}
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if "perplexities" in data:
            existing["perplexities"] = np.asarray(data["perplexities"], dtype=np.float32)
    row_map, n_src = _source_row_map(dataset)
    if row_map is not None:
        view_ppl = _scatter_view_to_source_cache(
            view_ppl, row_map, n_src, existing.get("perplexities"), np.float32
        )
    np.savez_compressed(cache_path, perplexities=view_ppl)
    model_key = _resolve_model_cache_key(model_path, dataset, model_index)
    log.info(
        "Cached prompt perplexities for model %s and dataset size %d (prompt_variant=%s) to %s",
        model_key,
        len(dataset.x),
        prompt_variant or "default",
        cache_path,
    )


def compute_prompt_perplexities_for_model(
    model: WhiteboxModel,
    prompts: List[str],
    batch_size: int,
) -> np.ndarray:
    if len(prompts) == 0:
        return np.empty((0,), dtype=np.float32)

    model_device = model.device()
    ppl_batches: List[np.ndarray] = []
    pad_token_id = getattr(model.tokenizer, "pad_token_id", None)

    with torch.inference_mode():
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            batch = model.tokenize(batch_prompts)
            input_ids = batch["input_ids"].to(model_device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )
            logits = outputs.logits

            if logits.size(1) < 2:
                ppl_batches.append(np.ones((input_ids.size(0),), dtype=np.float32))
                continue

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            token_log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_nll = -torch.gather(
                token_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            if attention_mask is not None:
                token_mask = attention_mask[:, 1:].to(dtype=token_nll.dtype)
            else:
                token_mask = torch.ones_like(token_nll, dtype=token_nll.dtype)

            if pad_token_id is not None:
                token_mask = token_mask * (shift_labels != pad_token_id).to(
                    dtype=token_nll.dtype
                )

            token_count = torch.clamp(token_mask.sum(dim=1), min=1.0)
            mean_nll = (token_nll * token_mask).sum(dim=1) / token_count
            perplexity = torch.exp(mean_nll)
            ppl_batches.append(perplexity.detach().to(dtype=torch.float32).cpu().numpy())

    return np.concatenate(ppl_batches, axis=0).astype(np.float32, copy=False)


def should_use_sequential_perplexity_load(
    model_configs_for_sequential: Optional[List[Dict[str, Any]]],
    num_models: int,
) -> bool:
    if model_configs_for_sequential is None:
        return False
    if len(model_configs_for_sequential) != num_models:
        return False
    from wagering.utils.model_utils import should_load_prompt_perplexity_models_sequentially

    return should_load_prompt_perplexity_models_sequentially(num_models)


def _model_path_at_slot(
    models: Sequence[Union[WhiteboxModel, str]],
    model_index: int,
    model_configs: Optional[List[Dict[str, Any]]],
) -> str:
    model = models[model_index]
    if isinstance(model, WhiteboxModel):
        return str(model.model_path)
    if isinstance(model, str):
        return model
    if model_configs is not None and model_index < len(model_configs):
        return str(model_configs[model_index]["path"])
    raise ValueError(f"Cannot resolve model path for ensemble slot {model_index}")


def _model_config_group_key(model_cfg: Dict[str, Any]) -> str:
    try:
        return json.dumps(model_cfg, sort_keys=True, default=str)
    except TypeError:
        return repr(model_cfg)


def free_whitebox_model(model: WhiteboxModel) -> None:
    import gc

    if getattr(model, "model", None) is not None:
        del model.model
    if getattr(model, "tokenizer", None) is not None:
        del model.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_ensemble_whitebox_models(
    models: Sequence[Union[WhiteboxModel, str]],
) -> List[Union[WhiteboxModel, str]]:
    """Replace loaded ``WhiteboxModel`` slots with path strings and free GPU memory."""
    new_models: List[Union[WhiteboxModel, str]] = []
    seen_ids: set = set()
    to_free: List[WhiteboxModel] = []

    for m in models:
        if isinstance(m, WhiteboxModel):
            mp = getattr(m, "model_path", None) or ""
            new_models.append(str(mp) if mp else str(id(m)))
            mid = id(m)
            if mid not in seen_ids:
                seen_ids.add(mid)
                to_free.append(m)
        else:
            new_models.append(m)

    for wb in to_free:
        free_whitebox_model(wb)

    return new_models


def _perplexity_column_from_model(
    model: WhiteboxModel,
    model_path: str,
    dataset: Dataset,
    model_index: int,
    batch_size: int,
) -> np.ndarray:
    prompt_variant = get_model_prompt_variant(dataset, model_index=model_index)
    cached = load_cached_prompt_perplexities(
        model_path, dataset, prompt_variant=prompt_variant, model_index=model_index
    )
    if cached is not None:
        return cached

    model_prompts = get_model_specific_prompts(dataset, model_index=model_index)
    num_examples = len(dataset.x)
    if len(model_prompts) != num_examples:
        raise ValueError(
            "Prompt length mismatch while computing prompt perplexities. "
            f"prompts={len(model_prompts)}, examples={num_examples}"
        )

    perplexities = compute_prompt_perplexities_for_model(
        model, model_prompts, batch_size=batch_size
    )
    save_cached_prompt_perplexities(
        model_path,
        dataset,
        perplexities,
        prompt_variant=prompt_variant,
        model_index=model_index,
    )
    return perplexities


def compute_all_prompt_perplexities(
    models: Sequence[Union[WhiteboxModel, str]],
    dataset: Dataset,
    *,
    model_configs_for_sequential: Optional[List[Dict[str, Any]]] = None,
    load_cache_kwargs: Optional[Dict[str, Any]] = None,
    group_identical_model_configs: bool = False,
) -> np.ndarray:
    import gc

    from wagering.utils.model_utils import load_models_from_config

    num_examples = len(dataset.x)
    num_models = len(models)
    if num_models == 0:
        return np.empty((num_examples, 0), dtype=np.float32)

    all_perplexities = np.empty((num_examples, num_models), dtype=np.float32)
    batch_size = max(1, int(dataset.batch_size))
    load_cache_kwargs = load_cache_kwargs or {}

    if should_use_sequential_perplexity_load(model_configs_for_sequential, num_models):
        cfgs = model_configs_for_sequential
        if cfgs is None or len(cfgs) != num_models:
            raise RuntimeError(
                "Sequential perplexity requires model_configs matching ensemble size"
            )

        log.info(
            "Computing prompt perplexities sequentially (%d models; %d visible CUDA device(s))",
            num_models,
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        )

        if group_identical_model_configs:
            key_to_indices: Dict[str, List[int]] = {}
            for model_index, cfg in enumerate(cfgs):
                key_to_indices.setdefault(_model_config_group_key(cfg), []).append(
                    model_index
                )
            load_groups = list(key_to_indices.values())
        else:
            load_groups = [[i] for i in range(num_models)]

        for model_indices in load_groups:
            loaded, _ = load_models_from_config(
                [cfgs[model_indices[0]]],
                cache_kwargs=load_cache_kwargs,
                share_identical_models=group_identical_model_configs,
            )
            wb = loaded[0]
            try:
                for model_index in model_indices:
                    model_path = _model_path_at_slot(models, model_index, cfgs)
                    all_perplexities[:, model_index] = _perplexity_column_from_model(
                        wb, model_path, dataset, model_index, batch_size
                    )
            finally:
                free_whitebox_model(wb)
                del loaded, wb
                gc.collect()

        return all_perplexities

    for model_index, model in enumerate(models):
        if isinstance(model, str):
            raise RuntimeError(
                "Prompt-perplexity wagering requires loaded model objects, "
                f"but model at index {model_index} is a string path: {model}. "
                "Pass model_configs_for_sequential when there are more models than visible GPUs."
            )
        model_path = _model_path_at_slot(models, model_index, model_configs_for_sequential)
        all_perplexities[:, model_index] = _perplexity_column_from_model(
            model, model_path, dataset, model_index, batch_size
        )

    return all_perplexities


def resolve_label_to_index(label: object, option_tokens: List[str]) -> int:
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

def _resolve_option_token_ids(
    model: WhiteboxModel,
    option_tokens: List[str],
    sample_prompt: str,
) -> List[int]:
    token_ids: List[int] = []
    # Determine prompt suffix
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
                log.info(
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
            log.info(f"Using standalone tokenization for '{opt}' as fallback.")
    log.info(
        f"Resolved option token IDs for {getattr(model.tokenizer, 'name_or_path', 'unknown')}: "
        f"{dict(zip(option_tokens, token_ids))}"
    )
    return token_ids


def _hidden_states_last_token(out: Any) -> torch.Tensor:
    hidden_states = out.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model forward did not return hidden_states.")
    layer_hidden = hidden_states[-1]
    if layer_hidden.dim() == 3:
        return layer_hidden[:, -1, :]
    if layer_hidden.dim() == 2:
        return layer_hidden
    raise ValueError(f"Unexpected hidden state shape: {layer_hidden.shape}")


def get_mixed_context_assignments(
    dataset: Dataset,
    *,
    error_message: str,
) -> Optional[np.ndarray]:
    """Per-example context model indices for mixed-context datasets, else None."""
    dataset_type = get_mixed_context_dataset_type(dataset)
    if dataset_type is None:
        return None
    raw = getattr(dataset, f"{dataset_type}_context_assignment_by_example", None)
    if not isinstance(raw, list) or len(raw) != len(dataset.x):
        raise RuntimeError(error_message)
    return np.asarray(raw, dtype=np.int64)


def combine_hidden_states_per_model(
    hidden_states_per_model: Optional[List[np.ndarray]],
) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
    if not hidden_states_per_model:
        return None
    hidden_dims = [hs.shape[-1] for hs in hidden_states_per_model]
    if len(set(hidden_dims)) == 1:
        return np.stack(hidden_states_per_model, axis=0)
    return hidden_states_per_model


@dataclass
class StackedModelArtifacts:
    logits: np.ndarray
    hidden_states_per_model: Optional[List[np.ndarray]] = None
    labels: Optional[np.ndarray] = None
    context_assignments: Optional[np.ndarray] = None

    def combined_hidden_states(self) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        return combine_hidden_states_per_model(self.hidden_states_per_model)


def collect_stacked_model_artifacts(
    models: Sequence[Union[WhiteboxModel, str]],
    dataset: Dataset,
    option_tokens: Sequence[str],
    *,
    collect_hidden_states: bool = False,
    require_labels_consistency: bool = False,
    mixed_context_error_message: str = (
        "Mixed-context dataset missing per-example context assignments."
    ),
    resolve_model_on_cache_miss: Optional[
        Callable[[int, Union[WhiteboxModel, str]], WhiteboxModel]
    ] = None,
    release_model_after_collect: Optional[Callable[[int, WhiteboxModel], None]] = None,
) -> StackedModelArtifacts:
    """Load or collect per-model logits (and optionally hidden states), then stack logits."""
    option_tokens_list = list(option_tokens)
    num_models = len(models)
    context_assignments = get_mixed_context_assignments(
        dataset, error_message=mixed_context_error_message
    )

    logits_list: List[np.ndarray] = []
    hidden_states_list: Optional[List[np.ndarray]] = [] if collect_hidden_states else None
    labels: Optional[np.ndarray] = None

    for model_idx, model in enumerate(models):
        model_path = model if isinstance(model, str) else model.model_path
        prompt_variant = get_model_prompt_variant(dataset, model_index=model_idx)
        cached = load_cached_model_artifacts(
            model_path,
            dataset,
            option_tokens_list,
            prompt_variant=prompt_variant,
            model_index=model_idx,
        )

        cache_sufficient = cached is not None and cached.logits is not None and (
            (not collect_hidden_states) or cached.hidden_states is not None
        )
        if cache_sufficient:
            model_logits = cached.logits
            model_hidden_states = cached.hidden_states if collect_hidden_states else None
            model_labels = cached.labels
        else:
            if isinstance(model, str):
                if resolve_model_on_cache_miss is None:
                    raise RuntimeError(
                        f"Cache miss for model path {model}. "
                        "A loaded model instance is required to collect caches."
                    )
                wb_model = resolve_model_on_cache_miss(model_idx, model)
            else:
                wb_model = model
            log.info(
                "Model %d/%d: cache miss - collecting logits%s (device: %s)",
                model_idx + 1,
                num_models,
                " and hidden states" if collect_hidden_states else "",
                wb_model.device(),
            )
            model_prompts = get_model_specific_prompts(dataset, model_index=model_idx)
            model_logits, model_hidden_states, model_labels = (
                collect_option_logits_and_hidden_states_for_model(
                    wb_model,
                    dataset,
                    option_tokens_list,
                    model_identifier=str(model_path),
                    model_index=model_idx,
                    collect_hidden_states=collect_hidden_states,
                    model_prompts=model_prompts,
                    prompt_variant=prompt_variant,
                )
            )
            save_cached_model_artifacts(
                wb_model,
                dataset,
                option_tokens_list,
                CachedModelArtifacts(
                    logits=model_logits,
                    hidden_states=model_hidden_states,
                    labels=model_labels,
                ),
                prompt_variant=prompt_variant,
                model_index=model_idx,
            )
            if release_model_after_collect is not None:
                release_model_after_collect(model_idx, wb_model)

        logits_list.append(np.asarray(model_logits, dtype=np.float32))
        if collect_hidden_states and hidden_states_list is not None:
            if model_hidden_states is None:
                raise RuntimeError(
                    f"Hidden states required but missing for model {model_idx + 1}"
                )
            hidden_states_list.append(np.asarray(model_hidden_states, dtype=np.float32))

        if model_labels is not None:
            model_labels = np.asarray(model_labels, dtype=np.int64)
            if labels is None:
                labels = model_labels
            elif require_labels_consistency and not np.array_equal(labels, model_labels):
                raise RuntimeError(
                    "Labels must match across models for the same dataset"
                )

    return StackedModelArtifacts(
        logits=np.stack(logits_list, axis=0),
        hidden_states_per_model=hidden_states_list,
        labels=labels,
        context_assignments=context_assignments,
    )


def get_cached_logits_and_hidden_states_for_model(
    model_path: str,
    dataset: Dataset,
    option_tokens: List[str],
    prompt_variant: Optional[str] = None,
    model_index: Optional[int] = None,
) -> Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    artifacts = load_cached_model_artifacts(
        model_path, dataset, option_tokens, prompt_variant, model_index
    )
    if artifacts is None:
        return None, None, None
    return artifacts.logits, artifacts.hidden_states, artifacts.labels


def set_cached_logits_and_hidden_states_for_model(
    model: WhiteboxModel,
    dataset: Dataset,
    option_tokens: List[str],
    logits: Optional[np.ndarray],
    hidden_states: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    prompt_variant: Optional[str] = None,
    model_index: Optional[int] = None,
) -> None:
    save_cached_model_artifacts(
        model,
        dataset,
        option_tokens,
        CachedModelArtifacts(logits=logits, hidden_states=hidden_states, labels=labels),
        prompt_variant=prompt_variant,
        model_index=model_index,
    )


def collect_option_logits_and_hidden_states_for_model(
    model: WhiteboxModel,
    dataset: Dataset,
    option_tokens: List[str],
    max_new_tokens: int = 1,
    model_identifier: Optional[str] = None,
    model_index: int = 0,
    collect_hidden_states: bool = True,
    model_prompts: Optional[List[str]] = None,
    prompt_variant: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    del max_new_tokens
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
    if len(model_prompts) == 0:
        raise ValueError("Dataset is empty (0 examples).")

    if prompt_variant is not None:
        log.info("Using prompt_variant='%s' for model %s", prompt_variant, model_path)

    if torch.cuda.is_available() and getattr(model_device, "type", None) == "cuda":
        torch.cuda.set_device(model_device)

    sample_prompt = model_prompts[0]
    option_token_ids = _resolve_option_token_ids(model, option_tokens, sample_prompt=sample_prompt)

    all_log_probs: List[np.ndarray] = []
    all_hidden_states: List[np.ndarray] = []
    all_labels: List[int] = []

    with torch.inference_mode():
        for batch_start in range(0, len(model_prompts), dataset.batch_size):
            batch_end = min(batch_start + dataset.batch_size, len(model_prompts))
            batch_x = model_prompts[batch_start:batch_end]
            batch_y = dataset.y[batch_start:batch_end]

            batch = model.tokenize(batch_x)
            batch = {k: v.to(model_device) for k, v in batch.items()}

            out = model(
                **batch,
                output_hidden_states=collect_hidden_states,
            )
            step_log_probs = torch.log_softmax(out.logits[:, -1, :], dim=-1)
            batch_log_probs = step_log_probs[:, option_token_ids]

            batch_hidden_states = None
            if collect_hidden_states:
                batch_hidden_states = _hidden_states_last_token(out)

            if len(all_log_probs) == 0:
                log.info(
                    "Extracting logits for option tokens: %s",
                    dict(zip(option_tokens, option_token_ids)),
                )
                token_names = {
                    opt: model.tokenizer.convert_ids_to_tokens([tid])[0]
                    for opt, tid in zip(option_tokens, option_token_ids)
                }
                log.info("Token names: %s", token_names)

            all_log_probs.append(batch_log_probs.detach().to(dtype=torch.float32).cpu().numpy())
            if collect_hidden_states and batch_hidden_states is not None:
                all_hidden_states.append(
                    batch_hidden_states.detach().to(dtype=torch.float32).cpu().numpy()
                )

            for y in batch_y:
                all_labels.append(resolve_label_to_index(y, option_tokens))

    logits = np.concatenate(all_log_probs, axis=0)
    hidden_states = np.concatenate(all_hidden_states, axis=0) if collect_hidden_states else None
    labels = np.asarray(all_labels, dtype=np.int32)
    return logits, hidden_states, labels
