"""Prepare model objects or paths for training/evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from wagering.core.dataset import Dataset
from wagering.utils import load_models_from_config
from wagering.utils.cache_manager import (
    get_cached_logits_and_hidden_states_for_model,
    load_cached_prompt_perplexities,
)
from wagering.utils.model_utils import should_load_prompt_perplexity_models_sequentially
from wagering.utils.prompt_manager import get_model_prompt_variant

log = logging.getLogger("wagering")


@dataclass(frozen=True)
class CacheMiss:
    phase: str
    dataset_name: str
    model_index: int
    model_path: str
    artifact: str


def find_model_cache_misses(
    model_cfgs: Sequence[Dict[str, Any]],
    datasets: Sequence[Dataset],
    option_tokens: Sequence[str],
    *,
    phase: str,
    needs_hidden_states: bool,
    needs_perplexities: bool = False,
    model_indices: Optional[Sequence[int]] = None,
) -> List[CacheMiss]:
    misses: List[CacheMiss] = []
    if model_indices is not None and len(model_indices) != len(model_cfgs):
        raise ValueError(
            f"model_indices length ({len(model_indices)}) must match model_cfgs ({len(model_cfgs)})"
        )
    for enum_index, model_cfg in enumerate(model_cfgs):
        model_index = int(model_indices[enum_index]) if model_indices is not None else enum_index
        model_path = model_cfg["path"]
        for dataset in datasets:
            dataset_name = str(getattr(dataset, "cache_dataset_name", phase))
            prompt_variant = get_model_prompt_variant(dataset, model_index=model_index)
            cached_logits, cached_hidden_states, _ = get_cached_logits_and_hidden_states_for_model(
                model_path,
                dataset,
                list(option_tokens),
                prompt_variant=prompt_variant,
                model_index=model_index,
            )
            if cached_logits is None:
                misses.append(
                    CacheMiss(phase, dataset_name, model_index, model_path, "logits")
                )
            elif needs_hidden_states and cached_hidden_states is None:
                misses.append(
                    CacheMiss(phase, dataset_name, model_index, model_path, "hidden_states")
                )
            if needs_perplexities and load_cached_prompt_perplexities(
                model_path,
                dataset,
                prompt_variant=prompt_variant,
                model_index=model_index,
            ) is None:
                misses.append(
                    CacheMiss(phase, dataset_name, model_index, model_path, "perplexities")
                )
    return misses


def require_pipeline_caches_for_repeat(
    args: Dict[str, Any],
    *,
    skip_training: bool = False,
    skip_evaluation: bool = False,
) -> None:
    """Raise if train/eval forward-pass caches are incomplete before parallel repeats."""
    from wagering.calibration import calibration_enabled
    from wagering.methods.factory import load_wagering_method
    from wagering.utils import load_dataset_from_config, load_datasets_from_config
    from wagering.utils.cache_manager import configure_wagering_cache_dir
    from wagering.utils.prompt_manager import assign_pubmedqa_context_models

    configure_wagering_cache_dir(args["cache_path"])

    wagering_config = args["wagering_method"]
    model_cfgs = args["models"]
    num_models = len(model_cfgs)
    wagering_method = load_wagering_method(
        wagering_config["name"],
        num_models=num_models,
        config=wagering_config.get("config", {}),
    )
    needs_hidden_states = bool(getattr(wagering_method, "requires_hidden_states", True)) or (
        calibration_enabled(args)
    )
    needs_perplexities = bool(getattr(wagering_method, "requires_model_perplexities", False))
    option_tokens = args.get("option_tokens", ["A", "B", "C", "D"])
    model_paths = [model_cfg["path"] for model_cfg in model_cfgs]
    dataset_split_seed = int(args.get("dataset_split_seed", 42))

    misses: List[CacheMiss] = []
    requires_training = len(wagering_method.get_trainable_parameters()) > 0 and bool(
        args.get("dataset")
    )

    if not skip_training and requires_training:
        test_peer = [args["test_dataset"]] if args.get("test_dataset") else None
        train_dataset, _ = load_dataset_from_config(
            args["dataset"],
            split="train",
            random_seed=dataset_split_seed,
            partition_peer_dataset_configs=test_peer,
        )
        assign_pubmedqa_context_models(
            [train_dataset], model_paths, random_seed=dataset_split_seed
        )
        misses.extend(
            find_model_cache_misses(
                model_cfgs,
                [train_dataset],
                option_tokens,
                phase="train",
                needs_hidden_states=needs_hidden_states,
                needs_perplexities=needs_perplexities,
            )
        )

    if not skip_evaluation:
        tr_peer = [args["dataset"]] if args.get("dataset") else None
        eval_datasets: List[Dataset] = []
        if "test_dataset" in args:
            test_ds, _ = load_dataset_from_config(
                args["test_dataset"],
                split="test",
                random_seed=dataset_split_seed,
                partition_peer_dataset_configs=tr_peer,
                infer_eval_split_train_without_peer=False,
                force_partition=True,
            )
            eval_datasets.append(test_ds)
        if args.get("ood_datasets"):
            ood_ds, _ = load_datasets_from_config(
                args["ood_datasets"],
                split="test",
                random_seed=dataset_split_seed,
                partition_peer_dataset_configs=tr_peer,
                infer_eval_split_train_without_peer=False,
            )
            eval_datasets.extend(ood_ds)
        if eval_datasets:
            assign_pubmedqa_context_models(
                eval_datasets, model_paths, random_seed=dataset_split_seed
            )
            misses.extend(
                find_model_cache_misses(
                    model_cfgs,
                    eval_datasets,
                    option_tokens,
                    phase="eval",
                    needs_hidden_states=needs_hidden_states,
                    needs_perplexities=needs_perplexities,
                )
            )

    if not misses:
        return

    lines = [
        "Missing on-disk model caches required for parallel repeat runs.",
        "Run scripts/wagering_pipeline.py once (single GPU) to warm the cache first:",
        "",
        "  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/wagering_pipeline.py <config.yaml>",
        "",
        "Missing entries:",
    ]
    for miss in misses:
        lines.append(
            f"  [{miss.phase}] {miss.dataset_name} "
            f"model[{miss.model_index}] {miss.model_path}: {miss.artifact}"
        )
    raise RuntimeError("\n".join(lines))


def _model_cache_ok(
    model_cfg: Dict[str, Any],
    model_index: int,
    datasets: Sequence[Dataset],
    option_tokens: Sequence[str],
    needs_hidden_states: bool,
) -> bool:
    return not find_model_cache_misses(
        [model_cfg],
        datasets,
        option_tokens,
        phase="",
        needs_hidden_states=needs_hidden_states,
        model_indices=[model_index],
    )


def prepare_ensemble_for_run(
    model_cfgs: List[Dict[str, Any]],
    datasets: Sequence[Dataset],
    option_tokens: Sequence[str],
    *,
    needs_hidden_states: bool,
    force_load_all_for_perplexity: bool,
    cache_path: str,
    num_models: int,
) -> Tuple[List[Union[Any, str]], List[str]]:
    cache_miss_indices: List[int] = []
    model_names = [cfg["path"].replace("/", "_") for cfg in model_cfgs]

    for idx, model_cfg in enumerate(model_cfgs):
        if not _model_cache_ok(
            model_cfg, idx, datasets, option_tokens, needs_hidden_states
        ):
            cache_miss_indices.append(idx)

    indices_to_load = set(cache_miss_indices)
    if force_load_all_for_perplexity:
        indices_to_load = set(range(num_models))

    use_sequential_perplexity = (
        force_load_all_for_perplexity
        and should_load_prompt_perplexity_models_sequentially(num_models)
    )
    perplexity_cache_kwargs = {"cache_dir": cache_path}

    if use_sequential_perplexity:
        log.info(
            "Deferring full ensemble load for prompt perplexity (%d models on %d visible CUDA device(s)); "
            "will load one model at a time.",
            num_models,
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        )
        return [cfg["path"] for cfg in model_cfgs], model_names

    if not indices_to_load:
        log.info("All models are cached. Skipping model loading.")
        return [cfg["path"] for cfg in model_cfgs], model_names

    sorted_indices = sorted(indices_to_load)
    if force_load_all_for_perplexity:
        log.info(
            "Loading %d/%d models to compute prompt perplexities.",
            len(sorted_indices),
            num_models,
        )
    else:
        log.info(
            "Cache miss for %d/%d models. Loading missing models...",
            len(sorted_indices),
            num_models,
        )

    missing_cfgs = [model_cfgs[i] for i in sorted_indices]
    missing_models, missing_names = load_models_from_config(
        missing_cfgs,
        cache_kwargs=perplexity_cache_kwargs,
    )
    missing_name_map = {idx: name for idx, name in zip(sorted_indices, missing_names)}
    missing_iter = iter(missing_models)

    models: List[Union[Any, str]] = []
    for i in range(num_models):
        if i in indices_to_load:
            models.append(next(missing_iter))
            model_names[i] = missing_name_map.get(i, model_names[i])
        else:
            models.append(model_cfgs[i]["path"])

    log.info("Prepared %d models: %s", len(models), model_names)
    return models, model_names
