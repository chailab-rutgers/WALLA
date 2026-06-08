"""
Mixed-context prompt routing (PubMedQA) and per-model prompt selection.
"""

import hashlib
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wagering.core.dataset import Dataset

log = logging.getLogger("wagering")


def get_dataset_signature(dataset: Dataset) -> Tuple:
    """Dataset signature for cache keys and deterministic routing seeds."""
    dataset_cache_config = getattr(dataset, "cache_dataset_config", None)
    if not isinstance(dataset_cache_config, dict):
        raise ValueError(
            "Dataset missing cache_dataset_config; load via load_dataset_from_config"
        )
    signature = dataset_cache_config.get("signature")
    schema_version = dataset_cache_config.get("schema_version", 0)
    if not isinstance(signature, str) or not signature:
        raise ValueError("Dataset cache_dataset_config missing non-empty signature")
    return ("cfg", int(schema_version), signature)


def _is_pubmedqa_dataset(dataset: Dataset) -> bool:
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


def get_mixed_context_dataset_type(dataset: Dataset) -> Optional[str]:
    """Return mixed-context dataset type when model-specific routing is enabled."""
    if getattr(dataset, "pubmedqa_prompt_strategy", None) == "mixed_context" or _is_pubmedqa_dataset(
        dataset
    ):
        return "pubmedqa"
    return None


def requires_slot_specific_cache(dataset: Dataset) -> bool:
    """Whether disk cache keys must include model slot index."""
    return get_mixed_context_dataset_type(dataset) == "pubmedqa"


def _build_pubmedqa_balanced_assignments(
    num_examples: int,
    num_models: int,
    seed: int,
) -> np.ndarray:
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
        r = int(rng.randint(0, num_models - 1))
        wrong[i] = r if r < right_idx else r + 1
    return wrong.astype(np.int32, copy=False)


def _build_pubmedqa_wrong_context_example_indices(
    *,
    num_examples: int,
    seed: int,
) -> np.ndarray:
    if num_examples <= 0:
        return np.empty((0,), dtype=np.int32)
    if num_examples == 1:
        return np.zeros((1,), dtype=np.int32)

    rng = np.random.RandomState(int(seed))
    out = np.empty((num_examples,), dtype=np.int32)
    for i in range(num_examples):
        r = int(rng.randint(0, num_examples - 1))
        out[i] = r if r < i else r + 1
    return out.astype(np.int32, copy=False)


def _build_pubmedqa_wrong_context_prompts(dataset: Dataset) -> List[str]:
    questions = getattr(dataset, "pubmedqa_questions", None)
    long_answers = getattr(dataset, "pubmedqa_long_answers", None)
    contexts = getattr(dataset, "pubmedqa_context_texts", None)
    template = getattr(dataset, "pubmedqa_prompt_template_with_context", None)
    source_rows = getattr(dataset, "pubmedqa_wrong_context_source_example_by_example", None)

    if not (isinstance(questions, list) and isinstance(long_answers, list) and isinstance(contexts, list)):
        raise RuntimeError(
            "PubMedQA wrong-context routing requires dataset to expose pubmedqa_questions, "
            "pubmedqa_long_answers, and pubmedqa_context_texts."
        )
    if not (len(questions) == len(long_answers) == len(contexts)):
        raise RuntimeError("PubMedQA raw field arrays must have identical lengths")

    n = len(questions)
    if not isinstance(template, str) or not template.strip():
        raise RuntimeError("PubMedQA wrong-context routing requires pubmedqa_prompt_template_with_context")
    if not isinstance(source_rows, list) or len(source_rows) != n:
        raise RuntimeError(
            "pubmedqa_wrong_context_source_example_by_example missing or wrong length; "
            "call assign_pubmedqa_context_models first."
        )

    rendered: List[str] = []
    for i in range(n):
        src_idx = int(source_rows[i])
        if src_idx < 0 or src_idx >= n:
            raise RuntimeError("Wrong-context source index out of range")

        question = str(questions[i])
        long_answer = str(long_answers[i])
        rendered.append(
            template.format(
                question=question,
                context=str(contexts[src_idx]),
                long_answer=long_answer,
                text=question,
                answer=long_answer,
            )
        )

    return rendered


def assign_pubmedqa_context_model(
    dataset: Dataset,
    model_paths: Sequence[str],
    random_seed: Optional[int] = None,
    dataset_index: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    """
    Assign mixed-context PubMedQA prompts per-example via balanced randomized routing.

    Returns assignment metadata if the dataset uses mixed PubMedQA prompts, else None.
    """
    dataset_type = get_mixed_context_dataset_type(dataset)
    if dataset_type is None:
        return None

    paths = [str(path) for path in model_paths if path]
    if not paths:
        return None

    num_examples = len(dataset.x)
    num_models = len(paths)

    normalized_seed: Optional[int] = None if random_seed is None else int(random_seed)
    if dataset_index is not None:
        dataset.pubmedqa_context_dataset_index = int(dataset_index)

    assignment_attr = f"{dataset_type}_context_assignment_by_example"
    counts_attr = f"{dataset_type}_context_assignment_counts"
    hash_attr = f"{dataset_type}_context_assignment_hash"
    run_seed_attr = f"{dataset_type}_context_run_seed"

    dataset_signature = get_dataset_signature(dataset)
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

    wrong_context_enabled = bool(
        dataset_type == "pubmedqa" and bool(getattr(dataset, "pubmedqa_wrong_context_routing", False))
    )
    wrong_assignment_hash = None
    wrong_counts: Optional[List[int]] = None
    if wrong_context_enabled:
        wrong_seed_components = [
            f"{dataset_type}_wrong_context_model",
            str(get_dataset_signature(dataset)),
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

        example_seed_components = [
            f"{dataset_type}_wrong_context_source_row",
            str(get_dataset_signature(dataset)),
        ]
        if dataset_index is not None:
            example_seed_components.append(f"dataset_index={int(dataset_index)}")
        example_seed_input = "::".join(example_seed_components)
        example_seed = int(hashlib.md5(example_seed_input.encode("utf-8")).hexdigest()[:8], 16)
        source_indices = _build_pubmedqa_wrong_context_example_indices(
            num_examples=num_examples,
            seed=example_seed,
        )
        dataset.pubmedqa_wrong_context_source_example_by_example = source_indices.tolist()
        dataset.pubmedqa_wrong_context_source_hash = hashlib.md5(source_indices.tobytes()).hexdigest()[:12]
        dataset.pubmedqa_wrong_context_x = _build_pubmedqa_wrong_context_prompts(dataset)

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


def assign_pubmedqa_context_models(
    datasets: Sequence[Dataset],
    model_paths: Sequence[str],
    random_seed: Optional[int] = None,
) -> Dict[int, Dict[str, object]]:
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
    dataset_type = get_mixed_context_dataset_type(dataset)
    if dataset_type is None:
        return None

    assignments = getattr(dataset, f"{dataset_type}_context_assignment_by_example", None)
    if not isinstance(assignments, list) or len(assignments) != len(dataset.x):
        return None
    return np.asarray(assignments, dtype=np.int32)


def get_model_prompt_variant(
    dataset: Dataset,
    model_index: int,
) -> Optional[str]:
    """Return the prompt variant key used for disk cache lookup for this model slot."""
    dataset_type = get_mixed_context_dataset_type(dataset)
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
        if isinstance(wrong_hash, str) and wrong_hash:
            return f"balanced_random_context_wrong_m{model_index}_{assignment_hash}_{wrong_hash}"
        return f"balanced_random_context_m{model_index}_{assignment_hash}"
    raise RuntimeError(f"Unsupported mixed-context dataset type: {dataset_type}")


def get_model_specific_prompts(
    dataset: Dataset,
    model_index: int,
) -> List[str]:
    """Return prompt texts for this model, defaulting to dataset.x."""
    dataset_type = get_mixed_context_dataset_type(dataset)
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

            if dataset_type == "pubmedqa" and bool(getattr(dataset, "pubmedqa_wrong_context_routing", False)):
                wrong_prompts = getattr(dataset, "pubmedqa_wrong_context_x", None)
                if not isinstance(wrong_prompts, list) or len(wrong_prompts) != len(assignments):
                    raise RuntimeError(
                        "pubmedqa_wrong_context_routing enabled but wrong-context prompts are missing. "
                        "Ensure assign_pubmedqa_context_models ran before cache checks/collection."
                    )

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
