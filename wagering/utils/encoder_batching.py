"""Shared BERT/encoder batching for prompt-based wagering routers."""

from typing import Callable, Dict, List, Optional

import torch

from wagering.utils.prompt_embedding_utils import preprocess_pubmedqa_prompts_for_embedding


def encode_questions_per_model_batch(
    questions_per_model: List[List[str]],
    *,
    num_models: int,
    device: torch.device,
    concat_prompt_embeddings: bool,
    encode_batch: Callable[[List[str]], torch.Tensor],
    method_name: str,
) -> torch.Tensor:
    if not isinstance(questions_per_model, list) or len(questions_per_model) != num_models:
        raise ValueError(
            f"{method_name} expects questions_per_model as a list of length num_models "
            f"(expected {num_models})."
        )
    batch_size = len(questions_per_model[0]) if num_models > 0 else 0
    for mi, prompts in enumerate(questions_per_model):
        if len(prompts) != batch_size:
            raise ValueError(
                f"{method_name} questions_per_model batch mismatch: "
                f"model_index={mi}, len={len(prompts)}, expected={batch_size}"
            )

    if not concat_prompt_embeddings:
        return encode_batch(list(questions_per_model[0]))

    flat_prompts: List[str] = []
    for mi in range(num_models):
        flat_prompts.extend([str(p) for p in questions_per_model[mi]])

    unique_prompts: List[str] = []
    index_by_prompt: Dict[str, int] = {}
    for p in flat_prompts:
        if p in index_by_prompt:
            continue
        index_by_prompt[p] = len(unique_prompts)
        unique_prompts.append(p)

    unique_emb = encode_batch(unique_prompts)
    per_model_emb: List[torch.Tensor] = []
    for mi in range(num_models):
        idx = torch.as_tensor(
            [index_by_prompt[str(p)] for p in questions_per_model[mi]],
            device=device,
            dtype=torch.long,
        )
        per_model_emb.append(unique_emb.index_select(0, idx))
    return torch.cat(per_model_emb, dim=1)


def encode_transformer_batch(
    questions: List[str],
    *,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    max_seq_length: int,
    training: bool,
    freeze_backbone: bool,
    micro_batch_size: int = 0,
    output_dtype: Optional[torch.dtype] = None,
    strip_context: bool = False,
) -> torch.Tensor:
    processed = preprocess_pubmedqa_prompts_for_embedding(questions, strip_context=strip_context)
    mbs = int(micro_batch_size) if int(micro_batch_size) > 0 else len(processed)
    chunks: List[torch.Tensor] = []

    for start in range(0, len(processed), mbs):
        end = min(start + mbs, len(processed))
        inputs = tokenizer(
            processed[start:end],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
            padding=True,
        ).to(device)

        grad = training and not freeze_backbone
        with torch.set_grad_enabled(grad):
            outputs = model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        chunks.append(pooled)

    if not chunks:
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", 0)
        empty = torch.empty((0, int(hidden_size)), device=device, dtype=torch.float32)
        return empty.to(dtype=output_dtype) if output_dtype is not None else empty

    result = torch.cat(chunks, dim=0)
    if output_dtype is not None:
        result = result.to(dtype=output_dtype)
    return result
