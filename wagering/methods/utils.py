import re
from typing import Iterable, List

import torch

def compute_scoring_rule(model_probs: torch.Tensor, outcome: int, scoring_rule: str) -> torch.Tensor:
    """
    Compute strictly proper scoring rule s(p, omega) for each model.
    Supports both single samples and batches.
    
    Args:
        model_probs: Tensor of shape [num_models, num_options] (single sample)
                     or [batch_size, num_models, num_options] (batch)
        outcome: Integer index of the true outcome omega (single sample)
                 or Tensor of shape [batch_size] with outcome indices (batch)
        scoring_rule: The scoring rule to apply (e.g., 'logarithmic')
    
    Returns:
        scores: Tensor of shape [num_models] (single sample)
                or [batch_size, num_models] (batch)
    """
    # Detect batch mode
    is_batch = model_probs.ndim == 3  # [batch_size, num_models, num_options]
    
    if is_batch:
        # Batch mode: [batch_size, num_models, num_options]
        batch_size, num_models, num_options = model_probs.shape
        
        if scoring_rule == "logarithmic":
            # Logarithmic scoring rule: s(p, omega) = log(p[omega])
            # outcome should be [batch_size]
            batch_indices = torch.arange(batch_size, device=model_probs.device)
            scores = torch.log(model_probs[batch_indices, :, outcome] + 1e-10)  # [batch_size, num_models]
        elif scoring_rule == "brier":
            # Brier score: mean squared error between predicted probabilities and actual outcome
            outcome_one_hot = (torch.arange(num_options, device=model_probs.device).view(1, 1, -1) == 
                              outcome.view(batch_size, 1, 1)).float()  # [batch_size, 1, num_options]
            scores = 1 - ((model_probs - outcome_one_hot) ** 2).mean(dim=2)  # [batch_size, num_models]
        else:
            raise ValueError(f"Unknown scoring rule: {scoring_rule}")
        
        return scores
    else:
        # Single sample mode: [num_models, num_options]
        if scoring_rule == "logarithmic":
            # Logarithmic scoring rule: s(p, omega) = log(p[omega])
            scores = torch.log(model_probs[:, outcome] + 1e-10)
        elif scoring_rule == "brier":
            # Brier score: mean squared error between predicted probabilities and actual outcome
            scores = 1 -((model_probs - (torch.arange(model_probs.size(1)) == outcome).float().view(1, -1).to(model_probs.device)) ** 2).mean(dim=1)
        else:
            raise ValueError(f"Unknown scoring rule: {scoring_rule}")
        
        return scores


def is_likely_pubmedqa_prompt(prompt: str) -> bool:
    """Heuristic detection for PubMedQA-style prompts used in this codebase."""
    lowered = str(prompt).lower()
    # We support two common prompt shapes in this repo:
    # 1) "Question: ... Context: ... Long Answer: ... Answer with YES/NO ..."
    # 2) "Question: ... Context: ... Answer with YES/NO ..." (context-only, no Long Answer section)
    has_question = "question:" in lowered
    has_context = "context:" in lowered
    has_yes_no = "answer with yes or no" in lowered
    has_long_answer = "long answer:" in lowered
    # For stripping we really just need to be confident that a "Context:" block exists and that this
    # prompt is in the PubMedQA family.
    return bool(has_question and has_context and (has_yes_no or has_long_answer))


def strip_pubmedqa_context(prompt: str) -> str:
    """
    Remove the Context section from a PubMedQA prompt while preserving all other sections.

    Expected format includes Question, optional Context, and Long Answer blocks.
    If no Context block is found, returns the original prompt unchanged.
    """
    text = str(prompt)

    # Remove everything between the "Context:" header and the next section header.
    # Some prompt templates include "Long Answer:"; others jump directly to "Answer...".
    context_pattern = re.compile(
        r"(\n?\s*context:\s*\n?)(.*?)(?="
        r"\n\s*long answer:\s*"
        r"|\n\s*answer\s+with\s+yes\s+or\s+no\b"
        r"|\n\s*answer\s*:\s*"
        r"|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not context_pattern.search(text):
        return text

    stripped = context_pattern.sub("\n", text)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped


def preprocess_pubmedqa_prompts_for_embedding(
    prompts: Iterable[str],
    strip_context: bool = True,
) -> List[str]:
    """Optionally strip *any* `Context:` blocks before embedding generation.

    In this repo, multiple datasets reuse `mixed_context_routing: pubmedqa` and/or
    PubMedQA-derived prompt templates that include a `Context:` section but may not
    include the canonical PubMedQA `Long Answer:` section. When `strip_context` is
    enabled, we strip the `Context:` block whenever it appears rather than relying
    on a prompt-shape heuristic.
    """
    processed: List[str] = []
    for prompt in prompts:
        prompt_text = str(prompt)
        if strip_context:
            prompt_text = strip_pubmedqa_context(prompt_text)
        processed.append(prompt_text)
    return processed
