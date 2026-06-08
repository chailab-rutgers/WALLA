"""PubMedQA prompt preprocessing for embedding-based routers."""

import re
from typing import Iterable, List


def strip_pubmedqa_context(prompt: str) -> str:
    """
    Remove the Context section from a PubMedQA prompt while preserving all other sections.
    """
    text = str(prompt)
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
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def preprocess_pubmedqa_prompts_for_embedding(
    prompts: Iterable[str],
    strip_context: bool = True,
) -> List[str]:
    processed: List[str] = []
    for prompt in prompts:
        prompt_text = str(prompt)
        if strip_context:
            prompt_text = strip_pubmedqa_context(prompt_text)
        processed.append(prompt_text)
    return processed
