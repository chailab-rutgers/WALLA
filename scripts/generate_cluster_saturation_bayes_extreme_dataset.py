#!/usr/bin/env python3
"""Generate a PubMedQA-style Bayesian saturation-forecast dataset.

Each example samples:
- prior_prob in {0.1, 0.9}
- tpr in [0.80, 0.95]
- fpr in [0.05, 0.15]
- evidence W: when prior_prob = 0.1, W = 1; when prior_prob = 0.9, W = 0

Labels:
- binary_label: 1 if posterior_prob > 0.5 else 0
- posterior_prob: Bayes posterior P(S=1 | W=evidence)

Optionally, rendered prompt variants can be included in CSV:
- input: full prompt with update rule and evidence
- prompt_without_context: redacted variant with evidence withheld
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List


def _sample_float(rng: random.Random, lo: float, hi: float, decimals: int = 4) -> float:
    value = rng.uniform(lo, hi)
    return round(value, decimals)


def _posterior(prior: float, tpr: float, fpr: float, evidence: int) -> float:
    if evidence == 1:
        num = tpr * prior
        den = (tpr * prior) + (fpr * (1.0 - prior))
    else:
        num = (1.0 - tpr) * prior
        den = ((1.0 - tpr) * prior) + ((1.0 - fpr) * (1.0 - prior))
    if den <= 0.0:
        raise ValueError("Invalid denominator while computing posterior")
    return num / den


def _format_full_prompt(prior_prob: float, tpr: float, fpr: float, evidence: int) -> str:
    return (
        # "In a football game, let $S \\in \\{0, 1\\}$ indicate if the Red Team wins ($S=1$). "
        # f"The empirical probability of a Red Team victory is $P(S=1) =$ {prior_prob:.4f}.\n\n"
        # "Update Rule:An analyst uses a spurious environmental signal—a local windstorm ($W \\in \\{0, 1\\}$)—to predict the outcome. "
        # f"Historical tracking yields these conditional probabilities for a windstorm occurring ($W=1$):$P(W=1|S=1) =$ {tpr:.4f}"
        # f"$P(W=1|S=0) =$ {fpr:.4f}.\n\n"
        # f"Evidence: Meteorological sensors at the facility output a reading of $W=$ {evidence}.\n\n"
        # "Question:Determine if it is more likely than not (with probability $> 0.5$) that the Red Team will win."
        # "Output exactly one character: 1 if it is more likely that the Red Team will win, or 0 otherwise."
        "A central load-balancing mechanism must forecast compute cluster saturation. "
        "Let the state space be $\\Omega = \\{0, 1\\}$. Let $S \\in \\Omega$ be the random variable "
        "indicating whether the primary compute cluster will become saturated ($>95\\%$ "
        "capacity) within the next $10$ minutes, where $S=1$ denotes saturation. "
        f"Based on current time-of-day traffic models, the empirical probability of saturation is $P(S=1) =$ {prior_prob:.2f}.\n\n"
        "Update Rule:To improve forecasting, the mechanism queries a lightweight, predictive monitoring agent. "
        "Let $W \\in \\Omega$ be the random variable representing the agent's output warning. "
        "The agent possesses the following conditional probabilities:"
        f"True Positive Rate (Probability of a warning given imminent saturation): $P(W=1|S=1) =$ {tpr:.2f}."
        f"False Positive Rate (Probability of a warning given normal loads): $P(W=1|S=0) =$ {fpr:.2f}.\n\n"
        f"Evidence:The monitoring agent analyzes the telemetry and outputs $W=$ {evidence}.\n\n"
        "Question:It is more likely than not that the cluster will saturate? "
        "Output exactly one character: 1 if it is more likely that the cluster will saturate, or 0 otherwise."
    )


def _format_prompt_without_context(prior_prob: float, tpr: float, fpr: float, evidence: int) -> str:
    return (
        # "In a football game, let $S \\in \\{0, 1\\}$ indicate if the Red Team wins ($S=1$). "
        # f"The empirical probability of a Red Team victory is $P(S=1) =$ {prior_prob:.4f}.\n\n"
        # "Update Rule:An analyst uses a spurious environmental signal—a local windstorm ($W \\in \\{0, 1\\}$)—to predict the outcome. "
        # f"Historical tracking yields these conditional probabilities for a windstorm occurring ($W=1$):$P(W=1|S=1) =$ {tpr:.4f}"
        # f"$P(W=1|S=0) =$ {fpr:.4f}.\n\n"
        # f"Evidence: Meteorological sensors at the facility output a reading of $W=$ {evidence}.\n\n"
        # "Question:Determine if it is more likely than not (with probability $> 0.5$) that the Red Team will win."
        # "Output exactly one character: 1 if it is more likely that the Red Team will win, or 0 otherwise."
        "A central load-balancing mechanism must forecast compute cluster saturation. "
        "Let the state space be $\\Omega = \\{0, 1\\}$. Let $S \\in \\Omega$ be the random variable "
        "indicating whether the primary compute cluster will become saturated ($>95\\%$ "
        "capacity) within the next $10$ minutes, where $S=1$ denotes saturation. "
        f"Based on current time-of-day traffic models, the empirical probability of saturation is $P(S=1) =$ {prior_prob:.2f}.\n\n"
        "Based on this empirical probability, answer the following question."
        "Question: Is it more likely than not that the cluster will saturate? "
        "Output exactly one character: 1 if it is more likely that the cluster will saturate, or 0 otherwise."
    )


def _build_row(example_id: int, rng: random.Random) -> Dict[str, object]:
    if rng.choice([True, False]):
        prior = 0.1
        evidence = 1
    else:
        prior = 0.9
        evidence = 0
    prior = round(prior, 2)
    tpr = _sample_float(rng, 0.90, 0.95, decimals=2)
    fpr = _sample_float(rng, 0.05, 0.10, decimals=2)

    posterior = _posterior(prior=prior, tpr=tpr, fpr=fpr, evidence=evidence)
    binary_label = 1 if posterior > 0.5 else 0

    row: Dict[str, object] = {
        "id": example_id,
        "binary_label": binary_label,
        "posterior_prob": round(posterior, 8),
        "prior_prob": prior,
        "tpr": tpr,
        "fpr": fpr,
        "evidence": evidence,
    }

    return row


def _add_rendered_prompts(row: Dict[str, object]) -> Dict[str, object]:
    prior = float(row["prior_prob"])
    tpr = float(row["tpr"])
    fpr = float(row["fpr"])
    evidence = int(row["evidence"])

    row["input"] = _format_full_prompt(prior=prior, tpr=tpr, fpr=fpr, evidence=evidence)
    row["prompt_without_context"] = _format_prompt_without_context(
        prior=prior,
        tpr=tpr,
        fpr=fpr,
        evidence=evidence,
    )
    return row


def _write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cluster saturation Bayes dataset CSV")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("workdir/datasets/cluster_saturation_bayes_extreme.csv"),
        help="Path to output CSV file",
    )
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of examples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--include-rendered-prompts",
        action="store_true",
        help="Also materialize full text prompts in CSV (input + prompt_without_context columns)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    rng = random.Random(args.seed)
    rows = [_build_row(example_id=i, rng=rng) for i in range(args.num_samples)]
    if args.include_rendered_prompts:
        rows = [_add_rendered_prompts(row=row) for row in rows]
    _write_csv(rows=rows, output_path=args.output_path)

    print(f"Wrote {len(rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
