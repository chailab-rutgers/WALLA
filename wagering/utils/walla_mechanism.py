"""WALLA mechanism math: Brier scores, peer averages, BRs, and Nash gap."""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def brier_scores(
    probs: torch.Tensor,
    gold_label: torch.Tensor,
    gold_label_distribution: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, _, num_options = probs.shape

    if gold_label_distribution is not None:
        gt_dist = gold_label_distribution.to(device=probs.device, dtype=probs.dtype)
        if gt_dist.ndim != 2 or gt_dist.shape[0] != batch_size or gt_dist.shape[1] != num_options:
            raise ValueError(
                "gold_label_distribution must be shape [batch_size, num_options], "
                f"got {tuple(gt_dist.shape)}"
            )
        gt_dist_expanded = gt_dist.unsqueeze(1).expand(batch_size, probs.shape[1], num_options)
        return 1.0 + (probs**2).sum(dim=-1) - 2.0 * (probs * gt_dist_expanded).sum(dim=-1)

    gt_onehot = F.one_hot(gold_label, num_classes=num_options).float()
    gt_onehot_expanded = gt_onehot.unsqueeze(1).expand(batch_size, probs.shape[1], num_options)
    return ((probs - gt_onehot_expanded) ** 2).sum(dim=-1)


def scores_from_brier(brier_scores_tensor: torch.Tensor) -> torch.Tensor:
    return 0.5 * (2 - brier_scores_tensor)


def average_scores_v1(scores: torch.Tensor, sigmoid_wagers: torch.Tensor) -> torch.Tensor:
    wagers_except_i = torch.clamp(
        sigmoid_wagers.sum(dim=1, keepdim=True) - sigmoid_wagers,
        min=1e-16,
    )
    weighted_scores = scores * sigmoid_wagers
    return (
        weighted_scores.sum(dim=1, keepdim=True).expand_as(weighted_scores) - weighted_scores
    ) / wagers_except_i


def average_scores_v2(
    probs: torch.Tensor,
    sigmoid_wagers: torch.Tensor,
    gold_label: torch.Tensor,
    gold_label_distribution: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_options = probs.shape[-1]

    sigmoid_wagers_expanded = torch.repeat_interleave(
        sigmoid_wagers.unsqueeze(-1), repeats=num_options, dim=-1
    )
    wager_except_i = torch.clamp(
        sigmoid_wagers_expanded.sum(dim=1, keepdim=True) - sigmoid_wagers_expanded,
        min=1e-16,
    )
    agg_probs_without_i = (
        (probs * sigmoid_wagers_expanded).sum(dim=1, keepdim=True)
        - (probs * sigmoid_wagers_expanded)
    ) / wager_except_i

    return scores_from_brier(
        brier_scores(agg_probs_without_i, gold_label, gold_label_distribution)
    )


def extract_mechanism(
    sigmoid_wagers: torch.Tensor,
    model_logits: torch.Tensor,
    gold_label: torch.Tensor,
    gold_label_distribution: Optional[torch.Tensor],
    compute_average_scores: Callable[..., torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(model_logits, dim=-1)
    brier = brier_scores(probs, gold_label, gold_label_distribution)
    scores = scores_from_brier(brier)

    if compute_average_scores is average_scores_v1:
        average_scores = compute_average_scores(scores, sigmoid_wagers)
    else:
        average_scores = compute_average_scores(
            probs, sigmoid_wagers, gold_label, gold_label_distribution
        )

    score_diff = scores - average_scores
    brs = torch.clamp(score_diff, min=1e-16, max=1.0 - 1e-16)
    total_payout = sigmoid_wagers * (score_diff - 0.5 * sigmoid_wagers)
    nash_gap = brs * (score_diff - 0.5 * brs) - total_payout
    return brs, nash_gap, score_diff, total_payout


def extract_mechanism_components(
    sigmoid_wagers: torch.Tensor,
    model_logits: torch.Tensor,
    gold_label: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    brs, nash_gap, score_diff, total_payout = extract_mechanism(
        sigmoid_wagers,
        model_logits,
        gold_label,
        gold_label_distribution=None,
        compute_average_scores=average_scores_v1,
    )
    probs = F.softmax(model_logits, dim=-1)
    brier = brier_scores(probs, gold_label, None)
    scores = scores_from_brier(brier)
    average_scores = average_scores_v1(scores, sigmoid_wagers)
    return {
        "scores": scores,
        "average_scores": average_scores,
        "score_diff": score_diff,
        "brs": brs,
        "nash_gap": nash_gap,
        "total_payout": total_payout,
    }
