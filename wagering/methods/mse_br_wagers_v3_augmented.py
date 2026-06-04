"""Augmented MSE-BR wagering V3 method."""

import torch

from .mse_br_wagers_augmented_base import MSEBrWagersAugmentedBase


class MSEBrWagersV3Augmented(MSEBrWagersAugmentedBase):
    """V3 augmented variant: average score computed from pooled peer probabilities."""

    def _compute_average_scores(
        self,
        probs: torch.Tensor,
        gt_onehot_expanded: torch.Tensor,
        scores: torch.Tensor,
        sigmoid_wagers: torch.Tensor,
    ) -> torch.Tensor:
        del scores
        num_options = probs.shape[-1]
        sigmoid_wagers_expanded = torch.repeat_interleave(
            sigmoid_wagers.unsqueeze(-1),
            repeats=num_options,
            dim=-1,
        )
        wagers_without_i = torch.clamp(
            sigmoid_wagers_expanded.sum(dim=1, keepdim=True) - sigmoid_wagers_expanded,
            min=1e-16,
        )
        agg_probs_without_i = (
            (probs * sigmoid_wagers_expanded).sum(dim=1, keepdim=True) - (probs * sigmoid_wagers_expanded)
        ) / wagers_without_i
        average_scores = 0.5 * (2 - ((agg_probs_without_i - gt_onehot_expanded) ** 2).sum(dim=-1))
        return average_scores
