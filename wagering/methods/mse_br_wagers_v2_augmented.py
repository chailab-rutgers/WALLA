"""Augmented MSE-BR wagering V2 method."""

import torch

from .mse_br_wagers_augmented_base import MSEBrWagersAugmentedBase


class MSEBrWagersV2Augmented(MSEBrWagersAugmentedBase):
    """V2 augmented variant: average score computed from weighted peer scores."""

    def _compute_average_scores(
        self,
        probs: torch.Tensor,
        gt_onehot_expanded: torch.Tensor,
        scores: torch.Tensor,
        sigmoid_wagers: torch.Tensor,
    ) -> torch.Tensor:
        del probs
        del gt_onehot_expanded
        wagers_except_i = torch.clamp(
            sigmoid_wagers.sum(dim=1, keepdim=True) - sigmoid_wagers,
            min=1e-16,
        )
        weighted_scores = scores * sigmoid_wagers
        average_scores = (
            weighted_scores.sum(dim=1, keepdim=True).expand_as(weighted_scores) - weighted_scores
        ) / wagers_except_i
        return average_scores
