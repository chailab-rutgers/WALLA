"""
WALLA V1: per-model routers with peer-weighted average scores for the mechanism.
"""

from typing import Callable

from .walla_hidden_base import WallaHiddenRouterBase
from wagering.utils.walla_mechanism import average_scores_v1


class WallaV1(WallaHiddenRouterBase):
    """Each model has its own router; mechanism uses peer-weighted average scores."""

    def _average_scores_fn(self) -> Callable[..., object]:
        return average_scores_v1
