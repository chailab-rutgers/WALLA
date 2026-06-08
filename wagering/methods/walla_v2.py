"""
WALLA V2: per-model routers with leave-one-out aggregated-probability average scores.
"""

from typing import Callable

from .walla_hidden_base import WallaHiddenRouterBase
from wagering.utils.walla_mechanism import average_scores_v2


class WallaV2(WallaHiddenRouterBase):
    """Each model has its own router; mechanism uses leave-one-out aggregated probs."""

    def _average_scores_fn(self) -> Callable[..., object]:
        return average_scores_v2
