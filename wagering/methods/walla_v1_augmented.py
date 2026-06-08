"""Augmented WALLA wagering V1 method."""

from .walla_augmented_base import WallaAugmentedBase


class WallaV1Augmented(WallaAugmentedBase):
    """V1 augmented variant: average score computed from weighted peer scores."""
