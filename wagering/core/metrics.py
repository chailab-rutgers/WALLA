from typing import List

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def bernoulli_tv_distance(pred_probs: List[float], target_probs: List[float]) -> float:
    """Mean total variation distance between Bernoulli(pred) and Bernoulli(target)."""
    pred = np.asarray(pred_probs, dtype=np.float64)
    target = np.asarray(target_probs, dtype=np.float64)

    if pred.shape != target.shape:
        raise ValueError("pred_probs and target_probs must have the same shape")
    if pred.ndim != 1:
        raise ValueError("pred_probs and target_probs must be 1D arrays")
    if np.any(pred < 0.0) or np.any(pred > 1.0):
        raise ValueError("pred_probs must be in [0, 1]")
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("target_probs must be in [0, 1]")

    # For Bernoulli distributions, TV distance simplifies to |p - q|.
    return float(np.mean(np.abs(pred - target)))


def bernoulli_kl_divergence(
    pred_probs: List[float],
    target_probs: List[float],
    eps: float = 1e-10,
) -> float:
    """Mean KL divergence D_KL(Bernoulli(target) || Bernoulli(pred))."""
    pred = np.asarray(pred_probs, dtype=np.float64)
    target = np.asarray(target_probs, dtype=np.float64)

    if pred.shape != target.shape:
        raise ValueError("pred_probs and target_probs must have the same shape")
    if pred.ndim != 1:
        raise ValueError("pred_probs and target_probs must be 1D arrays")
    if np.any(pred < 0.0) or np.any(pred > 1.0):
        raise ValueError("pred_probs must be in [0, 1]")
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("target_probs must be in [0, 1]")

    pred_safe = np.clip(pred, eps, 1.0 - eps)
    target_safe = np.clip(target, eps, 1.0 - eps)

    kl = (
        target_safe * np.log(target_safe / pred_safe)
        + (1.0 - target_safe) * np.log((1.0 - target_safe) / (1.0 - pred_safe))
    )
    return float(np.mean(kl))


class ECE:
    """Expected Calibration Error for confidence-style estimators."""

    def __init__(self, normalize: bool = False, n_bins: int = 20):
        self.normalize = normalize
        self.n_bins = n_bins

    def __str__(self) -> str:
        return "ece"

    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        scores_array = np.asarray(scores).reshape(-1, 1)
        return MinMaxScaler().fit_transform(scores_array).flatten()

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        if len(estimator) != len(target):
            raise ValueError("Estimator and target must have the same length.")

        estimator_array = np.asarray(estimator, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)

        # Standard ECE expects confidence scores in [0, 1].
        confidences = estimator_array

        if self.normalize:
            confidences = self.normalize_scores(confidences)

        if np.any(confidences < 0.0) or np.any(confidences > 1.0):
            raise ValueError("ECE confidences must be in [0, 1]")

        if np.any(target_array < 0.0) or np.any(target_array > 1.0):
            raise ValueError("ECE targets must be in [0, 1]")

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece, n_total = 0.0, len(confidences)

        for i in range(self.n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = (
                (confidences > lo) & (confidences <= hi)
                if i > 0
                else (confidences >= lo) & (confidences <= hi)
            )
            if not np.any(in_bin):
                continue

            acc_bin = np.mean(target_array[in_bin])
            conf_bin = np.mean(confidences[in_bin])
            ece += (np.sum(in_bin) / n_total) * abs(acc_bin - conf_bin)

        return float(ece)
