"""
Shared wagering training/evaluation metrics (regret, Brier, meta metrics, gold labels).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from wagering.core.dataset import Dataset

log = logging.getLogger("wagering")


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


def compute_model_probs_from_logits(model_logits: np.ndarray) -> np.ndarray:
    """Convert logits [num_examples, num_models, num_options] to probabilities."""
    max_logits = np.max(model_logits, axis=2, keepdims=True)
    stabilized = model_logits - max_logits
    exp_stabilized = np.exp(stabilized)
    return exp_stabilized / (np.sum(exp_stabilized, axis=2, keepdims=True) + 1e-20)


def compute_model_brier_scores(model_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-example, per-model multiclass Brier scores."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    num_options = model_logits.shape[2]
    one_hot_labels = np.eye(num_options, dtype=np.float64)[labels_arr]
    model_probs = compute_model_probs_from_logits(model_logits)
    return np.sum((model_probs - one_hot_labels[:, np.newaxis, :]) ** 2, axis=2)


def compute_model_brier_scores_soft_binary(
    model_logits: np.ndarray,
    *,
    gt_positive_probs: np.ndarray,
    positive_option_index: int,
) -> np.ndarray:
    """
    Compute per-model Brier scores when the binary ground-truth is a soft probability.

    Args:
        model_logits: [num_examples, num_models, 2]
        gt_positive_probs: [num_examples] probability of the positive option
        positive_option_index: index of the positive option in the 2-way output

    Returns:
        model_brier: [num_examples, num_models]
    """
    logits = np.asarray(model_logits, dtype=np.float64)
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("soft-binary Brier expects model_logits shape [N, M, 2]")
    gt = np.asarray(gt_positive_probs, dtype=np.float64)
    if gt.ndim != 1 or gt.shape[0] != logits.shape[0]:
        raise ValueError("gt_positive_probs must be 1D and match model_logits first dim")
    pos_idx = int(positive_option_index)
    if pos_idx not in (0, 1):
        raise ValueError("positive_option_index must be 0 or 1 for binary tasks")

    y = np.zeros((logits.shape[0], 2), dtype=np.float64)
    y[:, pos_idx] = gt
    y[:, 1 - pos_idx] = 1.0 - gt

    model_probs = compute_model_probs_from_logits(logits)
    return np.sum((model_probs - y[:, np.newaxis, :]) ** 2, axis=2)


def _compute_kendall_tau_from_scores(
    target_scores: np.ndarray,
    predicted_scores: np.ndarray,
) -> Optional[float]:
    """
    Compute average Kendall's tau using pairwise concordance over examples.

    Ties are counted in the denominator (total pairs) but contribute 0 to the
    numerator, matching (concordant - discordant) / total_pairs.
    """
    num_models = target_scores.shape[1]
    if num_models < 2:
        return None

    pair_i, pair_j = np.triu_indices(num_models, k=1)
    total_pairs = len(pair_i)
    if total_pairs == 0:
        return None

    target_diffs = target_scores[:, pair_i] - target_scores[:, pair_j]
    predicted_diffs = predicted_scores[:, pair_i] - predicted_scores[:, pair_j]
    pair_products = target_diffs * predicted_diffs

    concordant = np.sum(pair_products > 0, axis=1)
    discordant = np.sum(pair_products < 0, axis=1)
    tau_per_example = (concordant - discordant) / float(total_pairs)
    return float(np.mean(tau_per_example))


def is_cluster_saturation_dataset_name(dataset_name: Optional[str]) -> bool:
    """Return True when dataset name refers to cluster_saturation_bayes."""
    if not dataset_name:
        return False
    return "cluster_saturation" in str(dataset_name).strip().lower()


def resolve_positive_option_index(
    positive_label: Optional[Any],
    option_tokens: List[str],
    num_options: int,
) -> Optional[int]:
    """Resolve positive class index for binary probabilistic metrics."""
    option_token_to_index = {str(token): idx for idx, token in enumerate(option_tokens)}
    if positive_label is not None:
        resolved = option_token_to_index.get(str(positive_label))
        if resolved is not None:
            return int(resolved)
    if num_options == 2:
        return 1
    return None


def build_gold_label_distribution_for_rows(
    labels: np.ndarray,
    example_local_indices: Optional[np.ndarray],
    dataset: Dataset,
    option_tokens: List[str],
    num_options: int,
) -> np.ndarray:
    """
    Build per-row target distributions [N, num_options] for Brier / regret.

    Defaults to one-hot(labels). Rows from cluster_saturation* datasets with
    ``probabilistic_labels`` use the soft binary vector [p, 1-p].
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    n = int(labels_arr.shape[0])
    out = np.eye(num_options, dtype=np.float64)[labels_arr]
    if example_local_indices is None:
        return out

    dataset_name = getattr(dataset, "cache_dataset_name", None)
    if not is_cluster_saturation_dataset_name(dataset_name):
        return out
    if not hasattr(dataset, "probabilistic_labels"):
        return out
    if num_options != 2:
        raise ValueError(
            "probabilistic_labels are only supported for binary option sets "
            f"(num_options={num_options})"
        )
    pos_idx = resolve_positive_option_index(
        getattr(dataset, "positive_label", None),
        option_tokens,
        num_options,
    )
    if pos_idx is None:
        raise ValueError(
            "Could not resolve positive option index for probabilistic labels"
        )
    loc_ix = np.asarray(example_local_indices, dtype=np.int32).astype(np.int64, copy=False)
    gt_probs_all = np.asarray(dataset.probabilistic_labels, dtype=np.float64)
    p_pos = gt_probs_all[loc_ix]
    p_pos = np.clip(p_pos, 0.0, 1.0)
    neg_idx = 1 - int(pos_idx)
    soft = np.zeros((n, num_options), dtype=np.float64)
    soft[:, int(pos_idx)] = p_pos
    soft[:, neg_idx] = 1.0 - p_pos
    return soft


def compute_model_bernoulli_kl_to_gt_scores(
    model_logits: np.ndarray,
    gt_positive_probs: np.ndarray,
    positive_option_index: int,
) -> np.ndarray:
    """Compute per-example, per-model KL(gt || pred) for binary probabilities."""
    model_probs = compute_model_probs_from_logits(model_logits)
    pred_positive_probs = np.clip(
        model_probs[:, :, positive_option_index],
        1e-10,
        1.0 - 1e-10,
    )
    target_positive_probs = np.clip(
        np.asarray(gt_positive_probs, dtype=np.float64)[:, np.newaxis],
        1e-10,
        1.0 - 1e-10,
    )
    target_negative_probs = 1.0 - target_positive_probs
    pred_negative_probs = 1.0 - pred_positive_probs
    kl_scores = (
        target_positive_probs * np.log(target_positive_probs / pred_positive_probs)
        + target_negative_probs * np.log(target_negative_probs / pred_negative_probs)
    )
    return kl_scores


def compute_mean_kl_to_gold_distribution(
    gold_distributions: np.ndarray,
    predicted_distributions: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Mean KL(gold || pred) over rows (optionally masked).

    Returns None when no rows are selected.
    """
    gold = np.asarray(gold_distributions, dtype=np.float64)
    pred = np.asarray(predicted_distributions, dtype=np.float64)
    if gold.ndim != 2 or pred.ndim != 2 or gold.shape != pred.shape:
        raise ValueError(f"Expected gold/pred shape [N, K] and equal; got {gold.shape} vs {pred.shape}")

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape[0] != gold.shape[0]:
            raise ValueError(f"Mask length mismatch: {m.shape[0]} vs N={gold.shape[0]}")
        if not np.any(m):
            return None
        gold = gold[m]
        pred = pred[m]

    gold = np.clip(gold, 0.0, 1.0)
    row_sums = np.sum(gold, axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    gold = gold / row_sums
    pred = np.clip(pred, 1e-10, 1.0)
    kl = np.sum(gold * (np.log(gold + 1e-10) - np.log(pred)), axis=1)
    out = float(np.mean(kl))
    if not np.isfinite(out):
        raise ValueError(f"Non-finite KL computed: {out}")
    return out

def compute_brier_dynamic_regret(
    model_logits: np.ndarray,
    aggregated_probs: np.ndarray,
    labels: np.ndarray,
    gt_positive_probs: Optional[np.ndarray] = None,
    positive_option_index: Optional[int] = None,
    gold_label_distribution: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Brier Dynamic Regret: aggregated_brier - best_expert_brier.
    """
    num_examples = int(aggregated_probs.shape[0])
    num_options = int(aggregated_probs.shape[1])

    if gold_label_distribution is not None:
        y = np.asarray(gold_label_distribution, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError("gold_label_distribution must be 2D [num_examples, num_options]")
        if y.shape != (num_examples, num_options):
            raise ValueError(
                "gold_label_distribution shape must match aggregated_probs "
                f"(got {y.shape}, expected {(num_examples, num_options)})"
            )
    elif gt_positive_probs is not None:
        if positive_option_index is None:
            raise ValueError("positive_option_index is required when gt_positive_probs is provided")
        if num_options != 2:
            raise ValueError("gt_positive_probs path requires binary num_options==2")
        gt_positive_probs_arr = np.asarray(gt_positive_probs, dtype=np.float64)
        if gt_positive_probs_arr.ndim != 1:
            raise ValueError("gt_positive_probs must be a 1D array")
        if gt_positive_probs_arr.shape[0] != num_examples:
            raise ValueError(
                "gt_positive_probs length must match number of examples in aggregated_probs"
            )
        pos_idx = int(positive_option_index)
        neg_idx = 1 - pos_idx
        y = np.zeros((num_examples, num_options), dtype=np.float64)
        y[:, pos_idx] = gt_positive_probs_arr
        y[:, neg_idx] = 1.0 - gt_positive_probs_arr
    else:
        labels_arr = np.asarray(labels, dtype=np.int64)
        y = np.eye(num_options, dtype=np.float64)[labels_arr]

    model_probs = compute_model_probs_from_logits(model_logits)
    model_brier = np.sum((model_probs - y[:, np.newaxis, :]) ** 2, axis=2)
    best_expert_brier = np.min(model_brier, axis=1)
    aggregated_brier = np.sum((aggregated_probs - y) ** 2, axis=1)
    return float(np.mean(aggregated_brier - best_expert_brier))


def compute_meta_metrics(
    wagers: np.ndarray,
    model_brier_scores: Optional[np.ndarray] = None,
    model_rank_scores: Optional[np.ndarray] = None,
    best_model_ids: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Compute meta metrics treating wagers as predictions of best expert."""
    kendall_tau = None
    best_model_mrr = None
    if model_rank_scores is not None and best_model_ids is not None:
        kendall_tau = _compute_kendall_tau_from_scores(model_rank_scores, wagers)
        predicted_order = np.argsort(-wagers, axis=1, kind="stable")
        best_model_ranks = np.argmax(
            predicted_order == best_model_ids[:, np.newaxis], axis=1
        ) + 1
        best_model_mrr = float(np.mean(1.0 / best_model_ranks))
    elif model_brier_scores is not None:
        kendall_tau = _compute_kendall_tau_from_scores(-model_brier_scores, wagers)
        best_model_ids = np.argmin(model_brier_scores, axis=1)
        predicted_order = np.argsort(-wagers, axis=1, kind="stable")
        best_model_ranks = np.argmax(
            predicted_order == best_model_ids[:, np.newaxis], axis=1
        ) + 1
        best_model_mrr = float(np.mean(1.0 / best_model_ranks))

    return {
        "kendall_tau": kendall_tau,
        "best_model_mrr": best_model_mrr,
    }


def compute_normalized_wager_probability_stats(
    wagers: np.ndarray,
    brier_best_model_ids: np.ndarray,
) -> Dict[str, Any]:
    """
    Summary stats for wager weights normalized to a probability simplex (w / sum w).
    """
    w = np.asarray(wagers, dtype=np.float64)
    best_ids = np.asarray(brier_best_model_ids, dtype=np.int64)
    if w.ndim != 2:
        raise ValueError(f"wagers must be 2D [N, M], got shape {w.shape}")
    if best_ids.ndim != 1 or best_ids.shape[0] != w.shape[0]:
        raise ValueError(
            "brier_best_model_ids must be shape [N] matching wagers rows "
            f"got {best_ids.shape} vs N={w.shape[0]}"
        )
    row_sums = w.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 1e-10):
        raise ValueError("Each row of wagers must have positive sum for normalization")
    norm = w / row_sums
    per_mean = norm.mean(axis=0)
    per_var = norm.var(axis=0, ddof=0)
    n = norm.shape[0]
    at_best = norm[np.arange(n, dtype=np.int64), best_ids]
    return {
        "wager_prob_mean_per_model": [float(x) for x in per_mean.tolist()],
        "wager_prob_var_per_model": [float(x) for x in per_var.tolist()],
        "brier_best_wager_prob_mean": float(np.mean(at_best)),
        "brier_best_wager_prob_var": float(np.var(at_best, ddof=0)),
    }


def compute_inverse_hhi(wagers: np.ndarray) -> float:
    """Effective number of models (inverse HHI) averaged over examples."""
    w = np.asarray(wagers, dtype=np.float64)
    sum_w = np.sum(w, axis=1)
    sum_w2 = np.sum(w * w, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.divide(sum_w * sum_w, sum_w2)
    return float(np.nanmean(n_eff))


def compute_classification_metrics(
    predictions: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    soft_binary_targets: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Accuracy, NLL, and multiclass Brier score."""
    predictions_arr = np.asarray(predictions, dtype=np.int64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)

    accuracy = float(np.mean(predictions_arr == labels_arr))
    correct_class_probs = probs_arr[np.arange(len(labels_arr)), labels_arr]
    nll = float(-np.mean(np.log(correct_class_probs + 1e-10)))

    num_options = probs_arr.shape[1]
    if soft_binary_targets is not None:
        y = np.asarray(soft_binary_targets, dtype=np.float64)
        if y.shape != probs_arr.shape:
            raise ValueError(
                f"soft_binary_targets shape {y.shape} must match probs {probs_arr.shape}"
            )
        brier = float(np.mean(np.sum((probs_arr - y) ** 2, axis=1)))
    else:
        one_hot_labels = np.eye(num_options, dtype=np.float64)[labels_arr]
        brier = float(np.mean(np.sum((probs_arr - one_hot_labels) ** 2, axis=1)))

    return {"accuracy": accuracy, "nll": nll, "brier": brier}


def compute_confidence_metrics(
    probs: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """AUC (correctness vs max prob) and ECE."""
    probs_arr = np.asarray(probs, dtype=np.float64)
    predictions_arr = np.asarray(predictions, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)

    max_probs = probs_arr.max(axis=1)
    correctness_int = (predictions_arr == labels_arr).astype(int)
    if len(np.unique(correctness_int)) >= 2:
        auc = float(roc_auc_score(correctness_int, max_probs))
    else:
        auc = float("nan")

    ece_metric = ECE(normalize=False, n_bins=20)
    confidences = max_probs
    correctness = (predictions_arr == labels_arr).astype(float)
    finite_mask = np.isfinite(confidences) & np.isfinite(correctness)
    if not np.any(finite_mask):
        raise ValueError("No finite confidence/correctness pairs available for ECE")
    ece = float(
        ece_metric(
            confidences[finite_mask].tolist(),
            correctness[finite_mask].tolist(),
        )
    )
    return {"auc": auc, "ece": ece}


def compute_avg_wager_summaries(
    wagers_history: np.ndarray,
    sigmoid_wagers_history: Optional[np.ndarray] = None,
    total_payout_history: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Average wagers, sigmoid wagers, and net payout per model."""
    w = np.asarray(wagers_history, dtype=np.float64)
    out: Dict[str, Any] = {
        "avg_wager_per_model": np.mean(w, axis=0).astype(np.float64),
        "avg_wager_total": float(np.mean(np.sum(w, axis=1))),
        "avg_sigmoid_wager_per_model": None,
        "avg_sigmoid_wager_total": None,
        "avg_net_payout_per_model": None,
        "avg_net_payout_total": None,
    }

    if sigmoid_wagers_history is not None:
        sw = np.asarray(sigmoid_wagers_history, dtype=np.float64)
        if sw.ndim != 2 or sw.shape[0] != w.shape[0]:
            raise ValueError(
                f"sigmoid_wagers_history shape {sw.shape} must be [N, M] matching wagers {w.shape}"
            )
        out["avg_sigmoid_wager_per_model"] = np.mean(sw, axis=0).astype(np.float64)
        out["avg_sigmoid_wager_total"] = float(np.mean(np.sum(sw, axis=1)))

    if total_payout_history is not None:
        payout_arr = np.asarray(total_payout_history, dtype=np.float64)
        if payout_arr.ndim != 2 or payout_arr.shape[0] != w.shape[0]:
            raise ValueError(
                f"total_payout_history shape {payout_arr.shape} must be [N, M] matching wagers {w.shape}"
            )
        out["avg_net_payout_per_model"] = np.mean(payout_arr, axis=0).astype(np.float64)
        out["avg_net_payout_total"] = float(np.mean(np.sum(payout_arr, axis=1)))

    return out


def resolve_binary_probability_labels(
    dataset: Dataset,
    num_examples: int,
) -> Optional[np.ndarray]:
    """
    Return per-example positive-class probabilities when configured.

    Uses ``probability_labels`` (from ``probability_label_column``). Raises if only
    ``probabilistic_labels`` is present without ``probability_labels``.
    """
    prob_labs = getattr(dataset, "probability_labels", None)
    if prob_labs is None and getattr(dataset, "probabilistic_labels", None) is not None:
        raise ValueError(
            "Dataset provides `probabilistic_labels` but not `probability_labels`. "
            "KL/TV must use `probability_labels` (configured via `probability_label_column`)."
        )
    if isinstance(prob_labs, (list, tuple)) and len(prob_labs) == num_examples:
        return np.asarray(prob_labs, dtype=np.float64)
    return None


def _resolve_binary_pos_idx(dataset: Dataset, option_tokens: List[str]) -> int:
    pos_marker = getattr(dataset, "positive_label", None)
    resolved = resolve_positive_option_index(pos_marker, option_tokens, 2)
    return int(resolved) if resolved is not None else 0


def compute_bernoulli_metrics_binary(
    aggregated_probs: np.ndarray,
    labels: np.ndarray,
    dataset: Dataset,
    option_tokens: List[str],
    *,
    prob_labs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Bernoulli KL/TV for binary tasks; optionally soft-binary Brier."""
    probs = np.asarray(aggregated_probs, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if probs.shape[1] != 2:
        raise ValueError("compute_bernoulli_metrics_binary requires binary probs")

    pos_idx = _resolve_binary_pos_idx(dataset, option_tokens)
    pred_vec = probs[:, pos_idx]
    if prob_labs is not None:
        target_vec = np.asarray(prob_labs, dtype=np.float64)
    else:
        target_vec = (labels_arr == pos_idx).astype(np.float64)

    out = {
        "bernoulli_kl": bernoulli_kl_divergence(pred_vec.tolist(), target_vec.tolist()),
        "bernoulli_tv": bernoulli_tv_distance(pred_vec.tolist(), target_vec.tolist()),
        "brier": float("nan"),
    }
    if prob_labs is not None:
        y_soft = np.zeros((probs.shape[0], 2), dtype=np.float64)
        y_soft[:, pos_idx] = prob_labs
        y_soft[:, 1 - pos_idx] = 1.0 - prob_labs
        out["brier"] = float(np.mean(np.sum((probs - y_soft) ** 2, axis=1)))
    return out


def compute_wagering_derived_metrics(
    model_logits: np.ndarray,
    aggregated_probs: np.ndarray,
    labels: np.ndarray,
    wagers: np.ndarray,
    dataset: Dataset,
    option_tokens: List[str],
    *,
    prob_labs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Brier dynamic regret, meta metrics, and normalized wager probability stats."""
    num_examples = aggregated_probs.shape[0]
    num_options = aggregated_probs.shape[1]
    pos_idx = _resolve_binary_pos_idx(dataset, option_tokens) if num_options == 2 else 0

    if num_options == 2 and prob_labs is not None and len(prob_labs) == num_examples:
        brier_d_regret = compute_brier_dynamic_regret(
            model_logits,
            aggregated_probs,
            labels,
            gt_positive_probs=prob_labs,
            positive_option_index=pos_idx,
        )
        model_brier_scores = compute_model_brier_scores_soft_binary(
            model_logits,
            gt_positive_probs=prob_labs,
            positive_option_index=pos_idx,
        )
    else:
        brier_d_regret = compute_brier_dynamic_regret(
            model_logits, aggregated_probs, labels
        )
        model_brier_scores = compute_model_brier_scores(model_logits, labels)

    meta_metrics = compute_meta_metrics(wagers, model_brier_scores)
    brier_best_model_ids = np.argmin(model_brier_scores, axis=1)
    wager_prob_stats = compute_normalized_wager_probability_stats(
        wagers, brier_best_model_ids
    )

    return {
        "brier_d_regret": brier_d_regret,
        "kendall_tau": meta_metrics["kendall_tau"],
        "best_model_mrr": meta_metrics["best_model_mrr"],
        "wager_prob_mean_per_model": wager_prob_stats["wager_prob_mean_per_model"],
        "wager_prob_var_per_model": wager_prob_stats["wager_prob_var_per_model"],
        "brier_best_wager_prob_mean": wager_prob_stats["brier_best_wager_prob_mean"],
        "brier_best_wager_prob_var": wager_prob_stats["brier_best_wager_prob_var"],
    }


def compute_subset_any_model_wrong_metrics(
    predictions: np.ndarray,
    aggregated_probs: np.ndarray,
    labels: np.ndarray,
    model_logits: np.ndarray,
    wagers_history: np.ndarray,
    dataset: Dataset,
    option_tokens: List[str],
    *,
    prob_labs: Optional[np.ndarray] = None,
    avg_inference_time_per_batch_s: float,
) -> Dict[str, Any]:
    """Metrics on examples where at least one base model prediction is wrong."""
    base_preds = np.argmax(model_logits, axis=2)
    base_correct = base_preds == labels[:, None]
    subset_mask = ~np.all(base_correct, axis=1)
    subset_n = int(np.sum(subset_mask))

    subset_metrics: Dict[str, Any] = {
        "subset_name": "any_model_wrong",
        "num_examples": subset_n,
    }
    if subset_n == 0:
        return subset_metrics

    sub_labels = labels[subset_mask]
    sub_probs = aggregated_probs[subset_mask]
    sub_preds = predictions[subset_mask]
    sub_wagers = wagers_history[subset_mask]
    sub_model_logits = model_logits[subset_mask]
    sub_prob_labs = prob_labs[subset_mask] if prob_labs is not None else None

    cls = compute_classification_metrics(sub_preds, sub_probs, sub_labels)
    subset_metrics.update(cls)

    conf = compute_confidence_metrics(sub_probs, sub_preds, sub_labels)
    subset_metrics.update(conf)

    subset_metrics["inverse_hhi"] = compute_inverse_hhi(sub_wagers)
    subset_metrics["avg_inference_time_per_batch_s"] = avg_inference_time_per_batch_s

    derived = compute_wagering_derived_metrics(
        sub_model_logits,
        sub_probs,
        sub_labels,
        sub_wagers,
        dataset,
        option_tokens,
        prob_labs=sub_prob_labs,
    )
    subset_metrics["brier_d_regret"] = derived["brier_d_regret"]
    subset_metrics["kendall_tau"] = derived["kendall_tau"]
    subset_metrics["best_model_mrr"] = derived["best_model_mrr"]

    if sub_probs.shape[1] == 2:
        bern = compute_bernoulli_metrics_binary(
            sub_probs,
            sub_labels,
            dataset,
            option_tokens,
            prob_labs=sub_prob_labs,
        )
        subset_metrics["bernoulli_kl"] = bern["bernoulli_kl"]
        subset_metrics["bernoulli_tv"] = bern["bernoulli_tv"]
        if sub_prob_labs is not None:
            subset_metrics["brier"] = bern["brier"]

    return subset_metrics


def compute_evaluation_metrics(
    *,
    predictions: np.ndarray,
    aggregated_probs: np.ndarray,
    labels: np.ndarray,
    model_logits_stacked: np.ndarray,
    wagers_history: np.ndarray,
    dataset: Dataset,
    option_tokens: List[str],
    inference_times_s: Sequence[float],
    sigmoid_wagers_history: Optional[np.ndarray] = None,
    total_payout_history: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics after the inference batch loop."""
    num_examples = len(labels)
    model_logits = np.transpose(model_logits_stacked, (1, 0, 2))
    prob_labs = resolve_binary_probability_labels(dataset, num_examples)

    inverse_hhi = compute_inverse_hhi(wagers_history)
    avg_inference_time_per_batch_s = (
        float(np.mean(np.asarray(inference_times_s, dtype=np.float64)))
        if inference_times_s
        else float("nan")
    )

    soft_binary_targets = None
    if aggregated_probs.shape[1] == 2 and prob_labs is not None:
        pos_idx = _resolve_binary_pos_idx(dataset, option_tokens)
        soft_binary_targets = np.zeros((num_examples, 2), dtype=np.float64)
        soft_binary_targets[:, pos_idx] = prob_labs
        soft_binary_targets[:, 1 - pos_idx] = 1.0 - prob_labs

    cls = compute_classification_metrics(
        predictions, aggregated_probs, labels, soft_binary_targets=soft_binary_targets
    )
    conf = compute_confidence_metrics(aggregated_probs, predictions, labels)

    bernoulli_kl = float("nan")
    bernoulli_tv = float("nan")
    if aggregated_probs.shape[1] == 2:
        bern = compute_bernoulli_metrics_binary(
            aggregated_probs,
            labels,
            dataset,
            option_tokens,
            prob_labs=prob_labs,
        )
        bernoulli_kl = bern["bernoulli_kl"]
        bernoulli_tv = bern["bernoulli_tv"]
        if prob_labs is not None:
            cls["brier"] = bern["brier"]

    derived = compute_wagering_derived_metrics(
        model_logits,
        aggregated_probs,
        labels,
        wagers_history,
        dataset,
        option_tokens,
        prob_labs=prob_labs,
    )
    wager_summaries = compute_avg_wager_summaries(
        wagers_history,
        sigmoid_wagers_history=sigmoid_wagers_history,
        total_payout_history=total_payout_history,
    )
    subset_metrics = compute_subset_any_model_wrong_metrics(
        predictions,
        aggregated_probs,
        labels,
        model_logits,
        wagers_history,
        dataset,
        option_tokens,
        prob_labs=prob_labs,
        avg_inference_time_per_batch_s=avg_inference_time_per_batch_s,
    )

    return {
        "inverse_hhi": inverse_hhi,
        "avg_inference_time_per_batch_s": avg_inference_time_per_batch_s,
        "accuracy": cls["accuracy"],
        "nll": cls["nll"],
        "brier": cls["brier"],
        "bernoulli_kl": bernoulli_kl,
        "bernoulli_tv": bernoulli_tv,
        "auc": conf["auc"],
        "ece": conf["ece"],
        "brier_d_regret": derived["brier_d_regret"],
        "kendall_tau": derived["kendall_tau"],
        "best_model_mrr": derived["best_model_mrr"],
        "wager_prob_mean_per_model": derived["wager_prob_mean_per_model"],
        "wager_prob_var_per_model": derived["wager_prob_var_per_model"],
        "brier_best_wager_prob_mean": derived["brier_best_wager_prob_mean"],
        "brier_best_wager_prob_var": derived["brier_best_wager_prob_var"],
        "subset_any_model_wrong": subset_metrics,
        **wager_summaries,
    }
