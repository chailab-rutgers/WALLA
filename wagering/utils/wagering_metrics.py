"""
Shared wagering training/evaluation metrics (regret, Brier, meta metrics, gold labels).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from wagering.core.dataset import Dataset

log = logging.getLogger("wagering")


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
    dataset_indices: np.ndarray,
    example_local_indices: Optional[np.ndarray],
    datasets: List[Dataset],
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
    ds_ix = np.asarray(dataset_indices, dtype=np.int32)
    loc_ix = np.asarray(example_local_indices, dtype=np.int32)
    for dataset_idx in np.unique(ds_ix).tolist():
        ds_idx = int(dataset_idx)
        if ds_idx < 0 or ds_idx >= len(datasets):
            continue
        ds = datasets[ds_idx]
        dataset_name = getattr(ds, "cache_dataset_name", None)
        if not is_cluster_saturation_dataset_name(dataset_name):
            continue
        if not hasattr(ds, "probabilistic_labels"):
            continue
        if num_options != 2:
            raise ValueError(
                "probabilistic_labels are only supported for binary option sets "
                f"(num_options={num_options})"
            )
        pos_idx = resolve_positive_option_index(
            getattr(ds, "positive_label", None),
            option_tokens,
            num_options,
        )
        if pos_idx is None:
            raise ValueError(
                "Could not resolve positive option index for probabilistic labels"
            )
        mask = ds_ix == ds_idx
        local = loc_ix[mask].astype(np.int64, copy=False)
        gt_probs_all = np.asarray(ds.probabilistic_labels, dtype=np.float64)
        p_pos = gt_probs_all[local]
        p_pos = np.clip(p_pos, 0.0, 1.0)
        neg_idx = 1 - int(pos_idx)
        soft = np.zeros((int(mask.sum()), num_options), dtype=np.float64)
        soft[:, int(pos_idx)] = p_pos
        soft[:, neg_idx] = 1.0 - p_pos
        out[mask] = soft
    return out


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
    best_expert_ids: np.ndarray,
    model_brier_scores: Optional[np.ndarray] = None,
    model_rank_scores: Optional[np.ndarray] = None,
    best_model_ids: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Compute meta metrics treating wagers as predictions of best expert."""
    kendall_tau = None
    best_model_mrr = None
    if model_rank_scores is not None and best_model_ids is not None:
        try:
            kendall_tau = _compute_kendall_tau_from_scores(model_rank_scores, wagers)

            predicted_order = np.argsort(-wagers, axis=1, kind="stable")
            best_model_ranks = np.argmax(
                predicted_order == best_model_ids[:, np.newaxis], axis=1
            ) + 1
            best_model_mrr = float(np.mean(1.0 / best_model_ranks))
        except Exception as e:
            log.warning(f"Failed to compute kendall_tau/best_model_mrr from rank scores: {e}")
            kendall_tau = None
            best_model_mrr = None
    elif model_brier_scores is not None:
        try:
            kendall_tau = _compute_kendall_tau_from_scores(-model_brier_scores, wagers)

            best_model_ids = np.argmin(model_brier_scores, axis=1)
            predicted_order = np.argsort(-wagers, axis=1, kind="stable")
            best_model_ranks = np.argmax(
                predicted_order == best_model_ids[:, np.newaxis], axis=1
            ) + 1
            best_model_mrr = float(np.mean(1.0 / best_model_ranks))
        except Exception as e:
            log.warning(f"Failed to compute kendall_tau/best_model_mrr: {e}")
            kendall_tau = None
            best_model_mrr = None

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
