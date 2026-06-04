"""
Training pipeline for multi-LLM wagering methods.
"""

import logging
import os
import shutil
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence
from collections import deque
import copy
import numpy as np
import torch
import pandas as pd

# Ensure local project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wagering.core.model import WhiteboxModel
from wagering.core.dataset import Dataset

# Local wagering imports
from wagering.methods.base import WageringMethod
from wagering.training.analytics import WageringAnalytics
from wagering.aggregation.base import AggregationFunction
from wagering.utils.multi_llm_ensemble import (
    collect_option_logits_and_hidden_states_for_model,
    extract_hidden_state_features,
    get_concatenated_router_prompts,
    get_model_specific_prompts,
    get_model_prompt_variant,
    get_cached_logits_and_hidden_states_for_model,
    resolve_hidden_state_layers_for_model,
    set_cached_logits_and_hidden_states_for_model,
    _get_mixed_context_dataset_type,
)

log = logging.getLogger("wagering")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

from sklearn.metrics import roc_auc_score
from wagering.core.metrics import ECE


def _union_hidden_state_layers_wagering_plus_last(
    wagering_layers: Optional[Sequence[int]],
    *,
    include_last_transformer_layer: bool,
) -> Optional[List[int]]:
    """Stable-unique merge of wagering layers with ``[-1]`` for calibration collection.

    When ``include_last_transformer_layer`` is false, returns a shallow copy of
    ``wagering_layers`` (or None).
    """
    if not include_last_transformer_layer:
        return list(wagering_layers) if wagering_layers is not None else None
    merged: List[int] = []
    seen: set[int] = set()
    for src in wagering_layers or []:
        xi = int(src)
        if xi in seen:
            continue
        seen.add(xi)
        merged.append(xi)
    if -1 not in seen:
        merged.append(-1)
    return merged if merged else [-1]


def _compute_model_probs_from_logits(model_logits: np.ndarray) -> np.ndarray:
    """Convert logits [num_examples, num_models, num_options] to probabilities."""
    max_logits = np.max(model_logits, axis=2, keepdims=True)
    stabilized = model_logits - max_logits
    exp_stabilized = np.exp(stabilized)
    return exp_stabilized / (np.sum(exp_stabilized, axis=2, keepdims=True) + 1e-20)


def _compute_model_brier_scores(model_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-example, per-model multiclass Brier scores."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    num_options = model_logits.shape[2]
    one_hot_labels = np.eye(num_options, dtype=np.float64)[labels_arr]
    model_probs = _compute_model_probs_from_logits(model_logits)
    return np.sum((model_probs - one_hot_labels[:, np.newaxis, :]) ** 2, axis=2)


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


def _is_cluster_saturation_dataset_name(dataset_name: Optional[str]) -> bool:
    """Return True when dataset name refers to cluster_saturation_bayes."""
    if not dataset_name:
        return False
    return "cluster_saturation" in str(dataset_name).strip().lower()


def _kl_qp_categorical_rows(q: np.ndarray, p: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """KL(q || p) per row for discrete distributions q, p of shape [batch, num_options]."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    q = np.clip(q, eps, 1.0)
    p = np.clip(p, eps, 1.0)
    return np.sum(q * (np.log(q) - np.log(p)), axis=-1)


def _tv_distance_per_model(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Total variation distance d_TV(p, q) = (1/2) * sum_k |p_k - q_k|.

    Args:
        p: [batch, num_models, num_options]
        q: [batch, num_options]

    Returns:
        [batch, num_models]
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return 0.5 * np.sum(np.abs(p - q[:, np.newaxis, :]), axis=-1)


def _context_model_index_per_row_for_batch(
    batch_dataset_indices: np.ndarray,
    batch_example_local_indices: np.ndarray,
    datasets: List[Dataset],
) -> np.ndarray:
    """
    For each row, return the ensemble slot index that receives the *with-context* prompt
    (e.g. ``prompt_helper`` / full prompt for PubMedQA-style mixed routing), or -1 if unknown.
    """
    ds_ix = np.asarray(batch_dataset_indices, dtype=np.int64)
    loc_ix = np.asarray(batch_example_local_indices, dtype=np.int64)
    bsz = int(ds_ix.shape[0])
    out = np.full(bsz, -1, dtype=np.int64)
    for b in range(bsz):
        d = int(ds_ix[b])
        if d < 0 or d >= len(datasets):
            continue
        ds = datasets[d]
        routing_type = _get_mixed_context_dataset_type(ds)
        if routing_type is None:
            continue
        raw = getattr(ds, f"{routing_type}_context_assignment_by_example", None)
        if raw is None:
            continue
        n = len(raw)
        loc = int(loc_ix[b])
        if 0 <= loc < n:
            try:
                out[b] = int(raw[loc])
            except (TypeError, ValueError, IndexError):
                out[b] = -1
    return out


def _debug_log_batch_prob_vs_gt_distribution(
    *,
    epoch: int,
    batch_idx: int,
    batch_start: int,
    batch_end: int,
    batch_gold_label_distribution: Optional[np.ndarray],
    batch_labels: np.ndarray,
    batch_model_probs: np.ndarray,
    batch_dataset_indices: np.ndarray,
    batch_example_local_indices: np.ndarray,
    datasets: List[Dataset],
    option_tokens: List[str],
) -> None:
    """
    Debug: compare each model's predicted class distribution to the batch ground-truth
    distribution (from ``probability_label_column`` / ``gold_label_distribution`` when
    available, else one-hot ``batch_labels``).

    Logs (1) counts of which model's P(positive) is closest to GT P(positive),
    (2) mean KL(q||p) for that best model, (3) mean over examples of mean KL(q||p)
    over non-best models,     (4) fraction of rows where that best model equals the
    mixed-context ``*_context_assignment_by_example`` slot using |P(pos)-Q(pos)|,
    TV distance on the full binary distribution, and argmin_m KL(q||p_m).
    """
    q = batch_gold_label_distribution
    if q is None:
        labels_arr = np.asarray(batch_labels, dtype=np.int64)
        num_options = int(batch_model_probs.shape[2])
        q = np.eye(num_options, dtype=np.float32)[labels_arr]
    q = np.asarray(q, dtype=np.float64)
    p_all = np.asarray(batch_model_probs, dtype=np.float64)
    bsz, num_models, num_options = p_all.shape
    if num_options != 2:
        log.info(
            "debug_batch_prob_alignment: skipped (num_options=%d != 2)",
            num_options,
        )
        return

    pos_idx_per_ex = np.empty(bsz, dtype=np.int64)
    for b in range(bsz):
        ds_idx = int(batch_dataset_indices[b])
        if ds_idx < 0 or ds_idx >= len(datasets):
            r = _resolve_positive_option_index(None, option_tokens, num_options)
            pos_idx_per_ex[b] = int(r) if r is not None else 1
            continue
        ds = datasets[ds_idx]
        r = _resolve_positive_option_index(
            getattr(ds, "positive_label", None),
            option_tokens,
            num_options,
        )
        pos_idx_per_ex[b] = int(r) if r is not None else 1

    ri = np.arange(bsz)[:, np.newaxis]
    mi = np.arange(num_models)[np.newaxis, :]
    pi = pos_idx_per_ex[:, np.newaxis]
    p_pos = p_all[ri, mi, pi]
    q_pos = q[np.arange(bsz), pos_idx_per_ex]
    abs_err = np.abs(p_pos - q_pos[:, np.newaxis])
    best_m = np.argmin(abs_err, axis=1)

    tv_bm = _tv_distance_per_model(p_all, q)
    best_m_tv = np.argmin(tv_bm, axis=1)

    p_best = p_all[np.arange(bsz), best_m, :]
    kl_best = _kl_qp_categorical_rows(q, p_best)

    kl_all = np.empty((bsz, num_models), dtype=np.float64)
    for m in range(num_models):
        kl_all[:, m] = _kl_qp_categorical_rows(q, p_all[:, m, :])
    best_m_kl = np.argmin(kl_all, axis=1)

    other_mask = np.ones((bsz, num_models), dtype=bool)
    other_mask[np.arange(bsz), best_m] = False
    n_other = np.maximum(other_mask.sum(axis=1), 1)
    kl_other_mean_per_ex = (kl_all * other_mask).sum(axis=1) / n_other

    counts = np.bincount(best_m, minlength=num_models)
    ctx_model = _context_model_index_per_row_for_batch(
        batch_dataset_indices,
        batch_example_local_indices,
        datasets,
    )
    ctx_ok = ctx_model >= 0
    if np.any(ctx_ok):
        pct_ctx = 100.0 * float(np.mean(best_m[ctx_ok] == ctx_model[ctx_ok]))
        pct_ctx_tv = 100.0 * float(np.mean(best_m_tv[ctx_ok] == ctx_model[ctx_ok]))
        pct_ctx_kl = 100.0 * float(np.mean(best_m_kl[ctx_ok] == ctx_model[ctx_ok]))
        ctx_pct_str = f"{pct_ctx:.2f}%"
        ctx_pct_tv_str = f"{pct_ctx_tv:.2f}%"
        ctx_pct_kl_str = f"{pct_ctx_kl:.2f}%"
    else:
        ctx_pct_str = "n/a_no_mixed_context_routing_on_batch"
        ctx_pct_tv_str = "n/a_no_mixed_context_routing_on_batch"
        ctx_pct_kl_str = "n/a_no_mixed_context_routing_on_batch"

    log.info(
        "debug_prob_align epoch=%d batch=%d rows=[%d:%d) |best_model_counts|=%s | "
        "mean_KL_best=%.6f mean_of_mean_KL_nonbest=%.6f | "
        "pct_best_model_is_context_slot=%s pct_best_model_is_context_slot_tv=%s "
        "pct_best_model_is_context_slot_kl=%s "
        "(over %d/%d rows with assignment; TV=d_TV=0.5*L1; KL=KL(q||p))",
        epoch,
        batch_idx,
        batch_start,
        batch_end,
        counts.tolist(),
        float(np.mean(kl_best)),
        float(np.mean(kl_other_mean_per_ex)),
        ctx_pct_str,
        ctx_pct_tv_str,
        ctx_pct_kl_str,
        int(np.count_nonzero(ctx_ok)),
        bsz,
    )


def _resolve_positive_option_index(
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
        # Binary fallback when positive label is not explicitly configured.
        return 1
    return None


def _build_gold_label_distribution_for_rows(
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
    ``probabilistic_labels`` (e.g. ``probability_label_column``) use the soft
    binary vector [p, 1-p] aligned to ``positive_label`` / option indices.
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
        if not _is_cluster_saturation_dataset_name(dataset_name):
            continue
        if not hasattr(ds, "probabilistic_labels"):
            continue
        if num_options != 2:
            raise ValueError(
                "probabilistic_labels are only supported for binary option sets "
                f"(num_options={num_options})"
            )
        pos_idx = _resolve_positive_option_index(
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


def _compute_model_bernoulli_kl_to_gt_scores(
    model_logits: np.ndarray,
    gt_positive_probs: np.ndarray,
    positive_option_index: int,
) -> np.ndarray:
    """Compute per-example, per-model KL(gt || pred) for binary probabilities."""
    model_probs = _compute_model_probs_from_logits(model_logits)
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


def _compute_mean_kl_to_gold_distribution(
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


def compute_dynamic_regret(
    model_logits: np.ndarray,
    aggregated_probs: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute Dynamic Regret (DRegret): aggregated_nll - best_expert_nll.
    
    Args:
        model_logits: [num_examples, num_models, num_options] - logits from each model
        aggregated_probs: [num_examples, num_options] - aggregated probability distributions
        labels: [num_examples] - true labels
        
    Returns:
        (d_regret, best_expert_ids): Average dynamic regret and best expert id per example
    """
    num_examples = model_logits.shape[0]
    labels_arr = np.asarray(labels, dtype=np.int64)
    model_probs = _compute_model_probs_from_logits(model_logits)

    # Compute NLL for each model on each example [num_examples, num_models].
    true_label_indices = labels_arr[:, np.newaxis, np.newaxis]
    model_true_probs = np.take_along_axis(model_probs, true_label_indices, axis=2).squeeze(axis=2)
    model_nlls = -np.log(model_true_probs + 1e-10)
    
    # Find best expert (lowest NLL) for each example
    best_expert_ids = np.argmin(model_nlls, axis=1)  # [num_examples]
    best_expert_nlls = model_nlls[np.arange(num_examples), best_expert_ids]  # [num_examples]
    
    # Compute aggregated NLL
    aggregated_nlls = -np.log(aggregated_probs[np.arange(num_examples), labels_arr] + 1e-10)  # [num_examples]
    
    # Dynamic regret = aggregated_nll - best_expert_nll
    d_regret = np.mean(aggregated_nlls - best_expert_nlls)
    
    return d_regret, best_expert_ids


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

    Target distribution ``y`` per row is chosen in order:
    ``gold_label_distribution`` (full vector, e.g. soft labels), else
    ``gt_positive_probs`` + ``positive_option_index`` (binary [p, 1-p] only),
    else one-hot(``labels``).

    Args:
        model_logits: [num_examples, num_models, num_options] - logits from each model
        aggregated_probs: [num_examples, num_options] - aggregated probability distributions
        labels: [num_examples] - true labels (used when no soft target is provided)

    Returns:
        Average Brier dynamic regret over examples.
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

    model_probs = _compute_model_probs_from_logits(model_logits)
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
    """
    Compute meta metrics treating wagers as predictions of best expert.
    
    Args:
        wagers: [num_examples, num_models] - wager distributions (probability simplex)
        best_expert_ids: [num_examples] - best expert id for each example
        
    Returns:
        Dictionary with meta_acc, meta_nll, meta_auc, kendall_tau, best_model_mrr
    """
    num_examples = wagers.shape[0]
    num_models = wagers.shape[1]
    
    # Meta accuracy: does argmax(wagers) match best expert?
    predicted_expert = np.argmax(wagers, axis=1)  # [num_examples]
    meta_acc = np.mean(predicted_expert == best_expert_ids)
    
    # Meta NLL: -log(wager[best_expert_id])
    meta_nlls = -np.log(wagers[np.arange(num_examples), best_expert_ids] + 1e-10)
    meta_nll = np.mean(meta_nlls)
    
    # Meta AUC: one-vs-rest AUC for each expert, then average
    # Create one-hot labels
    meta_auc = None
    try:
        # For each model, compute binary AUC (is this the best expert?)
        aucs = []
        for model_idx in range(num_models):
            binary_labels = (best_expert_ids == model_idx).astype(int)
            if len(np.unique(binary_labels)) >= 2:  # Need both classes
                auc = roc_auc_score(binary_labels, wagers[:, model_idx])
                aucs.append(auc)
        
        if len(aucs) > 0:
            meta_auc = np.mean(aucs)
        else:
            meta_auc = np.nan
    except Exception as e:
        log.warning(f"Failed to compute meta_auc: {e}")
        meta_auc = np.nan
    
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
            # Lower Brier is better, so negate scores to get higher-is-better ranking.
            kendall_tau = _compute_kendall_tau_from_scores(-model_brier_scores, wagers)

            # MRR of the Brier-best model position in wager ranking.
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
        "meta_acc": meta_acc,
        "meta_nll": meta_nll,
        "meta_auc": meta_auc if not np.isnan(meta_auc) else None,
        "kendall_tau": kendall_tau,
        "best_model_mrr": best_model_mrr,
    }


def compute_normalized_wager_probability_stats(
    wagers: np.ndarray,
    brier_best_model_ids: np.ndarray,
) -> Dict[str, Any]:
    """
    Summary stats for wager weights normalized to a probability simplex (w / sum w),
    matching LinearPooling / LogarithmicPooling.

    Args:
        wagers: [num_examples, num_models] non-negative wagers with positive row sums.
        brier_best_model_ids: [num_examples] index of the lowest per-example multiclass
            Brier score among experts (same tie-break as ``np.argmin`` on Brier).

    Returns:
        Per-model mean/variance of normalized wager mass across examples, and mean/variance
        of the normalized wager placed on the Brier-best expert on each example.
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


class WageringTrainer:
    """
    Trainer for multi-LLM wagering methods.
    
    Handles training loop, logging, checkpointing, and evaluation.
    """
    
    def __init__(
        self,
        models: List[WhiteboxModel],
        datasets: List[Dataset],
        wagering_method: WageringMethod,
        aggregation_function: AggregationFunction,
        option_tokens: List[str] = ["A", "B", "C", "D"],
        checkpoint_dir: Optional[str] = None,
        wandb_logger: Optional[Any] = None,
        save_every: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
        resume_from_checkpoint: Optional[str] = None,
        shuffle_data: bool = True,
        shuffle_seed: int = 42,
        early_stopping_patience: int = 10,
        stop_at_last_iteration: bool = False,
        batch_size: int = 100,  # Batch size for training loop
        validation_split_ratio: float = 0.1,  # Fraction of data to use for validation (default: 10%)
        balance_training_datasets: bool = True,
        early_stopping_criterion: str = "validation",
        use_brier_d_regret_for_early_stopping: bool = True,
        use_min_kl_for_early_stopping: bool = False,
        wager_score_plot_every: Optional[int] = None,
        logit_calibrator: Optional[Any] = None,
        save_epoch_checkpoints: bool = True,
        max_epoch_checkpoints: Optional[int] = None,
        enable_artifact_outputs: bool = True,
        reuse_prompt_perplexities_for_identical_models: bool = False,
        max_training_batches: Optional[int] = None,
        model_configs_for_sequential_perplexity: Optional[List[Dict[str, Any]]] = None,
        perplexity_load_cache_kwargs: Optional[Dict[str, Any]] = None,
        debug_batch_prob_alignment: bool = False,
    ):
        """
        Initialize the trainer.
        
        Args:
            models: List of WhiteboxModel instances
            datasets: List of Dataset instances (will be concatenated for training)
            wagering_method: WageringMethod instance
            aggregation_function: AggregationFunction instance
            option_tokens: List of option tokens (e.g., ["A", "B", "C", "D"])
            checkpoint_dir: Directory for saving checkpoints
            wandb_logger: Optional wandb logger
            save_every: Save checkpoint every N batches
            early_stopping_patience: Number of non-improving intervals before stopping.
                Uses epochs for ``validation`` criterion and batches for ``online_learning`` criterion.
            stop_at_last_iteration: If True, disable early stopping and always run
                through the final configured training iteration (epochs/batches).
            early_stopping_criterion: Early stopping strategy.
                - ``validation``: epoch-level stopping based on validation metrics (existing behavior)
                                - ``online_learning``: batch-level stopping on a rolling training window
                                    (window size chosen so ``window_batches * batch_size`` roughly matches
                                    validation-set size)
            use_brier_d_regret_for_early_stopping: If True, use Brier dynamic regret as the
                monitored early-stopping metric. For ``validation`` criterion, uses validation
                set ``brier_d_regret``. For ``online_learning`` criterion, uses the rolling
                training-window ``brier_d_regret``. When a row has a soft gold distribution
                (e.g. cluster_saturation* with ``probability_label_column`` / ``batch_gold_label_distribution``),
                Brier regret uses the full target vector; otherwise it uses one-hot class labels.
            use_min_kl_for_early_stopping: If True, use mean KL divergence between the
                ground-truth distribution and the predicted distribution (KL(gold || pred))
                on the validation set as the monitored early-stopping metric (lower is better).
                This is only applicable when the monitored split contains datasets with soft
                probabilistic labels (i.e. ``probability_label_column`` provided, exposed as
                ``dataset.probabilistic_labels``).
            balance_training_datasets: If True, randomly subsample each training
                dataset to the minimum dataset size before concatenation.
            save_epoch_checkpoints: If True, save transition checkpoints each epoch.
                Disable to reduce disk usage for large hyperparameter sweeps.
            max_epoch_checkpoints: Maximum number of transition checkpoints to keep.
                If set, older transition checkpoints are removed after each save.
            enable_artifact_outputs: If False, skip non-essential artifact writes
                (analytics CSVs and plot files) to reduce I/O overhead.
            reuse_prompt_perplexities_for_identical_models: If True, reuse prompt
                perplexities across ensemble slots when the same loaded model sees
                an identical prompt list (common in mixed-context PubMedQA setups).
            max_training_batches: If set, stop after this many training-loop batches
                (optimizer steps) across epochs. The cap counts batches executed in
                the current ``train()`` call from the resume point onward.
            model_configs_for_sequential_perplexity: Merged per-model YAML dicts; used
                to load one model at a time for prompt perplexity when visible GPUs
                are fewer than ensemble slots.
            perplexity_load_cache_kwargs: Optional kwargs (e.g. ``cache_dir``) for
                those sequential loads.
            debug_batch_prob_alignment: If True, each training batch logs which
                model's predicted positive-class probability is closest to the batch
                ground-truth distribution (``probability_label_column`` when present),
                plus mean KL(q||p) for the closest model and for the remaining models.
        """
        self.models = models
        self.datasets = datasets
        self.wagering_method = wagering_method
        self.aggregation_function = aggregation_function
        self.option_tokens = option_tokens
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.wandb_logger = wandb_logger
        self.save_every = save_every
        self.metadata = metadata or {}
        self.resume_from_checkpoint = resume_from_checkpoint
        self.shuffle_data = shuffle_data
        self.shuffle_seed = shuffle_seed
        self.early_stopping_patience = early_stopping_patience
        self.stop_at_last_iteration = bool(stop_at_last_iteration)
        self.early_stopping_criterion = str(early_stopping_criterion).strip().lower()
        if self.early_stopping_criterion not in {"validation", "online_learning"}:
            raise ValueError(
                "early_stopping_criterion must be one of {'validation', 'online_learning'}, "
                f"got: {early_stopping_criterion}"
            )
        self.use_brier_d_regret_for_early_stopping = bool(use_brier_d_regret_for_early_stopping)
        self.use_min_kl_for_early_stopping = bool(use_min_kl_for_early_stopping)
        if self.use_min_kl_for_early_stopping and self.use_brier_d_regret_for_early_stopping:
            raise ValueError(
                "Only one early-stopping metric override may be enabled at a time. "
                "Set at most one of use_brier_d_regret_for_early_stopping / use_min_kl_for_early_stopping."
            )
        if self.use_min_kl_for_early_stopping and self.early_stopping_criterion not in {"validation", "online_learning"}:
            raise ValueError(
                "use_min_kl_for_early_stopping=True requires early_stopping_criterion in "
                "{'validation', 'online_learning'}"
            )
        self.batch_size = batch_size
        self.validation_split_ratio = validation_split_ratio
        self.balance_training_datasets = balance_training_datasets
        self.save_epoch_checkpoints = bool(save_epoch_checkpoints)
        self.enable_artifact_outputs = bool(enable_artifact_outputs)
        self.max_epoch_checkpoints = None
        if max_epoch_checkpoints is not None:
            try:
                parsed_max_checkpoints = int(max_epoch_checkpoints)
                if parsed_max_checkpoints > 0:
                    self.max_epoch_checkpoints = parsed_max_checkpoints
                else:
                    log.warning(
                        "Ignoring non-positive max_epoch_checkpoints=%s. Keeping all epoch checkpoints.",
                        max_epoch_checkpoints,
                    )
            except (TypeError, ValueError):
                log.warning(
                    "Ignoring invalid max_epoch_checkpoints=%s. Keeping all epoch checkpoints.",
                    max_epoch_checkpoints,
                )
        self.wager_score_plot_every = None
        if wager_score_plot_every is not None:
            try:
                plot_every_value = int(wager_score_plot_every)
                if plot_every_value > 0:
                    self.wager_score_plot_every = plot_every_value
                else:
                    log.warning(
                        "Ignoring non-positive wager_score_plot_every=%s. "
                        "Wager-vs-score_diff plotting is disabled.",
                        wager_score_plot_every,
                    )
            except (TypeError, ValueError):
                log.warning(
                    "Ignoring invalid wager_score_plot_every=%s. "
                    "Wager-vs-score_diff plotting is disabled.",
                    wager_score_plot_every,
                )
        if not self.enable_artifact_outputs:
            self.wager_score_plot_every = None
        self.logit_calibrator = logit_calibrator
        self.max_training_batches: Optional[int] = None
        if max_training_batches is not None:
            try:
                mtb = int(max_training_batches)
                if mtb > 0:
                    self.max_training_batches = mtb
                else:
                    log.warning(
                        "Ignoring non-positive max_training_batches=%s.",
                        max_training_batches,
                    )
            except (TypeError, ValueError):
                log.warning(
                    "Ignoring invalid max_training_batches=%s.",
                    max_training_batches,
                )
        self.reuse_prompt_perplexities_for_identical_models = bool(
            reuse_prompt_perplexities_for_identical_models
        )
        self.requires_hidden_states = bool(getattr(self.wagering_method, "requires_hidden_states", True))
        self.use_concatenated_prompt_context = bool(
            getattr(self.wagering_method, "use_concatenated_prompt_context", False)
        )
        self.hidden_state_layers = getattr(self.wagering_method, "hidden_state_layers", None)
        self.hidden_state_layers_per_model = getattr(self.wagering_method, "hidden_state_layers_per_model", None)
        self.method_requires_model_perplexities = bool(
            getattr(self.wagering_method, "requires_model_perplexities", False)
        )
        self._model_configs_for_sequential_perplexity = model_configs_for_sequential_perplexity
        self._perplexity_load_cache_kwargs = perplexity_load_cache_kwargs or {}
        self._router_concatenated_prompts_by_dataset: Dict[int, List[str]] = {}
        self._router_prompts_per_model_by_dataset: Dict[int, List[List[str]]] = {}
        self.debug_batch_prob_alignment = bool(debug_batch_prob_alignment)

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_step = 0
        self.start_epoch = 0
        self.wagers_history = []
        self.metrics_history = []
        self.batch_metrics_history: List[Dict[str, Any]] = []
        
        # Running average tracker for last 5 batches (for wandb logging)
        self.running_avg_window = 10
        self.batch_metrics_buffer = deque(maxlen=self.running_avg_window)  # Store last N batches of metrics

        # Cache the most recent validation metrics (for final logging fallback)
        self.last_val_metrics: Optional[Dict[str, Any]] = None
        
        # Early stopping state (must be initialized before checkpoint loading)
        # Now epoch-based instead of step-based
        # Note: best_d_regret tracks validation d_regret if validation set exists, otherwise training d_regret
        # d_regret is a loss metric, so lower is better (initialized to infinity)
        self.best_d_regret = float('inf')
        self.best_brier_d_regret = float('inf')
        self.best_kl_to_gold = float("inf")
        self.best_nash_gap = float('inf')  # For methods that provide Nash gap metric
        self.best_batch_nash_gap = float('inf')  # For online-learning batch-level nash gap criterion
        self.best_batch_brier_d_regret = float('inf')
        self.best_batch_kl_to_gold = float("inf")
        self.epochs_since_improvement = 0
        self.batches_since_improvement = 0
        self.early_stopped = False
        self.best_wagering_method_state = None  # Store the best checkpoint state
        self.best_epoch = None  # Track which epoch had the best checkpoint
        self.best_batch_step = None  # Track global step for online-learning best checkpoint

        if self.resume_from_checkpoint:
            self._load_checkpoint(Path(self.resume_from_checkpoint))

        # If logging into an already-active wandb run (for example phase2 after
        # phase1), ensure training steps never move backward relative to run.step.
        run_step = self._get_wandb_run_step()
        if run_step is not None and self.current_step < run_step:
            log.info(
                "Aligning trainer current_step from %d to active wandb run step %d to keep logging monotonic",
                self.current_step,
                run_step,
            )
            self.current_step = int(run_step)
        
        # Collect per-dataset cached logits/hidden states first, then combine datasets and shuffle
        self._collect_logits()
        self._collect_hidden_states()
        self._apply_logit_calibration()
        self._prepare_datasets()
        # Sanity check: combined dataset length must match cached logits/hidden states
        if hasattr(self, "all_model_logits") and self.all_model_logits is not None:
            combined_len = len(self.combined_dataset.x)
            if self.all_model_logits.shape[1] != combined_len:
                raise RuntimeError(
                    f"Combined dataset size ({combined_len}) does not match cached logits size "
                    f"({self.all_model_logits.shape[1]})."
                )
        if hasattr(self, "all_hidden_states") and self.all_hidden_states is not None:
            combined_len = len(self.combined_dataset.x)
            if isinstance(self.all_hidden_states, list):
                for i, hs in enumerate(self.all_hidden_states):
                    if hs.shape[0] != combined_len:
                        raise RuntimeError(
                            f"Combined dataset size ({combined_len}) does not match cached hidden states "
                            f"for model {i} ({hs.shape[0]})."
                        )
            else:
                if self.all_hidden_states.shape[1] != combined_len:
                    raise RuntimeError(
                        f"Combined dataset size ({combined_len}) does not match cached hidden states size "
                        f"({self.all_hidden_states.shape[1]})."
                    )
        self._apply_shuffling()
        self._prepare_model_perplexities()

    @staticmethod
    def _compute_prompt_perplexities_for_model(
        model: WhiteboxModel,
        prompts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        """
        Compute true prompt perplexity per example using teacher-forced next-token loss.

        Returns:
            np.ndarray of shape [num_examples], where lower values indicate better
            prompt modeling by this model.
        """
        if len(prompts) == 0:
            return np.empty((0,), dtype=np.float32)

        model_device = model.device()
        ppl_batches: List[np.ndarray] = []
        pad_token_id = getattr(model.tokenizer, "pad_token_id", None)

        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            batch = model.tokenize(batch_prompts)
            input_ids = batch["input_ids"].to(model_device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                )
                logits = outputs.logits

            if logits.size(1) < 2:
                # Degenerate short prompt; assign neutral perplexity.
                ppl_batches.append(np.ones((input_ids.size(0),), dtype=np.float32))
                continue

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            token_log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_nll = -torch.gather(token_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            if attention_mask is not None:
                token_mask = attention_mask[:, 1:].to(dtype=token_nll.dtype)
            else:
                token_mask = torch.ones_like(token_nll, dtype=token_nll.dtype)

            if pad_token_id is not None:
                token_mask = token_mask * (shift_labels != pad_token_id).to(dtype=token_nll.dtype)

            token_count = torch.clamp(token_mask.sum(dim=1), min=1.0)
            mean_nll = (token_nll * token_mask).sum(dim=1) / token_count
            perplexity = torch.exp(mean_nll)
            ppl_batches.append(perplexity.detach().to(dtype=torch.float32).cpu().numpy())

        return np.concatenate(ppl_batches, axis=0).astype(np.float32, copy=False)

    def _should_use_sequential_perplexity_load(self) -> bool:
        if self._model_configs_for_sequential_perplexity is None:
            return False
        if len(self._model_configs_for_sequential_perplexity) != len(self.models):
            return False
        from wagering.utils.model_utils import should_load_prompt_perplexity_models_sequentially

        return should_load_prompt_perplexity_models_sequentially(len(self.models))

    def _compute_prompt_perplexities_sequential(self, dataset: Dataset) -> np.ndarray:
        """Load one HF model at a time when VRAM cannot hold the full ensemble."""
        import gc

        from wagering.utils.model_utils import load_models_from_config

        num_examples = len(dataset.x)
        num_models = len(self.models)
        cfgs = self._model_configs_for_sequential_perplexity
        if cfgs is None or len(cfgs) != num_models:
            raise RuntimeError("Sequential perplexity requires model_configs matching ensemble size")

        all_perplexities = np.empty((num_examples, num_models), dtype=np.float32)
        reused_columns = 0
        computed_columns = 0
        prompt_perplexity_cache: Dict[Tuple[str, str], np.ndarray] = {}
        batch_size = max(1, int(dataset.batch_size))

        log.info(
            "Computing prompt perplexities sequentially (%d models; %d visible CUDA device(s))",
            num_models,
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        )

        for model_index in range(num_models):
            model_prompts = get_model_specific_prompts(dataset, model_index=model_index)
            if len(model_prompts) != num_examples:
                raise ValueError(
                    "Prompt/label length mismatch while computing prompt perplexities. "
                    f"prompts={len(model_prompts)}, examples={num_examples}"
                )

            path_key = str(cfgs[model_index].get("path", ""))
            cache_key: Optional[Tuple[str, str]] = None
            if self.reuse_prompt_perplexities_for_identical_models:
                digest = hashlib.md5()
                for prompt_text in model_prompts:
                    digest.update(str(prompt_text).encode("utf-8"))
                    digest.update(b"\x1e")
                cache_key = (path_key, digest.hexdigest())
                if cache_key in prompt_perplexity_cache:
                    all_perplexities[:, model_index] = prompt_perplexity_cache[cache_key]
                    reused_columns += 1
                    continue

            loaded, _ = load_models_from_config(
                [cfgs[model_index]],
                cache_kwargs=self._perplexity_load_cache_kwargs,
                share_identical_models=False,
            )
            wb = loaded[0]
            try:
                all_perplexities[:, model_index] = self._compute_prompt_perplexities_for_model(
                    model=wb,
                    prompts=model_prompts,
                    batch_size=batch_size,
                )
            finally:
                try:
                    del wb.model
                    del wb.tokenizer
                except Exception:
                    pass
                del loaded, wb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            computed_columns += 1
            if cache_key is not None:
                prompt_perplexity_cache[cache_key] = all_perplexities[:, model_index].copy()

        if self.reuse_prompt_perplexities_for_identical_models and reused_columns > 0:
            log.info(
                "Reused prompt-perplexity columns for %d/%d models (computed=%d) using identical model+prompt variants",
                reused_columns,
                num_models,
                computed_columns,
            )

        return all_perplexities

    def _compute_prompt_perplexities(self, dataset: Dataset) -> np.ndarray:
        """
        Compute prompt perplexities for all models.

        Returns:
            np.ndarray with shape [num_examples, num_models].
        """
        if self._should_use_sequential_perplexity_load():
            return self._compute_prompt_perplexities_sequential(dataset)

        num_examples = len(dataset.x)
        num_models = len(self.models)
        all_perplexities = np.empty((num_examples, num_models), dtype=np.float32)
        reused_columns = 0
        computed_columns = 0
        prompt_perplexity_cache: Dict[Tuple[int, str], np.ndarray] = {}

        for model_index, model in enumerate(self.models):
            if isinstance(model, str):
                raise RuntimeError(
                    "PackLLM prompt-perplexity wagering requires loaded model objects, "
                    f"but model at index {model_index} is a string path: {model}. "
                    "With more models than visible GPUs, pass model_configs_for_sequential_perplexity "
                    "from the training script so perplexities can be computed one model at a time."
                )

            model_prompts = get_model_specific_prompts(dataset, model_index=model_index)
            if len(model_prompts) != num_examples:
                raise ValueError(
                    "Prompt/label length mismatch while computing prompt perplexities. "
                    f"prompts={len(model_prompts)}, examples={num_examples}"
                )

            cache_key: Optional[Tuple[int, str]] = None
            if self.reuse_prompt_perplexities_for_identical_models:
                digest = hashlib.md5()
                for prompt_text in model_prompts:
                    digest.update(str(prompt_text).encode("utf-8"))
                    digest.update(b"\x1e")
                cache_key = (id(model.model), digest.hexdigest())
                if cache_key in prompt_perplexity_cache:
                    all_perplexities[:, model_index] = prompt_perplexity_cache[cache_key]
                    reused_columns += 1
                    continue

            all_perplexities[:, model_index] = self._compute_prompt_perplexities_for_model(
                model=model,
                prompts=model_prompts,
                batch_size=max(1, int(dataset.batch_size)),
            )
            computed_columns += 1
            if cache_key is not None:
                prompt_perplexity_cache[cache_key] = all_perplexities[:, model_index].copy()

        if self.reuse_prompt_perplexities_for_identical_models and reused_columns > 0:
            log.info(
                "Reused prompt-perplexity columns for %d/%d models (computed=%d) using identical model+prompt variants",
                reused_columns,
                num_models,
                computed_columns,
            )

        return all_perplexities

    def _unload_language_models_after_prompt_perplexities(self) -> None:
        """Free VRAM once precomputed perplexities make live models unnecessary."""
        import gc

        if not self.method_requires_model_perplexities:
            return
        if not any(isinstance(m, WhiteboxModel) for m in self.models):
            return

        new_models: List[Any] = []
        to_free_ids: set = set()
        to_free: List[WhiteboxModel] = []

        for m in self.models:
            if isinstance(m, WhiteboxModel):
                mp = getattr(m, "model_path", None) or ""
                new_models.append(str(mp) if mp else str(id(m)))
                mid = id(m)
                if mid not in to_free_ids:
                    to_free_ids.add(mid)
                    to_free.append(m)
            else:
                new_models.append(m)

        self.models = new_models

        for wb in to_free:
            try:
                if getattr(wb, "model", None) is not None:
                    del wb.model
                if getattr(wb, "tokenizer", None) is not None:
                    del wb.tokenizer
            except Exception:
                pass
            try:
                del wb
            except Exception:
                pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Unloaded language-model weights after prompt perplexity precompute.")

    def _prepare_model_perplexities(self) -> None:
        """Precompute train/validation prompt perplexities when required by method."""
        self.model_prompt_perplexities = None
        self.validation_model_prompt_perplexities = None

        if not self.method_requires_model_perplexities:
            return

        self.model_prompt_perplexities = self._compute_prompt_perplexities(self.combined_dataset)
        if self.validation_dataset is not None:
            self.validation_model_prompt_perplexities = self._compute_prompt_perplexities(
                self.validation_dataset
            )

        self._unload_language_models_after_prompt_perplexities()

        log.info(
            "Computed prompt perplexities for training method: train_shape=%s%s",
            None if self.model_prompt_perplexities is None else self.model_prompt_perplexities.shape,
            "" if self.validation_model_prompt_perplexities is None else f", val_shape={self.validation_model_prompt_perplexities.shape}",
        )

    def _get_wandb_run_step(self) -> Optional[int]:
        """Return current wandb run step if available and parseable."""
        if not self.wandb_logger:
            return None

        if hasattr(self.wandb_logger, 'run') and self.wandb_logger.run is not None:
            run = self.wandb_logger.run
            if hasattr(run, 'step') and run.step is not None:
                try:
                    return int(run.step)
                except (TypeError, ValueError):
                    return None

        return None

    def _advance_wandb_plot_step(self) -> int:
        """Advance plot logging step while staying monotonic with run.step."""
        next_step = self.current_step + 1
        run_step = self._get_wandb_run_step()
        if run_step is not None:
            next_step = max(next_step, run_step + 1)
        self.current_step = next_step
        return self.current_step

    def _log_wandb_plot(self, payload: Dict[str, Any]) -> None:
        """Log plot payload to wandb with a safe monotonically increasing step."""
        if not self.wandb_logger:
            return

        plot_step = self._advance_wandb_plot_step()
        if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
            self.wandb_logger.run.log(payload, step=plot_step, commit=True)
        else:
            self.wandb_logger.log(payload, step=plot_step, commit=True)

        # Keep local step aligned with wandb's internal run.step, which can advance
        # by one after commit=True logs.
        run_step = self._get_wandb_run_step()
        if run_step is not None:
            self.current_step = max(self.current_step, run_step)

    def _resolve_training_dataset_names(self) -> Tuple[List[str], List[str]]:
        """Return display names and slugified keys for training datasets.
        
        Returns:
            (display_names, slug_names): two lists aligned with self.datasets
        """
        display_names: List[str] = []
        if isinstance(self.metadata, dict):
            for key in ["training_datasets", "dataset_names", "datasets", "train_datasets"]:
                v = self.metadata.get(key)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    display_names = [str(x) for x in v][: len(self.datasets)]
                    break
                if isinstance(v, str) and len(self.datasets) == 1:
                    display_names = [v]
                    break
        if not display_names:
            inferred: List[str] = []
            for i, ds in enumerate(self.datasets):
                ds_name = getattr(ds, "name", None) or getattr(ds, "dataset_name", None) or getattr(ds, "path", None)
                inferred.append(str(ds_name) if ds_name else f"dataset_{i}")
            display_names = inferred[: len(self.datasets)]
        if len(display_names) != len(self.datasets):
            display_names = [f"dataset_{i}" for i in range(len(self.datasets))]

        # Slugify for wandb keys
        def slugify(name: str, fallback: str) -> str:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
            return slug if slug else fallback

        slug_names: List[str] = [slugify(display_names[i], f"dataset_{i}") for i in range(len(display_names))]
        return display_names, slug_names
        
    def _prepare_datasets(self):
        """Concatenate training datasets WITHOUT shuffling (after cache collection).

        If ``self.balance_training_datasets`` is True, each dataset is randomly
        subsampled (without replacement) to the minimum dataset size across
        ``self.datasets`` before concatenation. Otherwise, the full datasets are
        concatenated.

        Shuffling and train/validation split happen AFTER cache loading in
        ``_apply_shuffling()``.
        """
        if not self.datasets:
            self.combined_dataset = Dataset([], [], batch_size=8)
            self.labels = np.array([], dtype=np.int32)
            self.dataset_indices = np.array([], dtype=np.int32)
            self.example_local_indices = np.array([], dtype=np.int32)
            return

        dataset_sizes = [len(dataset.x) for dataset in self.datasets]
        total_unbalanced_size = int(np.sum(dataset_sizes))
        if total_unbalanced_size <= 0:
            raise ValueError(
                "Cannot build training set: all datasets are empty."
            )

        min_dataset_size = min(dataset_sizes)
        if self.balance_training_datasets and min_dataset_size <= 0:
            raise ValueError(
                "Cannot build balanced training set: at least one dataset is empty."
            )

        if self.balance_training_datasets and len(set(dataset_sizes)) > 1:
            log.info(
                "Balancing training datasets to %d samples each via random subsampling (dataset sizes: %s)",
                min_dataset_size,
                dataset_sizes,
            )
        elif not self.balance_training_datasets:
            log.info(
                "Using full training datasets without balancing (dataset sizes: %s)",
                dataset_sizes,
            )

        all_x = []
        all_y = []
        dataset_indices = []  # Track which dataset each example came from
        example_local_indices = []  # Track example index within each source dataset
        selected_global_indices = []
        global_offset = 0
        sampling_rng = np.random.RandomState(self.shuffle_seed) if self.balance_training_datasets else None
        
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_len = len(dataset.x)
            if self.balance_training_datasets:
                if dataset_len == min_dataset_size:
                    selected_local_indices = np.arange(min_dataset_size, dtype=np.int64)
                else:
                    selected_local_indices = np.sort(
                        sampling_rng.choice(
                            dataset_len,
                            size=min_dataset_size,
                            replace=False,
                        )
                    )
            else:
                selected_local_indices = np.arange(dataset_len, dtype=np.int64)

            all_x.extend(dataset.x[i] for i in selected_local_indices)
            all_y.extend(dataset.y[i] for i in selected_local_indices)
            # Track dataset index for each example
            dataset_indices.extend([dataset_idx] * len(selected_local_indices))
            example_local_indices.extend(selected_local_indices.tolist())
            selected_global_indices.extend((global_offset + selected_local_indices).tolist())
            global_offset += dataset_len

        selected_global_indices = np.array(selected_global_indices, dtype=np.int64)

        full_selection_indices = np.arange(total_unbalanced_size, dtype=np.int64)
        did_select_subset = not np.array_equal(selected_global_indices, full_selection_indices)

        if did_select_subset:
            # Keep cached logits aligned with dataset selection.
            if hasattr(self, "all_model_logits") and self.all_model_logits is not None:
                if self.all_model_logits.shape[1] != total_unbalanced_size:
                    raise RuntimeError(
                        f"Unexpected logits size before selection: got {self.all_model_logits.shape[1]}, "
                        f"expected {total_unbalanced_size}."
                    )
                self.all_model_logits = self.all_model_logits[:, selected_global_indices, :]

            # Keep cached hidden states aligned with dataset selection.
            if hasattr(self, "all_hidden_states") and self.all_hidden_states is not None:
                if isinstance(self.all_hidden_states, list):
                    selected_hidden_states = []
                    for model_idx, model_hidden_states in enumerate(self.all_hidden_states):
                        if model_hidden_states.shape[0] != total_unbalanced_size:
                            raise RuntimeError(
                                "Unexpected hidden states size before selection for model "
                                f"{model_idx}: got {model_hidden_states.shape[0]}, "
                                f"expected {total_unbalanced_size}."
                            )
                        selected_hidden_states.append(
                            model_hidden_states[selected_global_indices, ...]
                        )
                    self.all_hidden_states = selected_hidden_states
                elif self.all_hidden_states.ndim == 3:
                    if self.all_hidden_states.shape[1] != total_unbalanced_size:
                        raise RuntimeError(
                            "Unexpected hidden states size before selection: "
                            f"got {self.all_hidden_states.shape[1]}, "
                            f"expected {total_unbalanced_size}."
                        )
                    self.all_hidden_states = self.all_hidden_states[:, selected_global_indices, :]
                else:
                    if self.all_hidden_states.shape[0] != total_unbalanced_size:
                        raise RuntimeError(
                            "Unexpected hidden states size before selection: "
                            f"got {self.all_hidden_states.shape[0]}, "
                            f"expected {total_unbalanced_size}."
                        )
                    self.all_hidden_states = self.all_hidden_states[selected_global_indices, ...]
        
        # Convert labels to indices if needed
        labels = []
        for y in all_y:
            if isinstance(y, str):
                idx = self.option_tokens.index(y)
            else:
                idx = int(y)
            labels.append(idx)
        
        # Store unshuffled data (will be used for cache key generation)
        batch_size = self.datasets[0].batch_size if self.datasets else 8
        self.combined_dataset = Dataset(all_x, all_y, batch_size=batch_size)
        self.labels = np.array(labels, dtype=np.int32)
        self.dataset_indices = np.array(dataset_indices, dtype=np.int32)
        self.example_local_indices = np.array(example_local_indices, dtype=np.int32)

    def _get_router_questions_for_batch(
        self,
        base_questions: List[str],
        batch_start: int,
        batch_end: int,
        *,
        validation: bool,
    ) -> List[str]:
        """Return router questions for a batch, optionally using concatenated model prompts."""
        # When pubmedqa_strip_context is enabled for the wagering method, the router must only
        # see the dataset's prompt_without_context stream (verbatim). Do not concatenate
        # per-model prompts (which may include correct/wrong context under PubMedQA routing modes).
        if bool(getattr(self.wagering_method, "pubmedqa_strip_context", False)):
            return base_questions

        if not self.use_concatenated_prompt_context:
            return base_questions

        if validation:
            dataset_indices = getattr(self, "validation_dataset_indices", None)
            local_indices = getattr(self, "validation_example_local_indices", None)
        else:
            dataset_indices = getattr(self, "dataset_indices", None)
            local_indices = getattr(self, "example_local_indices", None)

        if dataset_indices is None or local_indices is None:
            return base_questions

        batch_dataset_indices = np.asarray(dataset_indices[batch_start:batch_end], dtype=np.int32)
        batch_local_indices = np.asarray(local_indices[batch_start:batch_end], dtype=np.int32)
        if (
            batch_dataset_indices.shape[0] != len(base_questions)
            or batch_local_indices.shape[0] != len(base_questions)
        ):
            return base_questions

        router_questions: List[str] = []
        for row_idx, fallback_question in enumerate(base_questions):
            dataset_idx = int(batch_dataset_indices[row_idx])
            local_idx = int(batch_local_indices[row_idx])
            if dataset_idx < 0 or dataset_idx >= len(self.datasets):
                router_questions.append(fallback_question)
                continue

            if dataset_idx not in self._router_concatenated_prompts_by_dataset:
                self._router_concatenated_prompts_by_dataset[dataset_idx] = get_concatenated_router_prompts(
                    self.datasets[dataset_idx],
                    num_models=len(self.models),
                )

            concat_prompts = self._router_concatenated_prompts_by_dataset[dataset_idx]
            if local_idx < 0 or local_idx >= len(concat_prompts):
                router_questions.append(fallback_question)
                continue

            router_questions.append(concat_prompts[local_idx])

        return router_questions

    def _get_router_questions_per_model_for_batch(
        self,
        base_questions: List[str],
        batch_start: int,
        batch_end: int,
        *,
        validation: bool,
    ) -> Optional[List[List[str]]]:
        """
        Return per-model prompt variants for mixed-context datasets (PubMedQA/RACE and
        CSV datasets configured with mixed_context_routing=pubmedqa).

        Output format: list length num_models; each element is a list of batch_size prompt strings.
        """
        if not bool(getattr(self.wagering_method, "expects_per_model_router_prompts", False)):
            return None

        if validation:
            dataset_indices = getattr(self, "validation_dataset_indices", None)
            local_indices = getattr(self, "validation_example_local_indices", None)
        else:
            dataset_indices = getattr(self, "dataset_indices", None)
            local_indices = getattr(self, "example_local_indices", None)

        if dataset_indices is None or local_indices is None:
            return None

        batch_dataset_indices = np.asarray(dataset_indices[batch_start:batch_end], dtype=np.int32)
        batch_local_indices = np.asarray(local_indices[batch_start:batch_end], dtype=np.int32)
        if (
            batch_dataset_indices.shape[0] != len(base_questions)
            or batch_local_indices.shape[0] != len(base_questions)
        ):
            return None

        num_models = len(self.models)
        if num_models <= 0:
            return None

        out: List[List[str]] = [[] for _ in range(num_models)]
        force_without_context = bool(getattr(self.wagering_method, "pubmedqa_strip_context", False))
        for row_idx, fallback_question in enumerate(base_questions):
            dataset_idx = int(batch_dataset_indices[row_idx])
            local_idx = int(batch_local_indices[row_idx])
            if dataset_idx < 0 or dataset_idx >= len(self.datasets):
                for mi in range(num_models):
                    out[mi].append(fallback_question)
                continue

            ds = self.datasets[dataset_idx]
            if _get_mixed_context_dataset_type(ds) is None:
                for mi in range(num_models):
                    out[mi].append(fallback_question)
                continue

            if force_without_context:
                # For mixed-context datasets, some prompt variants contain evidence/context in a non-PubMedQA format
                # (e.g. cluster_saturation_bayesX). When pubmedqa_strip_context is enabled for the wagering method,
                # force the router to see the without-context prompt for every model slot.
                without_ctx = getattr(ds, "pubmedqa_without_context_x", None)
                if isinstance(without_ctx, list) and 0 <= local_idx < len(without_ctx):
                    prompt = without_ctx[local_idx]
                else:
                    prompt = fallback_question
                for mi in range(num_models):
                    out[mi].append(prompt)
                continue

            if dataset_idx not in self._router_prompts_per_model_by_dataset:
                self._router_prompts_per_model_by_dataset[dataset_idx] = [
                    get_model_specific_prompts(ds, model_index=mi) for mi in range(num_models)
                ]

            per_model_lists = self._router_prompts_per_model_by_dataset[dataset_idx]
            if local_idx < 0 or local_idx >= len(per_model_lists[0]):
                for mi in range(num_models):
                    out[mi].append(fallback_question)
                continue

            for mi in range(num_models):
                out[mi].append(per_model_lists[mi][local_idx])

        return out

    def _apply_shuffling(self):
        """Apply shuffling to cached arrays and create train/validation splits.
        
        This is called AFTER cache loading so cache keys are based on unshuffled data.
        Shuffles:
        - Dataset (x, y, labels)
        - all_model_logits (if exists)
        - all_hidden_states (if exists)
        Then creates train/validation splits.
        """
        contiguous_tri_split = (
            len(self.datasets) == 1
            and bool(getattr(self.datasets[0], "source_tripartition_contiguous_train_val", False))
        )
        tri_boundary = int(
            getattr(self.datasets[0], "source_tripartition_train_val_boundary", 0) or 0
        )

        if not self.shuffle_data:
            # No shuffling requested - just create train/validation splits in original order
            log.debug("Shuffling disabled - using original order")
            indices = np.arange(len(self.combined_dataset.x))
        elif contiguous_tri_split and tri_boundary > 0:
            rng = np.random.RandomState(self.shuffle_seed)
            n = len(self.combined_dataset.x)
            idx_train = rng.permutation(tri_boundary)
            idx_val = tri_boundary + rng.permutation(max(n - tri_boundary, 0))
            indices = np.concatenate([idx_train, idx_val]) if idx_val.size else idx_train
            log.info(
                "Shuffling within shared-source train/val partitions only (boundary=%d of %d examples; seed=%d)",
                tri_boundary,
                n,
                int(self.shuffle_seed),
            )
        else:
            # Generate shuffle indices
            rng = np.random.RandomState(self.shuffle_seed)
            indices = np.arange(len(self.combined_dataset.x))
            rng.shuffle(indices)
            log.debug(f"Shuffled dataset with seed {self.shuffle_seed}")
        
        # Shuffle dataset
        shuffled_x = [self.combined_dataset.x[i] for i in indices]
        shuffled_y = [self.combined_dataset.y[i] for i in indices]
        shuffled_labels = self.labels[indices]
        shuffled_dataset_indices = self.dataset_indices[indices]
        shuffled_example_local_indices = self.example_local_indices[indices]
        
        # Shuffle cached logits if they exist
        if hasattr(self, 'all_model_logits') and self.all_model_logits is not None:
            # all_model_logits shape: [num_models, num_examples, num_options]
            # Shuffle along the num_examples dimension (axis=1)
            self.all_model_logits = self.all_model_logits[:, indices, :]
            log.debug("Shuffled cached logits")
        
        # Shuffle cached hidden states if they exist
        if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
            if isinstance(self.all_hidden_states, list):
                # List of arrays: shuffle each array
                self.all_hidden_states = [hs[indices, :] for hs in self.all_hidden_states]
            else:
                # Single array: [num_models, num_examples, hidden_dim] or [num_examples, hidden_dim]
                if self.all_hidden_states.ndim == 3:
                    self.all_hidden_states = self.all_hidden_states[:, indices, :]
                else:
                    self.all_hidden_states = self.all_hidden_states[indices, :]
            log.debug("Shuffled cached hidden states")
        
        # Create train/validation splits AFTER shuffling
        batch_size = self.combined_dataset.batch_size
        total_size = len(shuffled_x)
        
        log.debug(f"Creating train/validation split: validation_split_ratio={self.validation_split_ratio}, total_size={total_size}")
        
        if self.validation_split_ratio > 0 and self.validation_split_ratio < 1:
            if contiguous_tri_split and tri_boundary > 0:
                train_size = tri_boundary
                val_size = total_size - train_size
            else:
                val_size = int(total_size * self.validation_split_ratio)
                train_size = total_size - val_size

            # Split the shuffled data
            train_x = shuffled_x[:train_size]
            train_y = shuffled_y[:train_size]
            train_labels = shuffled_labels[:train_size]
            train_dataset_indices = shuffled_dataset_indices[:train_size]
            train_example_local_indices = shuffled_example_local_indices[:train_size]
            
            val_x = shuffled_x[train_size:]
            val_y = shuffled_y[train_size:]
            val_labels = shuffled_labels[train_size:]
            val_dataset_indices = shuffled_dataset_indices[train_size:]
            val_example_local_indices = shuffled_example_local_indices[train_size:]
            
            self.combined_dataset = Dataset(train_x, train_y, batch_size=batch_size)
            self.labels = np.array(train_labels, dtype=np.int32)
            self.dataset_indices = np.array(train_dataset_indices, dtype=np.int32)
            self.example_local_indices = np.array(train_example_local_indices, dtype=np.int32)
            
            self.validation_dataset = Dataset(val_x, val_y, batch_size=batch_size)
            self.validation_labels = np.array(val_labels, dtype=np.int32)
            self.validation_dataset_indices = np.array(val_dataset_indices, dtype=np.int32)
            self.validation_example_local_indices = np.array(val_example_local_indices, dtype=np.int32)
            
            log.debug(f"Created validation_dataset with {len(self.validation_dataset.x)} examples")
            
            # Split cached logits if they exist
            if hasattr(self, 'all_model_logits') and self.all_model_logits is not None:
                self.all_model_val_logits = self.all_model_logits[:, train_size:, :]
                self.all_model_logits = self.all_model_logits[:, :train_size, :]
                log.debug(f"Split logits: training={self.all_model_logits.shape}, validation={self.all_model_val_logits.shape if self.all_model_val_logits is not None else 'None'}")
            else:
                raise Exception("No all_model_logits to split for validation")
            
            # Split cached hidden states if they exist
            if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
                if isinstance(self.all_hidden_states, list):
                    self.all_val_hidden_states = [hs[train_size:, :] for hs in self.all_hidden_states]
                    self.all_hidden_states = [hs[:train_size, :] for hs in self.all_hidden_states]
                else:
                    if self.all_hidden_states.ndim == 3:
                        self.all_val_hidden_states = self.all_hidden_states[:, train_size:, :]
                        self.all_hidden_states = self.all_hidden_states[:, :train_size, :]
                    else:
                        self.all_val_hidden_states = self.all_hidden_states[train_size:, :]
                        self.all_hidden_states = self.all_hidden_states[:train_size, :]
            
            log.debug(f"Split dataset after shuffling: {train_size} train, {val_size} validation ({self.validation_split_ratio*100:.1f}% validation)")
        else:
            # No validation split - use all shuffled data for training
            self.combined_dataset = Dataset(shuffled_x, shuffled_y, batch_size=batch_size)
            self.labels = shuffled_labels
            self.dataset_indices = shuffled_dataset_indices
            self.validation_dataset = None
            self.validation_labels = None
            self.validation_dataset_indices = None
            self.validation_example_local_indices = None
            
            # No need to split cached arrays - they're already shuffled
            if hasattr(self, 'all_model_logits') and self.all_model_logits is not None:
                self.all_model_val_logits = None
            if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
                self.all_val_hidden_states = None
            
            log.debug(f"Shuffled dataset: {len(self.combined_dataset.x)} examples (no validation split)")
    
    def _compute_grouped_metrics(
        self,
        predictions: np.ndarray,
        probs: np.ndarray,
        labels: np.ndarray,
        dataset_indices: np.ndarray,
        example_local_indices: Optional[np.ndarray] = None,
        wagers_history: Optional[np.ndarray] = None,
        model_logits: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics grouped by original dataset.
        
        Args:
            predictions: Array of predictions
            probs: Array of probability distributions
            labels: Array of true labels
            dataset_indices: Array indicating which dataset each example came from
            example_local_indices: Optional local indices within each source dataset
            wagers_history: Optional array of wagers [num_examples, num_models]
            model_logits: Optional array of model logits [num_examples, num_models, num_options] for computing d_regret
            
        Returns:
            Dictionary mapping dataset index to metrics dictionary
        """
        grouped_metrics = {}
        
        for dataset_idx in range(len(self.datasets)):
            # Get mask for this dataset
            mask = dataset_indices == dataset_idx
            if not np.any(mask):
                continue
            
            # Extract data for this dataset
            dataset_preds = predictions[mask]
            dataset_probs = probs[mask]
            dataset_labels = labels[mask]
            
            # Compute accuracy
            accuracy = np.mean(dataset_preds == dataset_labels)
            
            # Compute NLL
            correct_class_probs = dataset_probs[np.arange(len(dataset_labels)), dataset_labels]
            nll = -np.mean(np.log(correct_class_probs + 1e-10))
            
            # Compute ECE
            ece = None
            try:
                ece_metric = ECE(n_bins=20)
                confidences = dataset_probs.max(axis=1)
                correctness = (dataset_preds == dataset_labels).astype(float)
                ece = ece_metric(confidences.tolist(), correctness.tolist())
            except Exception as e:
                ece = np.nan
            
            # Compute AUC
            auc = None
            max_probs = dataset_probs.max(axis=1)
            correctness = (dataset_preds == dataset_labels).astype(int)
            if len(np.unique(correctness)) >= 2:
                try:
                    auc = roc_auc_score(correctness, max_probs)
                except ValueError:
                    auc = np.nan
            else:
                auc = np.nan
            
            # Compute Dynamic Regret and Meta Metrics if model_logits and wagers provided
            d_regret = None
            brier_d_regret = None
            meta_acc = None
            meta_nll = None
            meta_auc = None
            kendall_tau = None
            best_model_mrr = None
            if model_logits is not None and wagers_history is not None:
                try:
                    dataset_model_logits = model_logits[mask]
                    dataset_wagers = wagers_history[mask]
                    dataset_obj = self.datasets[dataset_idx]
                    dataset_name = getattr(dataset_obj, "cache_dataset_name", None)
                    is_cluster_saturation = _is_cluster_saturation_dataset_name(dataset_name)
                    d_regret, best_expert_ids = compute_dynamic_regret(
                        dataset_model_logits, dataset_probs, dataset_labels
                    )
                    if is_cluster_saturation and hasattr(dataset_obj, "probabilistic_labels"):
                        if example_local_indices is None:
                            raise ValueError(
                                "example_local_indices is required for cluster_saturation grouped metrics"
                            )
                        gt_probs_all = np.asarray(dataset_obj.probabilistic_labels, dtype=np.float64)
                        dataset_local_indices = np.asarray(example_local_indices[mask], dtype=np.int64)
                        dataset_gt_probs = gt_probs_all[dataset_local_indices]
                        positive_option_index = _resolve_positive_option_index(
                            getattr(dataset_obj, "positive_label", None),
                            self.option_tokens,
                            dataset_probs.shape[1],
                        )
                        if positive_option_index is None:
                            raise ValueError(
                                "Could not resolve positive option index for cluster_saturation metrics"
                            )

                        num_opt = int(dataset_probs.shape[1])
                        pos_idx = int(positive_option_index)
                        neg_idx = 1 - pos_idx
                        y_soft = np.zeros((dataset_gt_probs.shape[0], num_opt), dtype=np.float64)
                        y_soft[:, pos_idx] = dataset_gt_probs
                        y_soft[:, neg_idx] = 1.0 - dataset_gt_probs
                        brier_d_regret = compute_brier_dynamic_regret(
                            dataset_model_logits,
                            dataset_probs,
                            dataset_labels,
                            gold_label_distribution=y_soft,
                        )
                        d_regret = brier_d_regret
                        model_kl_scores = _compute_model_bernoulli_kl_to_gt_scores(
                            dataset_model_logits,
                            dataset_gt_probs,
                            positive_option_index,
                        )
                        meta_metrics = compute_meta_metrics(
                            dataset_wagers,
                            best_expert_ids,
                            model_rank_scores=-model_kl_scores,
                            best_model_ids=np.argmin(model_kl_scores, axis=1),
                        )
                    else:
                        brier_d_regret = compute_brier_dynamic_regret(
                            dataset_model_logits, dataset_probs, dataset_labels
                        )
                        dataset_model_brier_scores = _compute_model_brier_scores(
                            dataset_model_logits, dataset_labels
                        )
                        meta_metrics = compute_meta_metrics(
                            dataset_wagers,
                            best_expert_ids,
                            dataset_model_brier_scores,
                        )
                    meta_acc = meta_metrics["meta_acc"]
                    meta_nll = meta_metrics["meta_nll"]
                    meta_auc = meta_metrics["meta_auc"]
                    kendall_tau = meta_metrics["kendall_tau"]
                    best_model_mrr = meta_metrics["best_model_mrr"]
                except Exception as e:
                    log.warning(f"Could not compute d_regret/meta metrics for dataset {dataset_idx}: {e}")
            
            grouped_metrics[dataset_idx] = {
                "accuracy": float(accuracy),
                "nll": float(nll),
                "ece": float(ece) if not np.isnan(ece) else None,
                "auc": float(auc) if not np.isnan(auc) else None,
                "d_regret": float(d_regret) if d_regret is not None and not np.isnan(d_regret) else None,
                "brier_d_regret": float(brier_d_regret) if brier_d_regret is not None and not np.isnan(brier_d_regret) else None,
                "meta_acc": float(meta_acc) if meta_acc is not None and not np.isnan(meta_acc) else None,
                "meta_nll": float(meta_nll) if meta_nll is not None and not np.isnan(meta_nll) else None,
                "meta_auc": float(meta_auc) if meta_auc is not None and not np.isnan(meta_auc) else None,
                "kendall_tau": float(kendall_tau) if kendall_tau is not None and not np.isnan(kendall_tau) else None,
                "best_model_mrr": float(best_model_mrr) if best_model_mrr is not None and not np.isnan(best_model_mrr) else None,
                "num_examples": int(np.sum(mask)),
            }
            
            # Add average wagers if provided
            if wagers_history is not None:
                dataset_wagers = wagers_history[mask]
                for model_idx in range(dataset_wagers.shape[1]):
                    grouped_metrics[dataset_idx][f"avg_wager_model_{model_idx}"] = float(np.mean(dataset_wagers[:, model_idx]))
        
        return grouped_metrics
    
    def _compute_running_averages(self) -> Dict[str, float]:
        """
        Compute running averages over the last N batches stored in buffer.
        
        Returns:
            Dictionary with running average metrics
        """
        if len(self.batch_metrics_buffer) == 0:
            return {}
        
        # Collect all metric keys from all batches
        all_keys = set()
        for batch_metrics in self.batch_metrics_buffer:
            all_keys.update(batch_metrics.keys())
        
        # Compute averages for each metric
        running_avgs = {}
        for key in all_keys:
            values = []
            for batch_metrics in self.batch_metrics_buffer:
                if key in batch_metrics:
                    values.append(batch_metrics[key])
            
            if len(values) > 0:
                running_avgs[key] = float(np.mean(values))
        
        return running_avgs

    def _compute_batch_max_nash_gap(self, nash_gap: Any) -> Optional[float]:
        """Compute a scalar max-nash-gap statistic for batch-level early stopping."""
        if nash_gap is None:
            return None

        nash_gap_arr = np.asarray(nash_gap)
        if nash_gap_arr.size == 0:
            return None

        # Treat axis 0 as batch axis; if input is 1D this promotes it to a
        # single-batch 2D array so axis=0 still works consistently.
        nash_gap_arr = np.atleast_2d(nash_gap_arr)
        per_model_nash_gap = np.mean(nash_gap_arr, axis=0)
        max_nash_gap = float(np.max(per_model_nash_gap))

        if not np.isfinite(max_nash_gap):
            raise ValueError(f"Non-finite max nash gap computed: {max_nash_gap} (nash_gap input: {nash_gap})")

        return max_nash_gap

    def _load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint state for exact training continuation."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint format at {checkpoint_path}")

        method_state = checkpoint.get("wagering_method_state")
        if method_state is None:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} is missing 'wagering_method_state' and cannot be resumed"
            )

        self.wagering_method.load_state_dict(method_state)

        loaded_epoch = int(checkpoint.get("epoch", -1))
        self.start_epoch = max(0, loaded_epoch + 1)
        self.current_step = int(checkpoint.get("current_step", self.current_step))

        if "best_d_regret" in checkpoint:
            self.best_d_regret = float(checkpoint["best_d_regret"])
        if "best_brier_d_regret" in checkpoint:
            self.best_brier_d_regret = float(checkpoint["best_brier_d_regret"])
        if "best_kl_to_gold" in checkpoint:
            self.best_kl_to_gold = float(checkpoint["best_kl_to_gold"])
        if "best_nash_gap" in checkpoint:
            self.best_nash_gap = float(checkpoint["best_nash_gap"])
        if "best_batch_nash_gap" in checkpoint:
            self.best_batch_nash_gap = float(checkpoint["best_batch_nash_gap"])
        if "best_batch_brier_d_regret" in checkpoint:
            self.best_batch_brier_d_regret = float(checkpoint["best_batch_brier_d_regret"])
        if "epochs_since_improvement" in checkpoint:
            self.epochs_since_improvement = int(checkpoint["epochs_since_improvement"])
        if "batches_since_improvement" in checkpoint:
            self.batches_since_improvement = int(checkpoint["batches_since_improvement"])
        if "best_epoch" in checkpoint and checkpoint["best_epoch"] is not None:
            self.best_epoch = int(checkpoint["best_epoch"])
        if "best_batch_step" in checkpoint and checkpoint["best_batch_step"] is not None:
            self.best_batch_step = int(checkpoint["best_batch_step"])

        log.info(
            "Resumed trainer state from %s (loaded_epoch=%d, resume_epoch=%d, current_step=%d)",
            checkpoint_path,
            loaded_epoch,
            self.start_epoch,
            self.current_step,
        )
    
    def _plot_validation_wagers_by_dataset(
        self,
        val_wagers: np.ndarray,
        val_results: Dict[str, Any],
    ):
        """
        Plot average validation wagers grouped by dataset.
        
        Args:
            val_wagers: np.ndarray of shape [num_val_examples, num_models] with wager values
            val_results: Dictionary containing 'dataset_indices' for grouping
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if "dataset_indices" not in val_results or self.checkpoint_dir is None:
            return
        
        dataset_indices = val_results["dataset_indices"]
        num_datasets = len(self.datasets)
        num_models = val_wagers.shape[1]
        
        # Get model names
        model_names: List[str] = []
        if isinstance(self.metadata, dict) and "models" in self.metadata:
            raw_names = self.metadata["models"]
            if isinstance(raw_names, (list, tuple)):
                model_names = [str(name) for name in raw_names][:num_models]
        
        if len(model_names) != num_models and getattr(self, "models", None):
            inferred_names: List[str] = []
            for i, model in enumerate(self.models):
                name = getattr(model, "model_path", None)
                if not name:
                    name = getattr(model, "model_name", None)
                if not name:
                    name = f"Model {i+1}"
                inferred_names.append(str(name))
            model_names = inferred_names[:num_models]
        
        if len(model_names) != num_models:
            model_names = [f"Model {i+1}" for i in range(num_models)]
        
        # Get dataset names
        def _resolve_validation_dataset_names() -> List[str]:
            names: List[str] = []
            if isinstance(self.metadata, dict):
                for key in ["training_datasets", "dataset_names", "datasets", "train_datasets"]:
                    v = self.metadata.get(key)
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        names = [str(x) for x in v][: len(self.datasets)]
                        break
                    if isinstance(v, str) and len(self.datasets) == 1:
                        names = [v]
                        break
            if not names:
                inferred = []
                for i, ds in enumerate(self.datasets):
                    ds_name = getattr(ds, "name", None) or getattr(ds, "dataset_name", None) or getattr(ds, "path", None)
                    inferred.append(str(ds_name) if ds_name else f"dataset_{i}")
                names = inferred[: len(self.datasets)]
            if len(names) != len(self.datasets):
                names = [f"dataset_{i}" for i in range(len(self.datasets))]
            return names
        
        dataset_names = _resolve_validation_dataset_names()
        
        # Plot: Average wagers per dataset (bar plot)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        x = np.arange(num_datasets)
        width = 0.8 / num_models
        
        for i in range(num_models):
            avg_wagers = []
            for dataset_idx in range(num_datasets):
                mask = dataset_indices == dataset_idx
                if np.any(mask):
                    avg_wager = np.mean(val_wagers[mask, i])
                else:
                    avg_wager = 0.0
                avg_wagers.append(avg_wager)
            
            ax.bar(x + i * width, avg_wagers, width, label=model_names[i], alpha=0.8)
        
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        ax.set_title("Average Wagers by Dataset (Validation)", fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels(dataset_names, rotation=20, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if self.checkpoint_dir:
            avg_save_path = self.checkpoint_dir / "average_wagers_by_dataset_validation.png"
            plt.savefig(avg_save_path, dpi=150, bbox_inches='tight')
            log.debug(f"Saved average validation wagers by dataset plot to {avg_save_path}")
            
            if self.wandb_logger:
                import wandb
                self._log_wandb_plot({"wagers_plot/average_by_dataset/val": wandb.Image(str(avg_save_path))})
        
        plt.close()

    def _validate_non_context_identical_model_scores(self, tolerance: float = 1e-6) -> None:
        """
        Validate that duplicated model paths have identical non-context scores.

        For mixed-context datasets (PubMedQA/RACE), exactly one model receives
        context per example. If multiple ensemble slots point to the same model
        path, any pair of those slots that are both non-context on an example
        should produce identical per-example scores.
        """
        if self.validation_dataset is None:
            return
        if not hasattr(self, "all_model_val_logits") or self.all_model_val_logits is None:
            return
        if not hasattr(self, "validation_labels") or self.validation_labels is None:
            return

        assignments = None
        assignment_attr = None
        for candidate in ("pubmedqa_context_assignment_by_example", "race_context_assignment_by_example"):
            value = getattr(self.validation_dataset, candidate, None)
            if isinstance(value, list):
                assignments = np.asarray(value, dtype=np.int32)
                assignment_attr = candidate
                break
        if assignments is None:
            return

        num_examples = len(self.validation_dataset.x)
        if assignments.shape[0] != num_examples:
            raise RuntimeError(
                f"Mixed-context assignment length mismatch for validation dataset: "
                f"{assignment_attr}={assignments.shape[0]}, num_examples={num_examples}"
            )

        model_paths: List[str] = []
        for model in self.models:
            if isinstance(model, str):
                model_paths.append(str(model))
            else:
                model_path = getattr(model, "model_path", None)
                model_paths.append(str(model_path) if model_path else str(model))

        path_to_indices: Dict[str, List[int]] = {}
        for model_idx, model_path in enumerate(model_paths):
            path_to_indices.setdefault(model_path, []).append(model_idx)
        duplicate_groups = {path: indices for path, indices in path_to_indices.items() if len(indices) > 1}
        if not duplicate_groups:
            return

        val_model_logits = np.transpose(self.all_model_val_logits, (1, 0, 2))
        model_scores = 0.5 * (2.0 - _compute_model_brier_scores(val_model_logits, self.validation_labels))

        for model_path, slot_indices in duplicate_groups.items():
            for left_pos in range(len(slot_indices)):
                left_idx = slot_indices[left_pos]
                for right_idx in slot_indices[left_pos + 1 :]:
                    non_context_mask = (assignments != left_idx) & (assignments != right_idx)
                    if not np.any(non_context_mask):
                        continue

                    score_diffs = np.abs(
                        model_scores[non_context_mask, left_idx] - model_scores[non_context_mask, right_idx]
                    )
                    max_abs_diff = float(np.max(score_diffs))
                    if max_abs_diff > float(tolerance):
                        masked_indices = np.where(non_context_mask)[0]
                        local_worst = int(np.argmax(score_diffs))
                        example_idx = int(masked_indices[local_worst])
                        left_score = float(model_scores[example_idx, left_idx])
                        right_score = float(model_scores[example_idx, right_idx])
                        raise RuntimeError(
                            "Non-context identical-model score mismatch detected: "
                            f"model_path={model_path}, slot_pair=({left_idx},{right_idx}), "
                            f"example_idx={example_idx}, left_score={left_score:.8f}, "
                            f"right_score={right_score:.8f}, max_abs_diff={max_abs_diff:.8e}, "
                            f"tolerance={float(tolerance):.1e}."
                        )
    
    def _evaluate_validation(self) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Evaluate the wagering method on the validation set using batch processing.
        
        Returns:
            Tuple of:
                - metrics dictionary (accuracy, nll, ece, auc, ...)
                - val_nash_gaps averaged across validation examples per model (or None)
                - val_score_diffs per validation sample and model (or None)
                - val_wagers per validation sample and model
        """
        # Debug: Check validation state
        has_val_dataset = self.validation_dataset is not None
        has_val_logits = hasattr(self, 'all_model_val_logits') and self.all_model_val_logits is not None
        
        log.debug(f"_evaluate_validation state: validation_dataset={has_val_dataset}, all_model_val_logits={has_val_logits}")
        
        if not has_val_dataset:
            raise RuntimeError("No validation_dataset set - cannot evaluate validation metrics")
        
        if not has_val_logits:
            raise RuntimeError("No all_model_val_logits set - cannot evaluate validation metrics. This may happen if no validation split was configured.")
        
        # log.info("Evaluating on validation set...")
        
        # Set wagering method to eval mode (no gradient updates)
        self.wagering_method.eval_mode()
        self._validate_non_context_identical_model_scores()
        
        val_predictions = []
        val_probs = []
        val_wagers = []  # Track wagers for each example
        num_val_examples = len(self.validation_dataset.x)
        val_nash_gaps = np.zeros((num_val_examples, len(self.models)))  # Track Nash gaps if provided by wagering method
        val_score_diffs = np.zeros((num_val_examples, len(self.models)))  # Track score differences if provided by wagering method
        val_sigmoid_wagers = np.zeros((num_val_examples, len(self.models)))  # Track sigmoid wagers if provided by wagering method
        eval_batch_size = self.batch_size  # Process validation in batches
        
        for batch_start in range(0, num_val_examples, eval_batch_size):
            batch_end = min(batch_start + eval_batch_size, num_val_examples)
            batch_size_actual = batch_end - batch_start
            
            # Get batch of logits
            batch_logits = self.all_model_val_logits[:, batch_start:batch_end, :]  # [num_models, batch_size, num_options]
            batch_logits_transposed = np.transpose(batch_logits, (1, 0, 2))  # [batch_size, num_models, num_options]
            batch_labels = self.validation_labels[batch_start:batch_end]  # [batch_size]
            
            # Get questions for batch (for wagering methods that need them)
            batch_questions = self.validation_dataset.x[batch_start:batch_end]  # List of question strings
            batch_questions = self._get_router_questions_for_batch(
                batch_questions,
                batch_start,
                batch_end,
                validation=True,
            )
            batch_questions_per_model = self._get_router_questions_per_model_for_batch(
                batch_questions,
                batch_start,
                batch_end,
                validation=True,
            )
            
            # Get hidden states for batch if available
            batch_hidden_states = None
            if hasattr(self, 'all_val_hidden_states') and self.all_val_hidden_states is not None:
                if isinstance(self.all_val_hidden_states, list):
                    batch_hidden_states = []
                    for i in range(len(self.all_val_hidden_states)):
                        model_hs = self.all_val_hidden_states[i][batch_start:batch_end, :]
                        batch_hidden_states.append(model_hs)
                else:
                    batch_hidden_states_array = self.all_val_hidden_states[:, batch_start:batch_end, :]
                    # Convert to list of [num_models] arrays, each [batch_size, hidden_dim]
                    batch_hidden_states = [batch_hidden_states_array[i, :, :] for i in range(batch_hidden_states_array.shape[0])]
            
            # Compute wagers for batch
                # Variable hidden dimensions per model - use batch heterogeneous processing
            wagering_kwargs = {
                "model_logits": batch_logits_transposed,
                "gold_label": batch_labels,
                "hidden_states_list": batch_hidden_states,
                "questions": batch_questions,
            }
            if batch_questions_per_model is not None:
                wagering_kwargs["questions_per_model"] = batch_questions_per_model
            if self.method_requires_model_perplexities:
                if self.validation_model_prompt_perplexities is None:
                    raise RuntimeError(
                        "Wagering method requires model_perplexities but validation perplexities are unavailable"
                    )
                wagering_kwargs["model_perplexities"] = self.validation_model_prompt_perplexities[
                    batch_start:batch_end
                ]

            res_dict = self.wagering_method.compute_wagers(**wagering_kwargs)  # [batch_size, num_models]
            batch_wagers = res_dict["wagers"]  # [batch_size, num_models]
            batch_sigmoid_wagers = res_dict.get("sigmoid_wagers", None)  # [batch_size, num_models]
            batch_nash_gap = res_dict.get("nash_gap", None)
            batch_score_diff = res_dict.get("score_diff", None)
            # Aggregate predictions for batch
            batch_aggregated_log_probs, batch_aggregated_probs = self.aggregation_function.aggregate(
                batch_logits_transposed, batch_wagers
            )  # [batch_size, num_options] each
            
            batch_predictions = np.argmax(batch_aggregated_probs, axis=1)  # [batch_size]
            if batch_nash_gap is not None and val_nash_gaps is not None:
                val_nash_gaps[batch_start:batch_end] = batch_nash_gap
            else:
                val_nash_gaps = None
            if batch_score_diff is not None and val_score_diffs is not None:
                val_score_diffs[batch_start:batch_end] = batch_score_diff
            else:
                val_score_diffs = None
            if batch_sigmoid_wagers is not None and val_sigmoid_wagers is not None:
                val_sigmoid_wagers[batch_start:batch_end] = batch_sigmoid_wagers
            else:
                val_sigmoid_wagers = None
            val_predictions.extend(batch_predictions.tolist())
            val_probs.extend(batch_aggregated_probs.tolist())
            val_wagers.extend(batch_wagers.tolist())
        
        # Convert to arrays
        val_predictions = np.array(val_predictions, dtype=np.int32)
        val_probs = np.stack(val_probs, axis=0)
        val_wagers = np.stack(val_wagers, axis=0)  # [num_val_examples, num_models]

        if val_sigmoid_wagers is not None:
            val_sigmoid_wagers = np.asarray(val_sigmoid_wagers, dtype=np.float32)
        if val_nash_gaps is not None:
            val_nash_gaps = np.mean(val_nash_gaps, axis=0)  # Average Nash gap per model over validation set
        if val_score_diffs is not None:
            val_score_diffs = np.asarray(val_score_diffs, dtype=np.float32)
        # Compute metrics
        val_accuracy = np.mean(val_predictions == self.validation_labels)
        
        # Compute NLL
        correct_class_probs = val_probs[np.arange(len(self.validation_labels)), self.validation_labels]
        val_nll = -np.mean(np.log(correct_class_probs + 1e-10))
        
        # Compute ECE
        val_ece = None
        try:
            ece_metric = ECE(n_bins=20)
            confidences = val_probs.max(axis=1)
            correctness = (val_predictions == self.validation_labels).astype(float)
            val_ece = ece_metric(confidences.tolist(), correctness.tolist())
        except Exception as e:
            log.warning(f"Could not compute validation ECE: {e}")
            val_ece = np.nan
        
        # Compute AUC
        val_auc = None
        max_probs = val_probs.max(axis=1)
        correctness = (val_predictions == self.validation_labels).astype(int)
        if len(np.unique(correctness)) >= 2:
            try:
                val_auc = roc_auc_score(correctness, max_probs)
            except ValueError:
                log.warning("Could not compute validation AUC (all predictions same class)")
                val_auc = np.nan
        else:
            val_auc = np.nan
        
        # Compute Dynamic Regret and Meta Metrics
        val_d_regret = None
        val_brier_d_regret = None
        val_meta_acc = None
        val_meta_nll = None
        val_meta_auc = None
        val_kendall_tau = None
        val_best_model_mrr = None
        try:
            # Get validation model logits in the right format [num_examples, num_models, num_options]
            val_model_logits = np.transpose(self.all_model_val_logits, (1, 0, 2))
            val_d_regret, best_expert_ids = compute_dynamic_regret(
                val_model_logits, val_probs, self.validation_labels
            )
            val_gold_dist = _build_gold_label_distribution_for_rows(
                self.validation_labels,
                self.validation_dataset_indices,
                self.validation_example_local_indices,
                self.datasets,
                self.option_tokens,
                int(val_probs.shape[1]),
            )
            val_soft_label_mask = None
            try:
                ds_ix = np.asarray(self.validation_dataset_indices, dtype=np.int32)
                soft_mask = np.zeros((int(ds_ix.shape[0]),), dtype=bool)
                for dataset_idx in np.unique(ds_ix).tolist():
                    ds_idx = int(dataset_idx)
                    if ds_idx < 0 or ds_idx >= len(self.datasets):
                        continue
                    ds = self.datasets[ds_idx]
                    dataset_name = getattr(ds, "cache_dataset_name", None)
                    if not _is_cluster_saturation_dataset_name(dataset_name):
                        continue
                    if not hasattr(ds, "probabilistic_labels"):
                        continue
                    soft_mask |= ds_ix == ds_idx
                if np.any(soft_mask):
                    val_soft_label_mask = soft_mask
            except Exception:
                val_soft_label_mask = None

            val_kl_to_gold = None
            try:
                if val_soft_label_mask is not None:
                    val_kl_to_gold = _compute_mean_kl_to_gold_distribution(
                        val_gold_dist,
                        val_probs,
                        mask=val_soft_label_mask,
                    )
            except Exception as e:
                log.warning(f"Could not compute validation kl_to_gold: {e}")
                val_kl_to_gold = None
            val_brier_d_regret = compute_brier_dynamic_regret(
                val_model_logits,
                val_probs,
                self.validation_labels,
                gold_label_distribution=val_gold_dist,
            )
            val_model_brier_scores = _compute_model_brier_scores(
                val_model_logits,
                self.validation_labels,
            )
            meta_metrics = compute_meta_metrics(
                val_wagers,
                best_expert_ids,
                val_model_brier_scores,
            )
            val_meta_acc = meta_metrics["meta_acc"]
            val_meta_nll = meta_metrics["meta_nll"]
            val_meta_auc = meta_metrics["meta_auc"]
            val_kendall_tau = meta_metrics["kendall_tau"]
            val_best_model_mrr = meta_metrics["best_model_mrr"]
        except Exception as e:
            log.warning(f"Could not compute validation d_regret/meta metrics: {e}")
        
        # Set back to train mode
        self.wagering_method.train_mode()
        
        metrics = {
            "accuracy": val_accuracy,
            "nll": val_nll,
            "ece": val_ece if val_ece is not None and not np.isnan(val_ece) else None,
            "auc": val_auc if val_auc is not None and not np.isnan(val_auc) else None,
            "d_regret": val_d_regret if val_d_regret is not None and not np.isnan(val_d_regret) else None,
            "brier_d_regret": val_brier_d_regret if val_brier_d_regret is not None and not np.isnan(val_brier_d_regret) else None,
            "kl_to_gold": val_kl_to_gold if val_kl_to_gold is not None and not np.isnan(val_kl_to_gold) else None,
            "meta_acc": val_meta_acc if val_meta_acc is not None and not np.isnan(val_meta_acc) else None,
            "meta_nll": val_meta_nll if val_meta_nll is not None and not np.isnan(val_meta_nll) else None,
            "meta_auc": val_meta_auc if val_meta_auc is not None and not np.isnan(val_meta_auc) else None,
            "kendall_tau": val_kendall_tau if val_kendall_tau is not None and not np.isnan(val_kendall_tau) else None,
            "best_model_mrr": val_best_model_mrr if val_best_model_mrr is not None and not np.isnan(val_best_model_mrr) else None,
        }
        
        ece_str = f"{val_ece:.4f}" if val_ece is not None and not np.isnan(val_ece) else 'N/A'
        auc_str = f"{val_auc:.4f}" if val_auc is not None and not np.isnan(val_auc) else 'N/A'
        d_regret_str = f"{val_d_regret:.4f}" if val_d_regret is not None and not np.isnan(val_d_regret) else 'N/A'
        brier_d_regret_str = f"{val_brier_d_regret:.4f}" if val_brier_d_regret is not None and not np.isnan(val_brier_d_regret) else 'N/A'
        meta_acc_str = f"{val_meta_acc:.4f}" if val_meta_acc is not None and not np.isnan(val_meta_acc) else 'N/A'
        kendall_tau_str = f"{val_kendall_tau:.4f}" if val_kendall_tau is not None and not np.isnan(val_kendall_tau) else 'N/A'
        best_model_mrr_str = f"{val_best_model_mrr:.4f}" if val_best_model_mrr is not None and not np.isnan(val_best_model_mrr) else 'N/A'
        # log.info(f"Validation metrics: accuracy={val_accuracy:.4f}, nll={val_nll:.4f}, ece={ece_str}, auc={auc_str}, "
        #        f"d_regret={d_regret_str}, brier_d_regret={brier_d_regret_str}, meta_acc={meta_acc_str}, "
        #        f"kendall_tau={kendall_tau_str}, best_model_mrr={best_model_mrr_str}")
        
        # Compute grouped metrics by dataset
        if hasattr(self, 'validation_dataset_indices') and self.validation_dataset_indices is not None:
            # Transpose validation logits to [num_examples, num_models, num_options]
            val_model_logits_transposed = np.transpose(self.all_model_val_logits, (1, 0, 2)) if self.all_model_val_logits is not None else None
            grouped_metrics = self._compute_grouped_metrics(
                val_predictions,
                val_probs,
                self.validation_labels,
                self.validation_dataset_indices,
                self.validation_example_local_indices,
                val_wagers,
                val_model_logits_transposed,
            )
            metrics["grouped"] = grouped_metrics
            
            # Log grouped metrics
            # for dataset_idx, dataset_metrics in grouped_metrics.items():
            #     dataset_name = f"dataset_{dataset_idx}"
                # log.info(f"Validation metrics for {dataset_name}: accuracy={dataset_metrics['accuracy']:.4f}, "
                #         f"nll={dataset_metrics['nll']:.4f}, num_examples={dataset_metrics['num_examples']}")
            
            # Plot validation wagers by dataset
            if self.enable_artifact_outputs:
                val_results = {
                    "dataset_indices": self.validation_dataset_indices,
                }
                self._plot_validation_wagers_by_dataset(val_wagers, val_results)
        
        return metrics, val_nash_gaps, val_score_diffs, val_wagers, val_sigmoid_wagers

    def _collect_validation_plot_arrays(
        self,
        max_examples: int = 1000,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Collect validation wagers and score_diff arrays for visualization only.

        Uses up to ``max_examples`` samples from the validation set (or all
        validation samples if fewer are available).
        """
        has_val_dataset = self.validation_dataset is not None
        has_val_logits = hasattr(self, 'all_model_val_logits') and self.all_model_val_logits is not None
        if not has_val_dataset or not has_val_logits:
            return None

        num_val_examples = len(self.validation_dataset.x)
        if num_val_examples <= 0:
            return None

        num_plot_examples = min(int(max_examples), num_val_examples)
        if num_plot_examples <= 0:
            return None

        self.wagering_method.eval_mode()
        plot_wagers_chunks: List[np.ndarray] = []
        plot_score_diff_chunks: List[np.ndarray] = []
        plot_brier_chunks: List[np.ndarray] = []
        optional_plot_chunks: Dict[str, List[np.ndarray]] = {
            "estimated_score_diff": [],
            "scores": [],
            "estimated_score": [],
            "average_scores": [],
            "estimated_average_scores": [],
        }
        optional_enabled = {k: True for k in optional_plot_chunks.keys()}
        eval_batch_size = self.batch_size

        try:
            for batch_start in range(0, num_plot_examples, eval_batch_size):
                batch_end = min(batch_start + eval_batch_size, num_plot_examples)

                batch_logits = self.all_model_val_logits[:, batch_start:batch_end, :]
                batch_logits_transposed = np.transpose(batch_logits, (1, 0, 2))
                batch_labels = self.validation_labels[batch_start:batch_end]
                batch_questions = self.validation_dataset.x[batch_start:batch_end]
                batch_questions = self._get_router_questions_for_batch(
                    batch_questions,
                    batch_start,
                    batch_end,
                    validation=True,
                )
                batch_questions_per_model = self._get_router_questions_per_model_for_batch(
                    batch_questions,
                    batch_start,
                    batch_end,
                    validation=True,
                )

                batch_hidden_states = None
                if hasattr(self, 'all_val_hidden_states') and self.all_val_hidden_states is not None:
                    if isinstance(self.all_val_hidden_states, list):
                        batch_hidden_states = []
                        for i in range(len(self.all_val_hidden_states)):
                            model_hs = self.all_val_hidden_states[i][batch_start:batch_end, :]
                            batch_hidden_states.append(model_hs)
                    else:
                        batch_hidden_states_array = self.all_val_hidden_states[:, batch_start:batch_end, :]
                        batch_hidden_states = [
                            batch_hidden_states_array[i, :, :]
                            for i in range(batch_hidden_states_array.shape[0])
                        ]

                wagering_kwargs = {
                    "model_logits": batch_logits_transposed,
                    "gold_label": batch_labels,
                    "hidden_states_list": batch_hidden_states,
                    "questions": batch_questions,
                }
                if batch_questions_per_model is not None:
                    wagering_kwargs["questions_per_model"] = batch_questions_per_model
                if self.method_requires_model_perplexities:
                    if self.validation_model_prompt_perplexities is None:
                        raise RuntimeError(
                            "Wagering method requires model_perplexities but validation perplexities are unavailable"
                        )
                    wagering_kwargs["model_perplexities"] = self.validation_model_prompt_perplexities[
                        batch_start:batch_end
                    ]

                res_dict = self.wagering_method.compute_wagers(**wagering_kwargs)
                batch_score_diff = res_dict.get("score_diff", None)
                if batch_score_diff is None:
                    return None

                batch_sigmoid_wagers = res_dict.get("sigmoid_wagers", None)
                batch_wagers = res_dict.get("wagers", None)
                batch_plot_wagers = batch_sigmoid_wagers if batch_sigmoid_wagers is not None else batch_wagers
                if batch_plot_wagers is None:
                    return None

                plot_wagers_chunks.append(np.asarray(batch_plot_wagers, dtype=np.float32))
                plot_score_diff_chunks.append(np.asarray(batch_score_diff, dtype=np.float32))
                plot_brier_chunks.append(
                    np.asarray(_compute_model_brier_scores(batch_logits_transposed, batch_labels), dtype=np.float32)
                )

                for key in optional_plot_chunks.keys():
                    if not optional_enabled[key]:
                        continue
                    batch_values = res_dict.get(key, None)
                    if batch_values is None:
                        optional_enabled[key] = False
                        optional_plot_chunks[key] = []
                        continue
                    optional_plot_chunks[key].append(np.asarray(batch_values, dtype=np.float32))
        finally:
            self.wagering_method.train_mode()

        if not plot_wagers_chunks or not plot_score_diff_chunks:
            return None

        result = {
            "wagers": np.vstack(plot_wagers_chunks),
            "score_diff": np.vstack(plot_score_diff_chunks),
            "model_brier_scores": np.vstack(plot_brier_chunks) if plot_brier_chunks else None,
        }
        for key, chunks in optional_plot_chunks.items():
            if optional_enabled[key] and chunks:
                result[key] = np.vstack(chunks)

        # Context assignment is only used to color points gray vs colored; do not mask points out.
        context_assignment_mask, context_assignment_kind = self._get_validation_context_assignment_mask(
            num_examples=num_plot_examples,
            num_models_total=int(result["wagers"].shape[1]),
            dataset_indices=np.asarray(getattr(self, "validation_dataset_indices", None))[:num_plot_examples]
            if getattr(self, "validation_dataset_indices", None) is not None
            else None,
            local_indices=np.asarray(getattr(self, "validation_example_local_indices", None))[:num_plot_examples]
            if getattr(self, "validation_example_local_indices", None) is not None
            else None,
        )
        if context_assignment_mask is not None:
            result["context_assignment_mask"] = context_assignment_mask
            if context_assignment_kind is not None:
                result["context_assignment_kind"] = np.asarray([context_assignment_kind])

        return result
    
    def _collect_logits(self):
        """
        Collect logits AND hidden states from all models per dataset (no combined dataset cache).
        
        Uses the combined function to collect both logits and hidden states in a single forward pass,
        reducing forward passes from 2 to 1 per model.
        
        Uses shared cache to avoid recomputing logits and hidden states for the same models and datasets
        across different wagering methods. This is the default behavior since LLMs are not updated.
        
        Models are assigned to different GPUs (cuda:0, cuda:1, etc.) for parallel execution.
        
        Note: Validation split happens AFTER cache loading in _apply_shuffling(), so this
        only collects logits and hidden states for the full unshuffled datasets.
        
        TODO: Methods that update LLMs during training should disable caching.
        """
        collect_wagering_hidden_states = self.requires_hidden_states
        collect_calibration_hidden_states = self.logit_calibrator is not None
        collect_any_hidden_states = collect_wagering_hidden_states or collect_calibration_hidden_states

        if collect_any_hidden_states:
            log.info("Collecting logits and hidden states from all models (per-model, per-dataset cache, unshuffled)...")
        else:
            log.info("Collecting logits from all models (hidden states disabled for this wagering method)...")
        
        num_models = len(self.models)
        num_datasets = len(self.datasets)

        per_model_hidden_layers = [
            resolve_hidden_state_layers_for_model(
                self.hidden_state_layers,
                self.hidden_state_layers_per_model,
                model_index=model_idx,
                num_models=num_models,
            )
            if collect_wagering_hidden_states
            else None
            for model_idx in range(num_models)
        ]

        reuse_calibration_from_wagering = (
            collect_calibration_hidden_states
            and collect_wagering_hidden_states
            and all(tuple(layers) == (-1,) for layers in per_model_hidden_layers if layers is not None)
        )
        
        per_dataset_logits = []  # List of [num_models, num_examples_ds, num_options]
        per_dataset_hidden_states = [] if collect_wagering_hidden_states else None
        per_dataset_calibration_hidden_states = (
            [] if (self.logit_calibrator is not None and not reuse_calibration_from_wagering) else None
        )
        per_dataset_context_assignments: List[np.ndarray] = []
        
        for dataset_idx, dataset in enumerate(self.datasets):
            log.debug(f"Processing dataset {dataset_idx + 1}/{num_datasets} for cache collection")
            dataset_logits_list = []
            dataset_hidden_states_list = [] if collect_wagering_hidden_states else None
            dataset_calibration_hidden_states_list = (
                [] if (self.logit_calibrator is not None and not reuse_calibration_from_wagering) else None
            )

            dataset_type = _get_mixed_context_dataset_type(dataset)
            if dataset_type is not None:
                raw = getattr(dataset, f"{dataset_type}_context_assignment_by_example", None)
                if not isinstance(raw, list) or len(raw) != len(dataset.x):
                    raise RuntimeError(
                        "Mixed-context dataset missing per-example context assignments. "
                        "Ensure assign_pubmedqa_context_models ran before cache collection."
                    )
                per_dataset_context_assignments.append(np.asarray(raw, dtype=np.int64))
            else:
                per_dataset_context_assignments.append(np.full((len(dataset.x),), -1, dtype=np.int64))
            
            for model_idx, model in enumerate(self.models):
                model_path = model if isinstance(model, str) else model.model_path
                model_hidden_layers = per_model_hidden_layers[model_idx]
                separate_cal_hs = dataset_calibration_hidden_states_list is not None
                layers_union = _union_hidden_state_layers_wagering_plus_last(
                    model_hidden_layers,
                    include_last_transformer_layer=separate_cal_hs,
                )
                prompt_variant = get_model_prompt_variant(dataset, model_index=model_idx)
                cached_logits, cached_hidden_states, cached_labels = get_cached_logits_and_hidden_states_for_model(
                    model_path,
                    dataset,
                    self.option_tokens,
                    prompt_variant=prompt_variant,
                    model_index=model_idx,
                    hidden_state_layers=model_hidden_layers,
                )
                
                if cached_logits is not None and (
                    (not collect_wagering_hidden_states) or cached_hidden_states is not None
                ):
                    log.debug(
                        f"Model {model_idx + 1}/{num_models}, dataset {dataset_idx + 1}/{num_datasets}: "
                        "Using cached logits"
                    )
                    model_logits = cached_logits
                    model_hidden_states = cached_hidden_states if collect_wagering_hidden_states else None
                elif cached_logits is not None and collect_wagering_hidden_states:
                    log.debug(
                        f"Model {model_idx + 1}/{num_models}, dataset {dataset_idx + 1}/{num_datasets}: "
                        "Found cached logits but not hidden states - collecting both"
                    )
                    if isinstance(model, str):
                        raise RuntimeError(
                            f"Cache miss for model path {model}. Model must be loaded to collect logits."
                        )
                    model_prompts = get_model_specific_prompts(dataset, model_index=model_idx)
                    model_logits, model_hidden_states_all_layers, model_labels = collect_option_logits_and_hidden_states_for_model(
                        model,
                        dataset,
                        self.option_tokens,
                        model_identifier=str(model_path),
                        model_index=model_idx,
                        hidden_state_layers=layers_union,
                        collect_hidden_states=collect_any_hidden_states,
                        model_prompts=model_prompts,
                        prompt_variant=prompt_variant,
                    )
                    set_cached_logits_and_hidden_states_for_model(
                        model,
                        dataset,
                        self.option_tokens,
                        model_logits,
                        model_hidden_states_all_layers,
                        model_labels,
                        prompt_variant=prompt_variant,
                        model_index=model_idx,
                        hidden_state_layers=layers_union,
                    )
                    model_hidden_states = extract_hidden_state_features(
                        model_hidden_states_all_layers,
                        model_hidden_layers,
                        cached_requested_hidden_state_layers=layers_union,
                    )
                    if model_hidden_states is None:
                        raise RuntimeError(
                            "Hidden-state cache is in legacy format and cannot satisfy hidden_state_layers; recache is required"
                        )
                else:
                    if isinstance(model, str):
                        raise RuntimeError(
                            f"Cache miss for model path {model}. Model must be loaded to collect logits."
                        )
                    log.info(
                        f"Model {model_idx + 1}/{num_models}, dataset {dataset_idx + 1}/{num_datasets}: "
                        f"Cache miss - collecting logits and hidden states (device: {model.device()})"
                    )
                    try:
                        model_prompts = get_model_specific_prompts(dataset, model_index=model_idx)
                        model_logits, model_hidden_states_all_layers, model_labels = collect_option_logits_and_hidden_states_for_model(
                            model,
                            dataset,
                            self.option_tokens,
                            model_identifier=str(model_path),
                            model_index=model_idx,
                            hidden_state_layers=layers_union,
                            collect_hidden_states=collect_any_hidden_states,
                            model_prompts=model_prompts,
                            prompt_variant=prompt_variant,
                        )
                        set_cached_logits_and_hidden_states_for_model(
                            model,
                            dataset,
                            self.option_tokens,
                            model_logits,
                            model_hidden_states_all_layers,
                            model_labels,
                            prompt_variant=prompt_variant,
                            model_index=model_idx,
                            hidden_state_layers=layers_union,
                        )
                        if collect_wagering_hidden_states:
                            model_hidden_states = extract_hidden_state_features(
                                model_hidden_states_all_layers,
                                model_hidden_layers,
                                cached_requested_hidden_state_layers=layers_union,
                            )
                            if model_hidden_states is None:
                                raise RuntimeError(
                                    "Hidden-state cache is in legacy format and cannot satisfy hidden_state_layers; recache is required"
                                )
                        else:
                            model_hidden_states = None
                    except Exception as e:
                        raise RuntimeError(
                            f"Error collecting logits and hidden states for model {model_idx + 1} on dataset {dataset_idx + 1}: {e}"
                        ) from e
                
                dataset_logits_list.append(model_logits)
                if collect_wagering_hidden_states and dataset_hidden_states_list is not None:
                    dataset_hidden_states_list.append(model_hidden_states)

                if dataset_calibration_hidden_states_list is not None:
                    calibration_hidden_states = get_cached_logits_and_hidden_states_for_model(
                        model_path,
                        dataset,
                        self.option_tokens,
                        prompt_variant=prompt_variant,
                        model_index=model_idx,
                        hidden_state_layers=[-1],
                    )[1]
                    if calibration_hidden_states is None:
                        if isinstance(model, str):
                            raise RuntimeError(
                                f"Calibration hidden-state cache miss for model path {model}. "
                                "Logit calibration needs last-layer hidden states (layer -1) in the "
                                "on-disk cache alongside wagering layers. "
                                "Run training once with models loaded (not path-only) so missing layers "
                                "can be collected, or delete the affected cache entries."
                            )
                        model_prompts = get_model_specific_prompts(dataset, model_index=model_idx)
                        model_logits, model_hidden_states_all_layers, model_labels = collect_option_logits_and_hidden_states_for_model(
                            model,
                            dataset,
                            self.option_tokens,
                            model_identifier=str(model_path),
                            model_index=model_idx,
                            hidden_state_layers=layers_union,
                            collect_hidden_states=True,
                            model_prompts=model_prompts,
                            prompt_variant=prompt_variant,
                        )
                        set_cached_logits_and_hidden_states_for_model(
                            model,
                            dataset,
                            self.option_tokens,
                            model_logits,
                            model_hidden_states_all_layers,
                            model_labels,
                            prompt_variant=prompt_variant,
                            model_index=model_idx,
                            hidden_state_layers=layers_union,
                        )
                        calibration_hidden_states = extract_hidden_state_features(
                            model_hidden_states_all_layers,
                            [-1],
                            cached_requested_hidden_state_layers=layers_union,
                        )
                        if calibration_hidden_states is None:
                            raise RuntimeError(
                                "Temperature calibration requires last-layer hidden states"
                            )
                    dataset_calibration_hidden_states_list.append(calibration_hidden_states)
            
            # Stack logits for this dataset: [num_models, num_examples_ds, num_options]
            per_dataset_logits.append(np.stack(dataset_logits_list, axis=0))
            if collect_wagering_hidden_states and per_dataset_hidden_states is not None and dataset_hidden_states_list is not None:
                per_dataset_hidden_states.append(dataset_hidden_states_list)
            if per_dataset_calibration_hidden_states is not None and dataset_calibration_hidden_states_list is not None:
                per_dataset_calibration_hidden_states.append(dataset_calibration_hidden_states_list)
        
        # Combine per-dataset logits along the example dimension
        self.all_model_logits = np.concatenate(per_dataset_logits, axis=1)  # [num_models, num_examples, num_options]
        log.debug(f"All training logits shape (combined): {self.all_model_logits.shape}")

        # Mixed-context routing metadata used by context-conditioned calibration (optional).
        if per_dataset_context_assignments:
            combined_context = np.concatenate(per_dataset_context_assignments, axis=0)
            if combined_context.shape[0] == self.all_model_logits.shape[1] and np.any(combined_context >= 0):
                self.all_calibration_context_assignments = combined_context
            else:
                self.all_calibration_context_assignments = None
        
        # Combine hidden states per model across datasets
        if not collect_wagering_hidden_states:
            self.all_hidden_states = None
        elif num_datasets == 0 or num_models == 0:
            self.all_hidden_states = None
            return
        else:
            # Validate hidden state dims per model across datasets
            hidden_dims_per_model = [per_dataset_hidden_states[0][m].shape[-1] for m in range(num_models)]
            for dataset_idx in range(1, num_datasets):
                for m in range(num_models):
                    dim = per_dataset_hidden_states[dataset_idx][m].shape[-1]
                    if dim != hidden_dims_per_model[m]:
                        raise RuntimeError(
                            f"Hidden dimension mismatch for model {m} across datasets: "
                            f"{hidden_dims_per_model[m]} vs {dim} (dataset {dataset_idx})"
                        )

            # Concatenate per model
            combined_hidden_states_by_model = []
            for m in range(num_models):
                model_hs = [per_dataset_hidden_states[d][m] for d in range(num_datasets)]
                combined_hidden_states_by_model.append(np.concatenate(model_hs, axis=0))

            # Stack if all models share same hidden dimension
            if len(set(hidden_dims_per_model)) == 1:
                self.all_hidden_states = np.stack(combined_hidden_states_by_model, axis=0)
                log.debug(f"All training hidden states shape (combined): {self.all_hidden_states.shape}")
            else:
                log.debug(f"Models have different hidden dimensions: {dict(enumerate(hidden_dims_per_model))}")
                log.debug("Storing hidden states as list (will be handled by wagering method)")
                self.all_hidden_states = combined_hidden_states_by_model

        if reuse_calibration_from_wagering:
            # When wagering already uses last-layer-only hidden states, avoid
            # storing a duplicate calibration copy of the same arrays.
            self.all_calibration_hidden_states = self.all_hidden_states
        elif per_dataset_calibration_hidden_states is not None:
            calibration_hidden_by_model = []
            for m in range(num_models):
                model_hs = [per_dataset_calibration_hidden_states[d][m] for d in range(num_datasets)]
                calibration_hidden_by_model.append(np.concatenate(model_hs, axis=0))
            if len(set(hs.shape[-1] for hs in calibration_hidden_by_model)) == 1:
                self.all_calibration_hidden_states = np.stack(calibration_hidden_by_model, axis=0)
            else:
                self.all_calibration_hidden_states = calibration_hidden_by_model
        
        # Note: Validation split happens in _apply_shuffling() after cache loading


    def _collect_hidden_states(self):
        """
        Load hidden states from cache (collected together with logits in _collect_logits).
        
        Since _collect_logits now collects both logits and hidden states together,
        this function just ensures hidden states are loaded from cache if they weren't
        already set during _collect_logits.
        
        Note: Validation split happens AFTER cache loading in _apply_shuffling(), so this
        only loads hidden states for the full unshuffled dataset.
        """
        # Check if wagering method needs hidden states
        if not hasattr(self.wagering_method, 'compute_wagers'):
            return
        if not self.requires_hidden_states:
            self.all_hidden_states = None
            return
        
        # If hidden states were already collected in _collect_logits, we're done
        if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
            log.debug("Hidden states already collected in _collect_logits, skipping")
            return
        
        # Hidden states should always be collected with logits in _collect_logits with per-model caching
        raise RuntimeError("Hidden states not found. They should have been collected with logits in _collect_logits. "
                   "Some wagering methods may not work correctly.")

    def _apply_logit_calibration(self):
        """Apply frozen temperature scaling to cached logits before training logic runs."""
        if self.logit_calibrator is None:
            return

        if not hasattr(self, "all_model_logits") or self.all_model_logits is None:
            raise RuntimeError("Logit calibration requested but no cached logits are available")

        calibration_hidden_states = getattr(self, "all_calibration_hidden_states", None)
        if calibration_hidden_states is None:
            raise RuntimeError("Logit calibration requested but last-layer hidden states are unavailable")

        context_assignments = getattr(self, "all_calibration_context_assignments", None)
        try:
            if context_assignments is not None:
                self.all_model_logits = self.logit_calibrator.apply_to_stacked_logits(
                    self.all_model_logits,
                    calibration_hidden_states,
                    context_model_index_by_example=context_assignments,
                )
            else:
                self.all_model_logits = self.logit_calibrator.apply_to_stacked_logits(
                    self.all_model_logits,
                    calibration_hidden_states,
                )
        except TypeError:
            # Back-compat: older calibrators do not accept context assignments.
            self.all_model_logits = self.logit_calibrator.apply_to_stacked_logits(
                self.all_model_logits,
                calibration_hidden_states,
            )
        log.info("Applied frozen temperature scaling to cached training logits")
    
    def train(self, num_epochs: int = 100) -> Dict[str, Any]:
        """
        Train the wagering method.
        
        Args:
            num_epochs: Number of epochs to train (default: 100)
            
        Returns:
            Dictionary with training results and metrics
        """
        self.wagering_method.train_mode()

        requested_num_epochs = int(num_epochs)
        effective_num_epochs = requested_num_epochs
        reuse_static_epoch_results = False

        # Inference-only methods with no trainable parameters produce identical
        # per-batch outputs on repeated epochs over the same frozen cached logits.
        # Run one epoch and reuse those metrics for remaining epochs.
        if requested_num_epochs > 1:
            has_trainable_params = False
            try:
                trainable_params = self.wagering_method.get_trainable_parameters()
                has_trainable_params = bool(trainable_params)
            except Exception as exc:
                log.debug(
                    "Could not inspect trainable parameters for epoch reuse optimization: %s",
                    exc,
                )

            if not has_trainable_params:
                effective_num_epochs = 1
                reuse_static_epoch_results = self.max_training_batches is None
                log.info(
                    "Method %s has no trainable parameters: running one epoch%s.",
                    type(self.wagering_method).__name__,
                    (
                        f" and reusing results for {requested_num_epochs - 1} additional epoch(s)"
                        if reuse_static_epoch_results
                        else " (max_training_batches set; skipping synthetic multi-epoch metric reuse)"
                    ),
                )
        
        num_examples = len(self.combined_dataset.x)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size
        if self.max_training_batches is not None:
            log.info(
                "max_training_batches=%d (dataset yields %d batches per epoch).",
                self.max_training_batches,
                num_batches,
            )
        
        # Training loop
        batch_metrics = []
        
        # Track epoch-level metrics for early stopping
        epoch_accuracies = []

        # Track validation Nash-gap trajectory and related metrics over epochs
        val_nash_gap_history = []
        val_d_regret_history = []
        val_accuracy_history = []
        val_nash_gap_history_epochs = []

        # Initialize these lists (will be reset each epoch to only keep final epoch's predictions)
        all_predictions = []
        all_aggregated_probs = []
        wagers_history = []
        stop_training_now = False
        periodic_plot_count = 0
        last_completed_batches = 0
        online_window_batches = 1
        online_window_target_examples = self.batch_size
        online_metric_window: deque = deque(maxlen=1)

        if self.early_stopping_criterion == "online_learning":
            if self.validation_dataset is not None:
                validation_examples = len(self.validation_dataset.x)
            else:
                validation_examples = int(round(num_examples * self.validation_split_ratio))

            if validation_examples <= 0:
                validation_examples = self.batch_size

            online_window_batches = max(1, int(round(validation_examples / float(self.batch_size))))
            online_window_target_examples = online_window_batches * self.batch_size
            online_metric_window = deque(maxlen=online_window_batches)

        if self.wager_score_plot_every is not None:
            log.info(
                "Enabled validation wagers-vs-score_diff plotting every %d batches "
                "(capped at 1000 validation samples per plot).",
                self.wager_score_plot_every,
            )
            estimated_total_batches = effective_num_epochs * num_batches
            if self.wager_score_plot_every > estimated_total_batches:
                log.warning(
                    "wager_score_plot_every=%d exceeds estimated total batches=%d; "
                    "periodic plot trigger may never fire. A final fallback plot will still be logged.",
                    self.wager_score_plot_every,
                    estimated_total_batches,
                )

        if self.stop_at_last_iteration:
            log.info(
                "Early stopping disabled (stop_at_last_iteration=True): training will run to the final iteration."
            )
        elif self.early_stopping_patience > 0:
            if self.early_stopping_criterion == "online_learning":
                metric_name = (
                    "rolling-window kl_to_gold"
                    if self.use_min_kl_for_early_stopping
                    else (
                        "rolling-window brier_d_regret"
                        if self.use_brier_d_regret_for_early_stopping
                        else "rolling-window max nash gap"
                    )
                )
                log.info(
                    "Early stopping enabled: criterion=online_learning, metric=%s, "
                    "patience=%d batches, window=%d batches (~%d examples, validation=%d)",
                    metric_name,
                    self.early_stopping_patience,
                    online_window_batches,
                    online_window_target_examples,
                    validation_examples,
                )
            else:
                if self.use_min_kl_for_early_stopping:
                    metric_name = "validation kl_to_gold"
                else:
                    metric_name = (
                        "validation brier_d_regret"
                        if self.use_brier_d_regret_for_early_stopping
                        else "validation nash_gap/d_regret"
                    )
                log.info(
                    "Early stopping enabled: criterion=validation, metric=%s, "
                    "patience=%d epochs",
                    metric_name,
                    self.early_stopping_patience,
                )
        
        epoch_start = self.start_epoch
        epoch_stop = self.start_epoch + effective_num_epochs
        batches_processed = 0
        for epoch in range(epoch_start, epoch_stop):
            log.debug(
                f"Epoch {epoch - epoch_start + 1}/{effective_num_epochs} (absolute epoch {epoch + 1})"
            )
            
            # Reset predictions/probs/wagers at start of each epoch
            # We only want to keep the final epoch's predictions for evaluation
            all_predictions = []
            all_aggregated_probs = []
            wagers_history = []
            
            # Determine starting batch (for resume from mid-epoch)
            if epoch == self.start_epoch and self.current_step > 0:
                # Resume from the batch we left off at (within this epoch)
                start_batch = (self.current_step % num_examples) // self.batch_size
                start_step = start_batch * self.batch_size
                log.debug(f"Resuming from batch {start_batch} (step {start_step}) in epoch {epoch+1}")
                self._processed_start_idx = start_step
            else:
                start_batch = 0
                start_step = 0
                self._processed_start_idx = 0
            
            # Process epoch in batches
            epoch_predictions = []
            epoch_probs = []
            epoch_correct = 0
            epoch_nll_sum = 0.0
            
            for batch_idx in range(start_batch, num_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, num_examples)
                last_completed_batches = epoch * num_batches + (batch_idx + 1)
                
                # Process batch
                batch_logits = self.all_model_logits[:, batch_start:batch_end, :]  # [num_models, batch_size, num_options]
                batch_logits_transposed = np.transpose(batch_logits, (1, 0, 2))  # [batch_size, num_models, num_options]
                batch_labels = self.labels[batch_start:batch_end]  # [batch_size]
                batch_size_actual = batch_end - batch_start
                
                # Get questions for batch (for wagering methods that need them)
                batch_questions = self.combined_dataset.x[batch_start:batch_end]  # List of question strings
                batch_questions = self._get_router_questions_for_batch(
                    batch_questions,
                    batch_start,
                    batch_end,
                    validation=False,
                )
                batch_questions_per_model = self._get_router_questions_per_model_for_batch(
                    batch_questions,
                    batch_start,
                    batch_end,
                    validation=False,
                )
                
                # Get hidden states for batch if available
                batch_hidden_states = None
                if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
                    if isinstance(self.all_hidden_states, list):
                        # List of arrays with different dimensions - extract batch for each model
                        # Structure: List of [num_models], where each element is [batch_size, hidden_dim_i]
                        batch_hidden_states = []
                        for i in range(len(self.all_hidden_states)):
                            model_hs = self.all_hidden_states[i][batch_start:batch_end, :]  # [batch_size, hidden_dim_i]
                            batch_hidden_states.append(model_hs)
                        # Keep as list to preserve variable hidden dimensions per model
                        # Will be processed per-model in wagering_method.compute_wagers
                    else:
                        # Stacked array: [num_models, num_examples, hidden_dim]
                        batch_hidden_states_array = self.all_hidden_states[:, batch_start:batch_end, :]  # [num_models, batch_size, hidden_dim]
                        # Convert to list of [num_models] arrays, each [batch_size, hidden_dim]
                        batch_hidden_states = [batch_hidden_states_array[i, :, :] for i in range(batch_hidden_states_array.shape[0])]
                
                # Compute wagers for entire batch
                    # Variable hidden dimensions per model - use batch  processing
                batch_dataset_indices = np.asarray(
                    self.dataset_indices[batch_start:batch_end], dtype=np.int32
                )
                batch_gold_label_distribution = None
                try:
                    batch_local_indices = np.asarray(
                        self.example_local_indices[batch_start:batch_end], dtype=np.int32
                    )
                    num_options = int(batch_logits_transposed.shape[2])
                    # Default: one-hot labels for all examples.
                    batch_gold_label_distribution = np.eye(num_options, dtype=np.float32)[
                        batch_labels.astype(np.int64)
                    ]
                    # Override with probabilistic labels when available (cluster_saturation_* CSV datasets).
                    for dataset_idx in np.unique(batch_dataset_indices).tolist():
                        ds_idx = int(dataset_idx)
                        if ds_idx < 0 or ds_idx >= len(self.datasets):
                            continue
                        ds = self.datasets[ds_idx]
                        dataset_name = getattr(ds, "cache_dataset_name", None)
                        is_cluster_saturation = _is_cluster_saturation_dataset_name(dataset_name)
                        if not is_cluster_saturation or not hasattr(ds, "probabilistic_labels"):
                            continue
                        if num_options != 2:
                            raise ValueError(
                                "probabilistic_labels are only supported for binary option sets "
                                f"(num_options={num_options})"
                            )
                        pos_idx = _resolve_positive_option_index(
                            getattr(ds, "positive_label", None),
                            self.option_tokens,
                            num_options,
                        )
                        if pos_idx is None:
                            raise ValueError(
                                "Could not resolve positive option index for probabilistic labels"
                            )
                        mask = batch_dataset_indices == ds_idx
                        local = batch_local_indices[mask].astype(np.int64, copy=False)
                        gt_probs_all = np.asarray(ds.probabilistic_labels, dtype=np.float32)
                        p_pos = gt_probs_all[local]
                        p_pos = np.clip(p_pos, 0.0, 1.0)
                        neg_idx = 1 - int(pos_idx)
                        soft = np.zeros((int(mask.sum()), num_options), dtype=np.float32)
                        soft[:, int(pos_idx)] = p_pos
                        soft[:, neg_idx] = 1.0 - p_pos
                        batch_gold_label_distribution[mask] = soft
                except Exception as e:
                    # If anything goes wrong, fall back to hard labels and make it visible.
                    log.error(
                        "Failed to construct gold_label_distribution for batch [%d:%d]; "
                        "falling back to hard labels. Error: %s",
                        int(batch_start),
                        int(batch_end),
                        e,
                    )
                    batch_gold_label_distribution = None
                wagering_kwargs = {
                    "model_logits": batch_logits_transposed,
                    "gold_label": batch_labels,
                    "hidden_states_list": batch_hidden_states,
                    "questions": batch_questions,
                }
                if batch_gold_label_distribution is not None:
                    wagering_kwargs["gold_label_distribution"] = batch_gold_label_distribution
                if batch_questions_per_model is not None:
                    wagering_kwargs["questions_per_model"] = batch_questions_per_model
                if self.method_requires_model_perplexities:
                    if self.model_prompt_perplexities is None:
                        raise RuntimeError(
                            "Wagering method requires model_perplexities but training perplexities are unavailable"
                        )
                    wagering_kwargs["model_perplexities"] = self.model_prompt_perplexities[
                        batch_start:batch_end
                    ]

                res_dict = self.wagering_method.compute_wagers(**wagering_kwargs)  # [batch_size, num_models]
                
                batch_wagers = res_dict["wagers"]
                batch_sigmoid_wagers = res_dict.get("sigmoid_wagers", None)
                batch_total_payout_values = res_dict.get("total_payout", None)
                nash_gap = res_dict.get("nash_gap", None)
                batch_nash_gap_max = self._compute_batch_max_nash_gap(nash_gap)

                # Aggregate predictions for entire batch
                batch_aggregated_log_probs, batch_aggregated_probs = self.aggregation_function.aggregate(
                    batch_logits_transposed, batch_wagers
                )  # [batch_size, num_options] each
                
                batch_predictions = np.argmax(batch_aggregated_probs, axis=1)  # [batch_size]
                
                # Update wagering method with batch
                # Convert logits to probabilities for update method
                max_logits = np.max(batch_logits_transposed, axis=2, keepdims=True)  # [batch_size, num_models, 1]
                stabilized = batch_logits_transposed - max_logits
                log_z = max_logits + np.log(np.exp(stabilized).sum(axis=2, keepdims=True))
                batch_model_probs = np.exp(batch_logits_transposed - log_z)  # [batch_size, num_models, num_options]

                if self.debug_batch_prob_alignment:
                    batch_example_local_indices = np.asarray(
                        self.example_local_indices[batch_start:batch_end], dtype=np.int32
                    )
                    _debug_log_batch_prob_vs_gt_distribution(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batch_start=batch_start,
                        batch_end=batch_end,
                        batch_gold_label_distribution=batch_gold_label_distribution,
                        batch_labels=batch_labels,
                        batch_model_probs=batch_model_probs,
                        batch_dataset_indices=batch_dataset_indices,
                        batch_example_local_indices=batch_example_local_indices,
                        datasets=self.datasets,
                        option_tokens=self.option_tokens,
                    )

                batch_update_info = self.wagering_method.update(
                    aggregated_probs=batch_aggregated_probs,
                    aggregated_pred=batch_predictions,
                    gold_label=batch_labels,
                    model_probs=batch_model_probs,
                    model_logits=batch_logits_transposed,
                    hidden_states=batch_hidden_states,
                    gold_label_distribution=batch_gold_label_distribution,
                )
                

                # Compute batch metrics using vectorized operations
                batch_correct = (batch_predictions == batch_labels)
                batch_nll = -np.log(batch_aggregated_probs[np.arange(batch_size_actual), batch_labels] + 1e-10)
                batch_d_regret = None
                batch_brier_d_regret = None
                batch_kl_to_gold = None
                batch_soft_label_count = 0
                batch_kendall_tau = None
                batch_best_model_mrr = None
                batch_total_wagers = None
                batch_total_payout = None

                if batch_sigmoid_wagers is not None:
                    batch_sigmoid_wagers_arr = np.asarray(batch_sigmoid_wagers)
                    if batch_sigmoid_wagers_arr.ndim == 1:
                        batch_sigmoid_wagers_arr = batch_sigmoid_wagers_arr[np.newaxis, :]
                    if batch_sigmoid_wagers_arr.ndim == 2 and batch_sigmoid_wagers_arr.shape[1] > 0:
                        total_wagers_per_example = np.sum(batch_sigmoid_wagers_arr, axis=1)
                        batch_total_wagers = float(np.mean(total_wagers_per_example))

                if batch_total_payout_values is not None:
                    batch_total_payout_arr = np.asarray(batch_total_payout_values)
                    if batch_total_payout_arr.ndim == 1:
                        batch_total_payout = float(np.mean(batch_total_payout_arr))
                    elif batch_total_payout_arr.ndim == 2 and batch_total_payout_arr.shape[1] > 0:
                        total_payout_per_example = np.sum(batch_total_payout_arr, axis=1)
                        batch_total_payout = float(np.mean(total_payout_per_example))

                if batch_total_payout is None and batch_sigmoid_wagers is not None:
                    # Backward-compatible fallback if methods return score_diff but not total_payout.
                    batch_score_diff_values = res_dict.get("score_diff", None)
                    if batch_score_diff_values is not None:
                        batch_score_diff_arr = np.asarray(batch_score_diff_values)
                        batch_sigmoid_wagers_arr = np.asarray(batch_sigmoid_wagers)
                        if batch_score_diff_arr.ndim == 1:
                            batch_score_diff_arr = batch_score_diff_arr[np.newaxis, :]
                        if batch_sigmoid_wagers_arr.ndim == 1:
                            batch_sigmoid_wagers_arr = batch_sigmoid_wagers_arr[np.newaxis, :]
                        if (
                            batch_score_diff_arr.ndim == 2
                            and batch_sigmoid_wagers_arr.ndim == 2
                            and batch_score_diff_arr.shape == batch_sigmoid_wagers_arr.shape
                        ):
                            total_payout_matrix = batch_sigmoid_wagers_arr * (
                                batch_score_diff_arr - 0.5 * batch_sigmoid_wagers_arr
                            )
                            total_payout_per_example = np.sum(total_payout_matrix, axis=1)
                            batch_total_payout = float(np.mean(total_payout_per_example))

                try:
                    batch_d_regret, batch_best_expert_ids = compute_dynamic_regret(
                        batch_logits_transposed,
                        batch_aggregated_probs,
                        batch_labels,
                    )
                    if batch_gold_label_distribution is not None:
                        batch_brier_d_regret = compute_brier_dynamic_regret(
                            batch_logits_transposed,
                            batch_aggregated_probs,
                            batch_labels,
                            gold_label_distribution=np.asarray(
                                batch_gold_label_distribution, dtype=np.float64
                            ),
                        )
                    else:
                        batch_brier_d_regret = compute_brier_dynamic_regret(
                            batch_logits_transposed,
                            batch_aggregated_probs,
                            batch_labels,
                        )
                    batch_model_brier_scores = _compute_model_brier_scores(
                        batch_logits_transposed,
                        batch_labels,
                    )
                    batch_meta_metrics = compute_meta_metrics(
                        batch_wagers,
                        batch_best_expert_ids,
                        model_brier_scores=batch_model_brier_scores,
                    )
                    batch_kendall_tau = batch_meta_metrics.get("kendall_tau")
                    batch_best_model_mrr = batch_meta_metrics.get("best_model_mrr")
                except Exception as e:
                    # Keep training robust if a batch-level metric computation fails.
                    log.debug(f"Failed to compute batch d_regret/meta metrics: {e}")

                try:
                    if batch_gold_label_distribution is not None:
                        soft_mask = np.zeros((int(batch_size_actual),), dtype=bool)
                        for dataset_idx in np.unique(batch_dataset_indices).tolist():
                            ds_idx = int(dataset_idx)
                            if ds_idx < 0 or ds_idx >= len(self.datasets):
                                continue
                            ds = self.datasets[ds_idx]
                            dataset_name = getattr(ds, "cache_dataset_name", None)
                            if not _is_cluster_saturation_dataset_name(dataset_name):
                                continue
                            if not hasattr(ds, "probabilistic_labels"):
                                continue
                            soft_mask |= batch_dataset_indices == ds_idx
                        batch_soft_label_count = int(np.sum(soft_mask))
                        if batch_soft_label_count > 0:
                            batch_kl_to_gold = _compute_mean_kl_to_gold_distribution(
                                np.asarray(batch_gold_label_distribution, dtype=np.float64),
                                np.asarray(batch_aggregated_probs, dtype=np.float64),
                                mask=soft_mask,
                            )
                except Exception as e:
                    log.debug(f"Failed to compute batch kl_to_gold: {e}")
                    batch_kl_to_gold = None
                    batch_soft_label_count = 0

                if (
                    (not self.stop_at_last_iteration)
                    and self.early_stopping_criterion == "online_learning"
                    and self.early_stopping_patience > 0
                ):
                    if self.use_min_kl_for_early_stopping:
                        if batch_kl_to_gold is None or not np.isfinite(float(batch_kl_to_gold)):
                            raise RuntimeError(
                                "early_stopping_criterion='online_learning' with "
                                "use_min_kl_for_early_stopping=True requires a finite "
                                "batch kl_to_gold metric. This metric is only computed for "
                                "datasets with soft probabilistic labels (probability_label_column / "
                                "dataset.probabilistic_labels)."
                            )
                        if batch_soft_label_count <= 0:
                            raise RuntimeError(
                                "early_stopping_criterion='online_learning' with "
                                "use_min_kl_for_early_stopping=True requires each training batch "
                                "to include at least one example with soft probabilistic labels."
                            )

                        online_metric_window.append((float(batch_kl_to_gold), int(batch_soft_label_count)))
                        if len(online_metric_window) < online_window_batches:
                            improved = False
                            current_batch_metric = None
                        else:
                            weighted_sum = 0.0
                            total_weight = 0
                            for value, weight in online_metric_window:
                                weighted_sum += float(value) * int(weight)
                                total_weight += int(weight)
                            current_batch_metric = weighted_sum / float(max(total_weight, 1))
                            improved = current_batch_metric < self.best_batch_kl_to_gold
                        if improved:
                            self.best_batch_kl_to_gold = current_batch_metric
                    elif self.use_brier_d_regret_for_early_stopping:
                        if batch_brier_d_regret is None or not np.isfinite(batch_brier_d_regret):
                            raise RuntimeError(
                                "early_stopping_criterion='online_learning' with "
                                "use_brier_d_regret_for_early_stopping=True requires a finite "
                                "batch brier_d_regret metric."
                            )

                        online_metric_window.append((float(batch_brier_d_regret), batch_size_actual))
                        if len(online_metric_window) < online_window_batches:
                            improved = False
                            current_batch_metric = None
                        else:
                            weighted_sum = 0.0
                            total_weight = 0
                            for value, weight in online_metric_window:
                                weighted_sum += float(value) * int(weight)
                                total_weight += int(weight)
                            current_batch_metric = weighted_sum / float(max(total_weight, 1))
                            improved = current_batch_metric < self.best_batch_brier_d_regret
                        if improved:
                            self.best_batch_brier_d_regret = current_batch_metric
                    else:
                        if batch_nash_gap_max is None:
                            raise RuntimeError(
                                "early_stopping_criterion='online_learning' requires compute_wagers() "
                                "to return a finite 'nash_gap' for each training batch."
                            )

                        online_metric_window.append((float(batch_nash_gap_max), batch_size_actual))
                        if len(online_metric_window) < online_window_batches:
                            improved = False
                            current_batch_metric = None
                        else:
                            current_batch_metric = max(value for value, _ in online_metric_window)
                            improved = current_batch_metric < self.best_batch_nash_gap
                        if improved:
                            self.best_batch_nash_gap = current_batch_metric

                    if current_batch_metric is None:
                        pass
                    elif improved:
                        self.batches_since_improvement = 0
                        self.best_wagering_method_state = copy.deepcopy(self.wagering_method.state_dict())
                        self.best_epoch = epoch
                        self.best_batch_step = epoch * num_examples + batch_end
                        if self.use_min_kl_for_early_stopping:
                            log.debug(
                                "New best online-learning rolling kl_to_gold: %.6f "
                                "(window=%d batches) at epoch %d batch %d (global step %d)",
                                self.best_batch_kl_to_gold,
                                online_window_batches,
                                epoch + 1,
                                batch_idx + 1,
                                self.best_batch_step,
                            )
                        elif self.use_brier_d_regret_for_early_stopping:
                            log.debug(
                                "New best online-learning rolling brier_d_regret: %.6f "
                                "(window=%d batches) at epoch %d batch %d (global step %d)",
                                self.best_batch_brier_d_regret,
                                online_window_batches,
                                epoch + 1,
                                batch_idx + 1,
                                self.best_batch_step,
                            )
                        else:
                            log.debug(
                                "New best online-learning rolling max nash gap: %.6f "
                                "(window=%d batches) at epoch %d batch %d (global step %d)",
                                self.best_batch_nash_gap,
                                online_window_batches,
                                epoch + 1,
                                batch_idx + 1,
                                self.best_batch_step,
                            )
                    else:
                        self.batches_since_improvement += 1

                    if self.batches_since_improvement >= self.early_stopping_patience:
                        if self.use_min_kl_for_early_stopping:
                            log.info(
                                "Early stopping (online_learning): rolling-window kl_to_gold "
                                "(window=%d batches) did not improve for %d batches. "
                                "Best rolling kl_to_gold: %.6f%s",
                                online_window_batches,
                                self.early_stopping_patience,
                                self.best_batch_kl_to_gold,
                                (
                                    f" (epoch {self.best_epoch + 1}, step {self.best_batch_step})"
                                    if self.best_epoch is not None and self.best_batch_step is not None
                                    else ""
                                ),
                            )
                        elif self.use_brier_d_regret_for_early_stopping:
                            log.info(
                                "Early stopping (online_learning): rolling-window brier_d_regret "
                                "(window=%d batches) did not improve for %d batches. "
                                "Best rolling brier_d_regret: %.6f%s",
                                online_window_batches,
                                self.early_stopping_patience,
                                self.best_batch_brier_d_regret,
                                (
                                    f" (epoch {self.best_epoch + 1}, step {self.best_batch_step})"
                                    if self.best_epoch is not None and self.best_batch_step is not None
                                    else ""
                                ),
                            )
                        else:
                            log.info(
                                "Early stopping (online_learning): rolling-window max nash gap "
                                "(window=%d batches) did not improve for %d batches. "
                                "Best rolling max nash gap: %.6f%s",
                                online_window_batches,
                                self.early_stopping_patience,
                                self.best_batch_nash_gap,
                                (
                                    f" (epoch {self.best_epoch + 1}, step {self.best_batch_step})"
                                    if self.best_epoch is not None and self.best_batch_step is not None
                                    else ""
                                ),
                            )
                        self.early_stopped = True
                        stop_training_now = True
                        break
                
                epoch_correct += int(np.sum(batch_correct))
                epoch_nll_sum += np.sum(batch_nll)
                
                # Store batch results for epoch metrics
                all_predictions.extend(batch_predictions.tolist())
                all_aggregated_probs.extend(batch_aggregated_probs.tolist())
                wagers_history.extend(batch_wagers.tolist())
                epoch_predictions.extend(batch_predictions.tolist())
                epoch_probs.extend(batch_aggregated_probs.tolist())
                
                # Log batch-level metrics
                global_step = int(self.current_step + batch_size_actual)
                batch_ece = None
                try:
                    ece_metric = ECE(n_bins=20)
                    batch_confidences = batch_aggregated_probs.max(axis=1)
                    batch_correctness = batch_correct.astype(float)
                    batch_ece = ece_metric(batch_confidences.tolist(), batch_correctness.tolist())
                except Exception:
                    batch_ece = np.nan

                batch_auc = None
                try:
                    batch_max_probs = batch_aggregated_probs.max(axis=1)
                    batch_binary_correct = batch_correct.astype(int)
                    if len(np.unique(batch_binary_correct)) >= 2:
                        batch_auc = roc_auc_score(batch_binary_correct, batch_max_probs)
                    else:
                        batch_auc = np.nan
                except Exception:
                    batch_auc = np.nan

                batch_record = {
                    "global_step": int(global_step),
                    "epoch": int(epoch + 1),
                    "batch_index_in_epoch": int(batch_idx + 1),
                    "batch_size": int(batch_size_actual),
                    "accuracy": float(np.mean(batch_correct)),
                    "nll": float(np.mean(batch_nll)),
                    "auc": float(batch_auc) if batch_auc is not None and not np.isnan(batch_auc) else None,
                    "ece": float(batch_ece) if batch_ece is not None and not np.isnan(batch_ece) else None,
                    "d_regret": float(batch_d_regret) if batch_d_regret is not None and not np.isnan(batch_d_regret) else None,
                    "brier_d_regret": float(batch_brier_d_regret) if batch_brier_d_regret is not None and not np.isnan(batch_brier_d_regret) else None,
                    "kendall_tau": float(batch_kendall_tau) if batch_kendall_tau is not None and not np.isnan(batch_kendall_tau) else None,
                    "best_model_mrr": float(batch_best_model_mrr) if batch_best_model_mrr is not None and not np.isnan(batch_best_model_mrr) else None,
                    "nash_gap_max": float(batch_nash_gap_max) if batch_nash_gap_max is not None and np.isfinite(batch_nash_gap_max) else None,
                }

                # Add wagering-specific batch summaries for offline analysis/plotting.
                # Keep these mirrored with wandb keys where possible.
                if batch_total_wagers is not None and np.isfinite(batch_total_wagers):
                    batch_record["total_wagers"] = float(batch_total_wagers)
                if batch_total_payout is not None and np.isfinite(batch_total_payout):
                    batch_record["total_payout"] = float(batch_total_payout)
                try:
                    if batch_wagers is not None and hasattr(batch_wagers, "shape") and batch_wagers.shape[1] > 0:
                        for i in range(batch_wagers.shape[1]):
                            batch_record[f"wager_model_{i}"] = float(np.mean(batch_wagers[:, i]))
                except Exception:
                    pass
                try:
                    batch_sigmoid_wagers = res_dict.get("sigmoid_wagers", None)
                    if batch_sigmoid_wagers is not None:
                        sw = np.asarray(batch_sigmoid_wagers)
                        if sw.ndim == 1:
                            sw = sw[np.newaxis, :]
                        if sw.ndim == 2 and sw.shape[1] > 0:
                            for i in range(sw.shape[1]):
                                batch_record[f"sigmoid_wager_model_{i}"] = float(np.mean(sw[:, i]))
                except Exception:
                    pass
                try:
                    batch_total_payout_values = res_dict.get("total_payout", None)
                    if batch_total_payout_values is not None:
                        payout_arr = np.asarray(batch_total_payout_values)
                        if payout_arr.ndim == 2 and payout_arr.shape[1] > 0:
                            for i in range(payout_arr.shape[1]):
                                batch_record[f"net_payout_model_{i}"] = float(np.mean(payout_arr[:, i]))
                except Exception:
                    pass
                self.batch_metrics_history.append(batch_record)

                if self.wandb_logger:
                    wandb_log_dict = {
                        "train/batch/accuracy": float(np.mean(batch_correct)),
                        "train/batch/nll": float(np.mean(batch_nll)),
                        "train/batch/batch_size": batch_size_actual,
                    }
                    if batch_auc is not None and not np.isnan(batch_auc):
                        wandb_log_dict["train/batch/auc"] = float(batch_auc)
                    if batch_ece is not None and not np.isnan(batch_ece):
                        wandb_log_dict["train/batch/ece"] = float(batch_ece)
                    if batch_d_regret is not None and not np.isnan(batch_d_regret):
                        wandb_log_dict["train/batch/d_regret"] = float(batch_d_regret)
                    if batch_brier_d_regret is not None and not np.isnan(batch_brier_d_regret):
                        wandb_log_dict["train/batch/brier_d_regret"] = float(batch_brier_d_regret)
                    if batch_kendall_tau is not None and not np.isnan(batch_kendall_tau):
                        wandb_log_dict["train/batch/kendall_tau"] = float(batch_kendall_tau)
                    if batch_best_model_mrr is not None and not np.isnan(batch_best_model_mrr):
                        wandb_log_dict["train/batch/best_model_mrr"] = float(batch_best_model_mrr)
                    if batch_nash_gap_max is not None:
                        wandb_log_dict["train/batch/nash_gap_max"] = float(batch_nash_gap_max)
                    if batch_total_wagers is not None and np.isfinite(batch_total_wagers):
                        wandb_log_dict["train/batch/total_wagers"] = float(batch_total_wagers)
                    if batch_total_payout is not None and np.isfinite(batch_total_payout):
                        wandb_log_dict["train/batch/total_payout"] = float(batch_total_payout)
                    
                    # Add average wager statistics
                    for i in range(batch_wagers.shape[1]):
                        wandb_log_dict[f"train/batch/wager_model_{i}"] = float(np.mean(batch_wagers[:, i]))
                    
                    # Add update info if available
                    if batch_update_info:
                        for key, value in batch_update_info.items():
                            if isinstance(value, (int, float, np.number)):
                                wandb_log_dict[f"train/batch/update_{key}"] = float(value)
                    
                    self.wandb_logger.log(wandb_log_dict, step=global_step)
                    
                    # Add to buffer for running averages
                    self.batch_metrics_buffer.append({
                        "batch_accuracy": float(np.mean(batch_correct)),
                        "batch_nll": float(np.mean(batch_nll)),
                        "batch_size": batch_size_actual,
                        **(
                            {"batch_auc": float(batch_auc)}
                            if batch_auc is not None and not np.isnan(batch_auc)
                            else {}
                        ),
                        **(
                            {"batch_ece": float(batch_ece)}
                            if batch_ece is not None and not np.isnan(batch_ece)
                            else {}
                        ),
                        **(
                            {"batch_d_regret": float(batch_d_regret)}
                            if batch_d_regret is not None and not np.isnan(batch_d_regret)
                            else {}
                        ),
                        **(
                            {"batch_brier_d_regret": float(batch_brier_d_regret)}
                            if batch_brier_d_regret is not None and not np.isnan(batch_brier_d_regret)
                            else {}
                        ),
                        **(
                            {"batch_kendall_tau": float(batch_kendall_tau)}
                            if batch_kendall_tau is not None and not np.isnan(batch_kendall_tau)
                            else {}
                        ),
                        **(
                            {"batch_best_model_mrr": float(batch_best_model_mrr)}
                            if batch_best_model_mrr is not None and not np.isnan(batch_best_model_mrr)
                            else {}
                        ),
                        **(
                            {"batch_total_wagers": float(batch_total_wagers)}
                            if batch_total_wagers is not None and np.isfinite(batch_total_wagers)
                            else {}
                        ),
                        **(
                            {"batch_total_payout": float(batch_total_payout)}
                            if batch_total_payout is not None and np.isfinite(batch_total_payout)
                            else {}
                        ),
                    })
                    
                    # Compute and log running averages
                    running_avgs = self._compute_running_averages()
                    wandb_avg_dict = {}
                    for key, value in running_avgs.items():
                        wandb_avg_dict[f"train/batch/running_avg_{key}"] = value
                    self.wandb_logger.log(wandb_avg_dict, step=global_step)
                    
                    # Update current_step to track the latest logged step
                    self.current_step = global_step
                else:
                    # Update current_step even without wandb logger
                    self.current_step = global_step

                batches_processed += 1
                if self.max_training_batches is not None and batches_processed >= self.max_training_batches:
                    log.info(
                        "Stopping after %d training batch(es) (max_training_batches).",
                        self.max_training_batches,
                    )
                    stop_training_now = True

                if self.wager_score_plot_every is not None and self.validation_dataset is not None:
                    completed_batches = epoch * num_batches + (batch_idx + 1)
                    if completed_batches % self.wager_score_plot_every == 0:
                        plot_arrays = self._collect_validation_plot_arrays(max_examples=1000)
                        if plot_arrays is not None:
                            periodic_plot_count += 1
                            self._plot_val_wagers_vs_score_diff_for_epoch(
                                val_wagers=plot_arrays["wagers"],
                                val_score_diffs=plot_arrays["score_diff"],
                                model_brier_scores=plot_arrays.get("model_brier_scores"),
                                context_assignment_mask=plot_arrays.get("context_assignment_mask"),
                                context_assignment_kind=(
                                    str(plot_arrays["context_assignment_kind"][0])
                                    if "context_assignment_kind" in plot_arrays
                                    else None
                                ),
                                epoch=epoch,
                                batch_step=completed_batches,
                            )

                            if "estimated_score_diff" in plot_arrays:
                                self._plot_val_estimated_score_diff_vs_wagers_for_epoch(
                                    val_wagers=plot_arrays["wagers"],
                                    val_estimated_score_diffs=plot_arrays["estimated_score_diff"],
                                    model_brier_scores=plot_arrays.get("model_brier_scores"),
                                    context_assignment_mask=plot_arrays.get("context_assignment_mask"),
                                    context_assignment_kind=(
                                        str(plot_arrays["context_assignment_kind"][0])
                                        if "context_assignment_kind" in plot_arrays
                                        else None
                                    ),
                                    epoch=epoch,
                                    batch_step=completed_batches,
                                )

                            if "scores" in plot_arrays and "estimated_score" in plot_arrays:
                                self._plot_val_own_score_vs_estimated_score_for_epoch(
                                    val_own_scores=plot_arrays["scores"],
                                    val_estimated_scores=plot_arrays["estimated_score"],
                                    model_brier_scores=plot_arrays.get("model_brier_scores"),
                                    context_assignment_mask=plot_arrays.get("context_assignment_mask"),
                                    context_assignment_kind=(
                                        str(plot_arrays["context_assignment_kind"][0])
                                        if "context_assignment_kind" in plot_arrays
                                        else None
                                    ),
                                    epoch=epoch,
                                    batch_step=completed_batches,
                                )

                            if "average_scores" in plot_arrays and "estimated_average_scores" in plot_arrays:
                                self._plot_val_average_score_vs_estimated_average_score_for_epoch(
                                    val_average_scores=plot_arrays["average_scores"],
                                    val_estimated_average_scores=plot_arrays["estimated_average_scores"],
                                    model_brier_scores=plot_arrays.get("model_brier_scores"),
                                    context_assignment_mask=plot_arrays.get("context_assignment_mask"),
                                    context_assignment_kind=(
                                        str(plot_arrays["context_assignment_kind"][0])
                                        if "context_assignment_kind" in plot_arrays
                                        else None
                                    ),
                                    epoch=epoch,
                                    batch_step=completed_batches,
                                )
                        else:
                            log.debug(
                                "Skipping periodic wagers-vs-score_diff plot at epoch %d batch %d: "
                                "validation plot arrays unavailable.",
                                epoch + 1,
                                batch_idx + 1,
                            )

                if stop_training_now:
                    break
            

            # Checkpoint after epoch for deterministic resume/phase transition
            if self.checkpoint_dir and self.save_epoch_checkpoints:
                self._save_checkpoint(epoch)
                # Clear validation cache after checkpoint to free memory
                
            # Compute epoch-level metrics
            epoch_labels = self.labels[self._processed_start_idx:self._processed_start_idx + len(epoch_predictions)]
            epoch_accuracy = np.mean(np.array(epoch_predictions) == epoch_labels)
            epoch_nll = epoch_nll_sum / len(epoch_predictions)
            
            # Increment current_step to ensure epoch-level logging uses a step after batch logs
            # This prevents wandb warnings about logging to an already-used step
            self.current_step += 1
            
            epoch_accuracies.append(epoch_accuracy)
            # log.info(f"Epoch {epoch+1} training accuracy: {epoch_accuracy:.4f}, NLL: {epoch_nll:.4f}")

            if stop_training_now:
                break
            
            # Evaluate on validation set if available
            val_metrics = {}
            val_nash_gap = None
            val_nash_gap_max = None
            val_score_diff = None
            val_wagers = None
            val_sigmoid_wagers = None
            if self.validation_dataset is not None:
                val_metrics, val_nash_gap, val_score_diff, val_wagers, val_sigmoid_wagers = self._evaluate_validation()
                val_d_regret = val_metrics.get("d_regret", None)
                val_brier_d_regret = val_metrics.get("brier_d_regret", None)
                val_kl_to_gold = val_metrics.get("kl_to_gold", None)
                if val_metrics:
                    self.last_val_metrics = val_metrics
                if self.use_min_kl_for_early_stopping and (
                    val_kl_to_gold is None or not np.isfinite(float(val_kl_to_gold))
                ):
                    raise RuntimeError(
                        "use_min_kl_for_early_stopping=True requires a finite validation "
                        "kl_to_gold metric. This metric is only computed for datasets with "
                        "soft probabilistic labels (probability_label_column / dataset.probabilistic_labels)."
                    )
            else:
                val_d_regret = None
                val_brier_d_regret = None
                val_kl_to_gold = None
                if self.early_stopping_criterion == "validation":
                    log.info("No validation set available; validation-based early stopping is disabled")

            # Check for epsilon Nash equilibrium if wagering method provides Nash gap metric
            if (
                (not self.stop_at_last_iteration)
                and self.early_stopping_criterion == "validation"
                and not self.use_brier_d_regret_for_early_stopping
                and not self.use_min_kl_for_early_stopping
                and val_nash_gap is not None
            ):
                log.info(f"Validation Nash gap for epoch {epoch+1}: {val_nash_gap}")
                val_nash_gap_max = float(np.max(val_nash_gap))  # Use max Nash gap across models for early stopping
                val_nash_gap_mean = float(np.mean(val_nash_gap))
                val_d_regret_value = val_metrics.get("d_regret", None) if val_metrics else None
                val_accuracy_value = val_metrics.get("accuracy", None) if val_metrics else None
                val_nash_gap_history.append(val_nash_gap_mean)
                val_d_regret_history.append(
                    float(val_d_regret_value)
                    if val_d_regret_value is not None and not np.isnan(val_d_regret_value)
                    else np.nan
                )
                val_accuracy_history.append(
                    float(val_accuracy_value)
                    if val_accuracy_value is not None and not np.isnan(val_accuracy_value)
                    else np.nan
                )
                val_nash_gap_history_epochs.append(epoch + 1)
                
                if val_nash_gap_max < self.best_nash_gap:
                    self.best_nash_gap = val_nash_gap_max
                    self.epochs_since_improvement = 0
                    # Save the best checkpoint state (in memory and to disk)
                    # IMPORTANT: Use deep copy to avoid reference issues where subsequent
                    # training updates would modify the stored checkpoint state
                    self.best_wagering_method_state = copy.deepcopy(self.wagering_method.state_dict())
                    self.best_epoch = epoch
        
                    log.debug(f"Saving best checkpoint state dict keys: {list(self.best_wagering_method_state.keys())}")

                    log.debug(f"New best nash gap: {self.best_nash_gap:.4f} at epoch {epoch+1}")
                else:
                    self.epochs_since_improvement += 1
                
                # Check if we should stop early
                if self.epochs_since_improvement >= self.early_stopping_patience:
                    log.info(
                        f"Early stopping: No improvement on validation set for {self.early_stopping_patience} epochs. "
                        f"Best validation nash gap: {self.best_nash_gap:.4f} (from epoch {self.best_epoch + 1})"
                    )
                    self.early_stopped = True
                    # Load the best checkpoint before breaking
                    if self.best_wagering_method_state is not None:
                        log.info(f"Loading best checkpoint from epoch {self.best_epoch + 1} (nash gap={self.best_nash_gap:.4f})")
                        log.debug(f"State dict keys before load: {list(self.wagering_method.state_dict().keys())}")
                        self.wagering_method.load_state_dict(self.best_wagering_method_state)
                        log.debug(f"State dict keys after load: {list(self.wagering_method.state_dict().keys())}")
                    break
            

            # Early stopping: check for improvement on validation set after each epoch
            # d_regret is a loss metric, so lower is better
            elif (
                (not self.stop_at_last_iteration)
                and self.early_stopping_criterion == "validation"
                and self.early_stopping_patience > 0
                and (
                    val_kl_to_gold is not None
                    if self.use_min_kl_for_early_stopping
                    else (
                        val_brier_d_regret is not None
                        if self.use_brier_d_regret_for_early_stopping
                        else val_d_regret is not None
                    )
                )
            ):
                monitored_metric_name = (
                    "kl_to_gold"
                    if self.use_min_kl_for_early_stopping
                    else ("brier_d_regret" if self.use_brier_d_regret_for_early_stopping else "d_regret")
                )
                monitored_metric_value = (
                    float(val_kl_to_gold)
                    if self.use_min_kl_for_early_stopping
                    else (
                        float(val_brier_d_regret)
                        if self.use_brier_d_regret_for_early_stopping
                        else float(val_d_regret)
                    )
                )
                best_metric_value = (
                    self.best_kl_to_gold
                    if self.use_min_kl_for_early_stopping
                    else (self.best_brier_d_regret if self.use_brier_d_regret_for_early_stopping else self.best_d_regret)
                )

                if monitored_metric_value < best_metric_value:
                    if self.use_min_kl_for_early_stopping:
                        self.best_kl_to_gold = monitored_metric_value
                    elif self.use_brier_d_regret_for_early_stopping:
                        self.best_brier_d_regret = monitored_metric_value
                    else:
                        self.best_d_regret = monitored_metric_value
                    self.epochs_since_improvement = 0
                    # Save the best checkpoint state (in memory and to disk)
                    # IMPORTANT: Use deep copy to avoid reference issues where subsequent
                    # training updates would modify the stored checkpoint state
                    self.best_wagering_method_state = copy.deepcopy(self.wagering_method.state_dict())
                    self.best_epoch = epoch
        
                    log.debug(f"Saving best checkpoint state dict keys: {list(self.best_wagering_method_state.keys())}")

                    best_metric_for_log = (
                        self.best_kl_to_gold
                        if self.use_min_kl_for_early_stopping
                        else (
                            self.best_brier_d_regret
                            if self.use_brier_d_regret_for_early_stopping
                            else self.best_d_regret
                        )
                    )
                    log.debug(
                        "New best %s: %.4f at epoch %d",
                        monitored_metric_name,
                        best_metric_for_log,
                        epoch + 1,
                    )
                else:
                    self.epochs_since_improvement += 1
                
                # Check if we should stop early
                if self.epochs_since_improvement >= self.early_stopping_patience:
                    best_metric_for_log = (
                        self.best_kl_to_gold
                        if self.use_min_kl_for_early_stopping
                        else (
                            self.best_brier_d_regret
                            if self.use_brier_d_regret_for_early_stopping
                            else self.best_d_regret
                        )
                    )
                    log.info(
                        f"Early stopping: No improvement on validation set for {self.early_stopping_patience} epochs. "
                        f"Best validation {monitored_metric_name}: {best_metric_for_log:.4f} (from epoch {self.best_epoch + 1})"
                    )
                    self.early_stopped = True
                    # Load the best checkpoint before breaking
                    if self.best_wagering_method_state is not None:
                        log.info(
                            "Loading best checkpoint from epoch %d (%s=%.4f)",
                            self.best_epoch + 1,
                            monitored_metric_name,
                            best_metric_for_log,
                        )
                        log.debug(f"State dict keys before load: {list(self.wagering_method.state_dict().keys())}")
                        self.wagering_method.load_state_dict(self.best_wagering_method_state)
                        log.debug(f"State dict keys after load: {list(self.wagering_method.state_dict().keys())}")
                    break
            
            # Log epoch-level metrics to wandb
            if self.wandb_logger and len(epoch_predictions) > 0:
                epoch_probs_array = np.stack(epoch_probs)
                
                # Compute ECE for epoch
                epoch_ece = None
                try:
                    ece_metric = ECE(n_bins=20)
                    confidences = epoch_probs_array.max(axis=1)
                    correctness = (np.array(epoch_predictions) == epoch_labels).astype(float)
                    epoch_ece = ece_metric(confidences.tolist(), correctness.tolist())
                except Exception as e:
                    epoch_ece = np.nan
                
                # Compute AUC for epoch
                epoch_auc = None
                max_probs = epoch_probs_array.max(axis=1)
                correctness = (np.array(epoch_predictions) == epoch_labels).astype(int)
                if len(np.unique(correctness)) >= 2:
                    try:
                        epoch_auc = roc_auc_score(correctness, max_probs)
                    except ValueError:
                        epoch_auc = np.nan
                else:
                    epoch_auc = np.nan
                
                # Compute Dynamic Regret and Meta Metrics for epoch
                epoch_d_regret = None
                epoch_meta_acc = None
                epoch_meta_nll = None
                epoch_meta_auc = None
                epoch_kendall_tau = None
                epoch_best_model_mrr = None
                try:
                    # Get epoch model logits in the right format [num_examples, num_models, num_options]
                    epoch_start_idx = self._processed_start_idx
                    epoch_end_idx = epoch_start_idx + len(epoch_predictions)
                    epoch_model_logits_transposed = self.all_model_logits[:, epoch_start_idx:epoch_end_idx, :]  # [num_models, num_examples, num_options]
                    epoch_model_logits = np.transpose(epoch_model_logits_transposed, (1, 0, 2))  # [num_examples, num_models, num_options]
                    epoch_wagers_array = np.array(wagers_history)  # [num_examples, num_models]
                    
                    epoch_d_regret, best_expert_ids = compute_dynamic_regret(
                        epoch_model_logits, epoch_probs_array, epoch_labels
                    )
                    epoch_model_brier_scores = _compute_model_brier_scores(
                        epoch_model_logits,
                        epoch_labels,
                    )
                    meta_metrics = compute_meta_metrics(
                        epoch_wagers_array,
                        best_expert_ids,
                        epoch_model_brier_scores,
                    )
                    epoch_meta_acc = meta_metrics["meta_acc"]
                    epoch_meta_nll = meta_metrics["meta_nll"]
                    epoch_meta_auc = meta_metrics["meta_auc"]
                    epoch_kendall_tau = meta_metrics["kendall_tau"]
                    epoch_best_model_mrr = meta_metrics["best_model_mrr"]
                except Exception as e:
                    log.warning(f"Could not compute epoch d_regret/meta metrics: {e}")
                
                # Log epoch metrics
                wandb_epoch_dict = {
                    "train/epoch/accuracy": epoch_accuracy,
                    "train/epoch/nll": epoch_nll,
                    "train/epoch/ece": epoch_ece if epoch_ece is not None and not np.isnan(epoch_ece) else None,
                    "train/epoch/auc": epoch_auc if epoch_auc is not None and not np.isnan(epoch_auc) else None,
                    "train/epoch/d_regret": epoch_d_regret if epoch_d_regret is not None and not np.isnan(epoch_d_regret) else None,
                    "train/epoch/meta_acc": epoch_meta_acc if epoch_meta_acc is not None and not np.isnan(epoch_meta_acc) else None,
                    "train/epoch/meta_nll": epoch_meta_nll if epoch_meta_nll is not None and not np.isnan(epoch_meta_nll) else None,
                    "train/epoch/meta_auc": epoch_meta_auc if epoch_meta_auc is not None and not np.isnan(epoch_meta_auc) else None,
                    "train/epoch/kendall_tau": epoch_kendall_tau if epoch_kendall_tau is not None and not np.isnan(epoch_kendall_tau) else None,
                    "train/epoch/best_model_mrr": epoch_best_model_mrr if epoch_best_model_mrr is not None and not np.isnan(epoch_best_model_mrr) else None,
                    "train/epoch": epoch + 1,
                }
                
                # Add validation metrics only when validation produced metrics.
                if val_metrics:
                    val_dict_update = {
                        "val/epoch/accuracy": val_metrics.get("accuracy", 0.0),
                        "val/epoch/nll": val_metrics.get("nll", 0.0),
                    }
                    # Only add optional metrics if they're not None/NaN
                    if val_metrics.get("ece") is not None and not np.isnan(val_metrics.get("ece", np.nan)):
                        val_dict_update["val/epoch/ece"] = val_metrics.get("ece")
                    if val_metrics.get("auc") is not None and not np.isnan(val_metrics.get("auc", np.nan)):
                        val_dict_update["val/epoch/auc"] = val_metrics.get("auc")
                    if val_metrics.get("d_regret") is not None and not np.isnan(val_metrics.get("d_regret", np.nan)):
                        val_dict_update["val/epoch/d_regret"] = val_metrics.get("d_regret")
                    if val_metrics.get("meta_acc") is not None and not np.isnan(val_metrics.get("meta_acc", np.nan)):
                        val_dict_update["val/epoch/meta_acc"] = val_metrics.get("meta_acc")
                    if val_metrics.get("meta_nll") is not None and not np.isnan(val_metrics.get("meta_nll", np.nan)):
                        val_dict_update["val/epoch/meta_nll"] = val_metrics.get("meta_nll")
                    if val_metrics.get("meta_auc") is not None and not np.isnan(val_metrics.get("meta_auc", np.nan)):
                        val_dict_update["val/epoch/meta_auc"] = val_metrics.get("meta_auc")
                    if val_metrics.get("kendall_tau") is not None and not np.isnan(val_metrics.get("kendall_tau", np.nan)):
                        val_dict_update["val/epoch/kendall_tau"] = val_metrics.get("kendall_tau")
                    if val_metrics.get("best_model_mrr") is not None and not np.isnan(val_metrics.get("best_model_mrr", np.nan)):
                        val_dict_update["val/epoch/best_model_mrr"] = val_metrics.get("best_model_mrr")
                    if val_nash_gap_max is not None and not np.isnan(val_nash_gap_max):
                        val_dict_update["val/epoch/nash_gap_max"] = val_nash_gap_max
                    
                    wandb_epoch_dict.update(val_dict_update)
                    val_brier_d_regret = val_metrics.get("brier_d_regret")
                    val_brier_d_regret_str = (
                        f"{val_brier_d_regret:.4f}"
                        if val_brier_d_regret is not None and np.isfinite(val_brier_d_regret)
                        else "N/A"
                    )
                    val_best_model_mrr = val_metrics.get("best_model_mrr")
                    val_best_model_mrr_str = (
                        f"{val_best_model_mrr:.4f}"
                        if val_best_model_mrr is not None and np.isfinite(val_best_model_mrr)
                        else "N/A"
                    )
                    log.info(
                        "  Validation accuracy=%.4f, nll=%.4f, brier_d_regret=%s, mrr=%s",
                        val_metrics.get("accuracy", 0.0),
                        val_metrics.get("nll", 0.0),
                        val_brier_d_regret_str,
                        val_best_model_mrr_str,
                    )
                    
                    # Add grouped validation metrics if available
                    if "grouped" in val_metrics:
                        _, slug_names = self._resolve_training_dataset_names()
                        grouped_count = 0
                        for dataset_idx, dataset_metrics in val_metrics["grouped"].items():
                            dataset_key = slug_names[dataset_idx] if dataset_idx < len(slug_names) else f"dataset_{dataset_idx}"
                            grouped_update = {
                                f"val/epoch/{dataset_key}/accuracy": dataset_metrics["accuracy"],
                                f"val/epoch/{dataset_key}/nll": dataset_metrics["nll"],
                                f"val/epoch/{dataset_key}/num_examples": dataset_metrics["num_examples"],
                            }
                            # Only add optional metrics if they're not None/NaN
                            if dataset_metrics.get("ece") is not None and not np.isnan(dataset_metrics.get("ece", np.nan)):
                                grouped_update[f"val/epoch/{dataset_key}/ece"] = dataset_metrics.get("ece")
                            if dataset_metrics.get("auc") is not None and not np.isnan(dataset_metrics.get("auc", np.nan)):
                                grouped_update[f"val/epoch/{dataset_key}/auc"] = dataset_metrics.get("auc")
                            wandb_epoch_dict.update(grouped_update)
                            grouped_count += 1
                elif self.validation_dataset is not None:
                    # Validation was expected but returned no metrics; do not crash training.
                    log.warning(
                        "Validation dataset is available but val_metrics is empty for epoch %d; "
                        "skipping validation logging for this epoch.",
                        epoch + 1,
                    )
                
                # Log to wandb with explicit error handling
                validation_metric_count = sum(1 for k in wandb_epoch_dict if k.startswith('val/'))
                
                try:
                    # Handle both wandb module and wandb.run patterns
                    if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                        # wandb module - use wandb.run.log() which is more reliable
                        self.wandb_logger.run.log(wandb_epoch_dict, step=self.current_step)
                    elif hasattr(self.wandb_logger, 'log'):
                        # Either wandb.run object or mock logger
                        self.wandb_logger.log(wandb_epoch_dict, step=self.current_step)
                    else:
                        raise RuntimeError(f"wandb_logger doesn't have 'log' method. Type: {type(self.wandb_logger)}")
                    
                except Exception as e:
                    raise RuntimeError(f"✗ Error logging to wandb: {e}") from e
            else:
                # Fallback: if the main logging block is skipped, still log validation metrics if available
                if self.wandb_logger and val_metrics:
                    wandb_val_dict = {
                        "val/epoch/accuracy": val_metrics.get("accuracy", 0.0),
                        "val/epoch/nll": val_metrics.get("nll", 0.0),
                    }
                    # Only add optional metrics if they're not None/NaN
                    if val_metrics.get("ece") is not None and not np.isnan(val_metrics.get("ece", np.nan)):
                        wandb_val_dict["val/epoch/ece"] = val_metrics.get("ece")
                    if val_metrics.get("auc") is not None and not np.isnan(val_metrics.get("auc", np.nan)):
                        wandb_val_dict["val/epoch/auc"] = val_metrics.get("auc")
                    if val_metrics.get("d_regret") is not None and not np.isnan(val_metrics.get("d_regret", np.nan)):
                        wandb_val_dict["val/epoch/d_regret"] = val_metrics.get("d_regret")
                    if val_metrics.get("meta_acc") is not None and not np.isnan(val_metrics.get("meta_acc", np.nan)):
                        wandb_val_dict["val/epoch/meta_acc"] = val_metrics.get("meta_acc")
                    if val_metrics.get("meta_nll") is not None and not np.isnan(val_metrics.get("meta_nll", np.nan)):
                        wandb_val_dict["val/epoch/meta_nll"] = val_metrics.get("meta_nll")
                    if val_metrics.get("meta_auc") is not None and not np.isnan(val_metrics.get("meta_auc", np.nan)):
                        wandb_val_dict["val/epoch/meta_auc"] = val_metrics.get("meta_auc")
                    if val_metrics.get("kendall_tau") is not None and not np.isnan(val_metrics.get("kendall_tau", np.nan)):
                        wandb_val_dict["val/epoch/kendall_tau"] = val_metrics.get("kendall_tau")
                    if val_metrics.get("best_model_mrr") is not None and not np.isnan(val_metrics.get("best_model_mrr", np.nan)):
                        wandb_val_dict["val/epoch/best_model_mrr"] = val_metrics.get("best_model_mrr")
                    if val_nash_gap_max is not None and not np.isnan(val_nash_gap_max):
                        wandb_val_dict["val/epoch/nash_gap_max"] = val_nash_gap_max
                    
                    log.debug(f"Fallback path: Logging {len(wandb_val_dict)} validation metrics to wandb")
                    try:
                        # Handle both wandb module and wandb.run patterns
                        if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                            self.wandb_logger.run.log(wandb_val_dict, step=self.current_step)
                        else:
                            self.wandb_logger.log(wandb_val_dict, step=self.current_step)
                        log.debug(f"✓ Successfully logged validation metrics to wandb via fallback path at epoch {epoch + 1}")
                    except Exception as e:
                        raise RuntimeError(f"✗ Error logging validation metrics via fallback: {e}") from e
            
            # Log hyperparameters to wandb config (only once)
            if self.wandb_logger and epoch == 0:
                hyperparams = {}
                
                # Wagering method hyperparameters
                if hasattr(self.wagering_method, 'hidden_dim'):
                    hyperparams["wagering/hidden_dim"] = self.wagering_method.hidden_dim
                if hasattr(self.wagering_method, 'hidden_layers'):
                    hyperparams["wagering/hidden_layers"] = str(self.wagering_method.hidden_layers)
                if hasattr(self.wagering_method, 'hidden_state_layers'):
                    hyperparams["wagering/hidden_state_layers"] = str(self.wagering_method.hidden_state_layers)
                if hasattr(self.wagering_method, 'learning_rate'):
                    hyperparams["wagering/learning_rate"] = self.wagering_method.learning_rate
                if hasattr(self.wagering_method, 'device_str'):
                    hyperparams["wagering/device"] = self.wagering_method.device_str
                
                # Training hyperparameters
                hyperparams["training/num_models"] = len(self.models)
                hyperparams["training/num_datasets"] = len(self.datasets)
                hyperparams["training/shuffle_data"] = self.shuffle_data
                hyperparams["training/shuffle_seed"] = self.shuffle_seed
                hyperparams["training/balance_training_datasets"] = self.balance_training_datasets
                hyperparams["training/early_stopping_patience"] = self.early_stopping_patience
                hyperparams["training/early_stopping_criterion"] = self.early_stopping_criterion
                hyperparams["training/use_brier_d_regret_for_early_stopping"] = self.use_brier_d_regret_for_early_stopping
                hyperparams["training/use_min_kl_for_early_stopping"] = self.use_min_kl_for_early_stopping
                hyperparams["training/save_every"] = self.save_every
                hyperparams["training/batch_size"] = self.batch_size
                hyperparams["training/validation_split_ratio"] = self.validation_split_ratio
                if self.max_training_batches is not None:
                    hyperparams["training/max_training_batches"] = int(self.max_training_batches)
                
                # Aggregation method
                hyperparams["aggregation/name"] = type(self.aggregation_function).__name__
                
                # Log metadata
                if self.metadata:
                    for k, v in self.metadata.items():
                        if k not in ["wagering_config", "aggregation_config"]:
                            hyperparams[f"metadata/{k}"] = v
                
                # Update wandb config
                if hyperparams:
                    # Runtime-resolved fields (e.g., hidden_state_layers) may differ from
                    # initial wandb config values across methods/phases.
                    self.wandb_logger.config.update(hyperparams, allow_val_change=True)
                    log.debug(f"Logged hyperparameters to wandb: {list(hyperparams.keys())}")
        
        # Ensure best checkpoint is loaded for downstream evaluation/checkpoint saving.
        # This also restores the best state after online-learning batch-level early stopping.
        if self.best_wagering_method_state is not None:
            if not self.early_stopped:
                log.debug(
                    "Training completed without early stopping. Loading best checkpoint state "
                    "for final checkpoint saving and evaluation."
                )
            elif self.early_stopping_criterion == "online_learning":
                log.debug("Loading best checkpoint state after online-learning early stopping.")
            self.wagering_method.load_state_dict(self.best_wagering_method_state)

        if self.wager_score_plot_every is not None and self.validation_dataset is not None:
            plot_arrays = self._collect_validation_plot_arrays(max_examples=1000)
            if plot_arrays is not None:
                final_batch_step = max(1, last_completed_batches)
                final_epoch = self.best_epoch if self.best_epoch is not None else max(0, epoch)
                log.info(
                    "Logging final wager plots using best available checkpoint state (epoch=%d, step=%d).",
                    final_epoch + 1,
                    final_batch_step,
                )
                self._plot_val_wagers_vs_score_diff_for_epoch(
                    val_wagers=plot_arrays["wagers"],
                    val_score_diffs=plot_arrays["score_diff"],
                    model_brier_scores=plot_arrays.get("model_brier_scores"),
                    context_assignment_mask=plot_arrays.get("context_assignment_mask"),
                    context_assignment_kind=(
                        str(plot_arrays["context_assignment_kind"][0])
                        if "context_assignment_kind" in plot_arrays
                        else None
                    ),
                    epoch=final_epoch,
                    batch_step=final_batch_step,
                    plot_tag="final",
                )

                if "estimated_score_diff" in plot_arrays:
                    self._plot_val_estimated_score_diff_vs_wagers_for_epoch(
                        val_wagers=plot_arrays["wagers"],
                        val_estimated_score_diffs=plot_arrays["estimated_score_diff"],
                        model_brier_scores=plot_arrays.get("model_brier_scores"),
                        context_assignment_mask=plot_arrays.get("context_assignment_mask"),
                        context_assignment_kind=(
                            str(plot_arrays["context_assignment_kind"][0])
                            if "context_assignment_kind" in plot_arrays
                            else None
                        ),
                        epoch=final_epoch,
                        batch_step=final_batch_step,
                        plot_tag="final",
                    )

                if "scores" in plot_arrays and "estimated_score" in plot_arrays:
                    self._plot_val_own_score_vs_estimated_score_for_epoch(
                        val_own_scores=plot_arrays["scores"],
                        val_estimated_scores=plot_arrays["estimated_score"],
                        model_brier_scores=plot_arrays.get("model_brier_scores"),
                        context_assignment_mask=plot_arrays.get("context_assignment_mask"),
                        context_assignment_kind=(
                            str(plot_arrays["context_assignment_kind"][0])
                            if "context_assignment_kind" in plot_arrays
                            else None
                        ),
                        epoch=final_epoch,
                        batch_step=final_batch_step,
                        plot_tag="final",
                    )

                if "average_scores" in plot_arrays and "estimated_average_scores" in plot_arrays:
                    self._plot_val_average_score_vs_estimated_average_score_for_epoch(
                        val_average_scores=plot_arrays["average_scores"],
                        val_estimated_average_scores=plot_arrays["estimated_average_scores"],
                        model_brier_scores=plot_arrays.get("model_brier_scores"),
                        context_assignment_mask=plot_arrays.get("context_assignment_mask"),
                        context_assignment_kind=(
                            str(plot_arrays["context_assignment_kind"][0])
                            if "context_assignment_kind" in plot_arrays
                            else None
                        ),
                        epoch=final_epoch,
                        batch_step=final_batch_step,
                        plot_tag="final",
                    )
            else:
                log.warning(
                    "Could not collect validation arrays for final wager plots; no final wager plot images were logged."
                )

        if (
            reuse_static_epoch_results
            and requested_num_epochs > effective_num_epochs
            and self.batch_metrics_history
            and self.max_training_batches is None
        ):
            base_epoch_rows = copy.deepcopy(self.batch_metrics_history)
            base_epoch_count = len(base_epoch_rows)
            epochs_to_reuse = requested_num_epochs - effective_num_epochs

            for epoch_offset in range(1, epochs_to_reuse + 1):
                step_offset = int(epoch_offset * num_examples)
                for row_idx, row in enumerate(base_epoch_rows):
                    cloned_row = copy.deepcopy(row)
                    cloned_row["epoch"] = int(cloned_row.get("epoch", 1) + epoch_offset)
                    cloned_row["batch_index_in_epoch"] = int(row_idx + 1)
                    cloned_row["global_step"] = int(cloned_row.get("global_step", 0) + step_offset)
                    self.batch_metrics_history.append(cloned_row)

            self.current_step = int(self.current_step + (epochs_to_reuse * num_examples))
            log.info(
                "Reused %d cached batch-metric epoch(s) for inference-only method (%d base rows -> %d total rows).",
                epochs_to_reuse,
                base_epoch_count,
                len(self.batch_metrics_history),
            )

        # Convert to arrays
        all_predictions = np.array(all_predictions, dtype=np.int32)
        all_aggregated_probs = np.stack(all_aggregated_probs, axis=0)
        wagers_history = np.stack(wagers_history, axis=0)  # [num_examples, num_models]
        
        # Get labels for the examples we actually processed (in case of resume or early stopping)
        # If we resumed, we only processed examples from start_step onwards
        num_processed = len(all_predictions)
        # If we resumed, we need to get the labels for the examples we processed
        # Since we always process sequentially from start_step, we can slice
        if hasattr(self, '_processed_start_idx'):
            start_idx = self._processed_start_idx
            processed_labels = self.labels[start_idx:start_idx + num_processed]
        else:
            # For fresh start or early stopping, we processed examples from 0 to num_processed-1
            processed_labels = self.labels[:num_processed]
        
        # Ensure processed_labels is a numpy array with the correct shape
        processed_labels = np.array(processed_labels, dtype=np.int32)
        
        # Verify shapes match
        if len(all_predictions) != len(processed_labels):
            log.error(
                f"Shape mismatch: all_predictions has {len(all_predictions)} elements, "
                f"but processed_labels has {len(processed_labels)} elements. "
                f"Total dataset size: {len(self.labels)}"
            )
            raise ValueError(
                f"Shape mismatch: predictions ({len(all_predictions)}) vs labels ({len(processed_labels)})"
            )

        if hasattr(self, "_processed_start_idx"):
            _proc_start = int(self._processed_start_idx)
            processed_dataset_indices = self.dataset_indices[_proc_start : _proc_start + num_processed]
            processed_example_local_indices = self.example_local_indices[
                _proc_start : _proc_start + num_processed
            ]
        else:
            processed_dataset_indices = self.dataset_indices[:num_processed]
            processed_example_local_indices = self.example_local_indices[:num_processed]
        
        # Compute final metrics
        accuracy = np.mean(all_predictions == processed_labels)
        
        # Compute NLL (negative log likelihood) for correct classes
        correct_class_probs = all_aggregated_probs[np.arange(len(processed_labels)), processed_labels]
        nll = -np.mean(np.log(correct_class_probs + 1e-10))
        
        # Compute ECE
        ece = None
        try:
            ece_metric = ECE(n_bins=20)
            confidences = all_aggregated_probs.max(axis=1)
            correctness = (all_predictions == processed_labels).astype(float)
            ece = ece_metric(confidences.tolist(), correctness.tolist())
        except Exception as e:
            log.warning(f"Could not compute ECE: {e}")
            ece = np.nan
        
        # Compute AUC
        auc = None
        max_probs = all_aggregated_probs.max(axis=1)
        correctness = (all_predictions == processed_labels).astype(int)
        if len(np.unique(correctness)) >= 2:
            try:
                auc = roc_auc_score(correctness, max_probs)
            except ValueError:
                log.warning("Could not compute AUC (all predictions same class)")
                auc = np.nan
        else:
            auc = np.nan
        
        # Compute Dynamic Regret and Meta Metrics
        d_regret = None
        brier_d_regret = None
        meta_acc = None
        meta_nll = None
        meta_auc = None
        try:
            # Get model logits for processed examples in the right format [num_examples, num_models, num_options]
            if hasattr(self, '_processed_start_idx'):
                start_idx = self._processed_start_idx
                final_model_logits_transposed = self.all_model_logits[:, start_idx:start_idx + num_processed, :]
            else:
                final_model_logits_transposed = self.all_model_logits[:, :num_processed, :]
            
            final_model_logits = np.transpose(final_model_logits_transposed, (1, 0, 2))  # [num_examples, num_models, num_options]
            
            d_regret, best_expert_ids = compute_dynamic_regret(
                final_model_logits, all_aggregated_probs, processed_labels
            )
            final_gold_dist = _build_gold_label_distribution_for_rows(
                processed_labels,
                processed_dataset_indices,
                processed_example_local_indices,
                self.datasets,
                self.option_tokens,
                int(all_aggregated_probs.shape[1]),
            )
            brier_d_regret = compute_brier_dynamic_regret(
                final_model_logits,
                all_aggregated_probs,
                processed_labels,
                gold_label_distribution=final_gold_dist,
            )
            meta_metrics = compute_meta_metrics(wagers_history, best_expert_ids)
            meta_acc = meta_metrics["meta_acc"]
            meta_nll = meta_metrics["meta_nll"]
            meta_auc = meta_metrics["meta_auc"]
        except Exception as e:
            log.warning(f"Could not compute final d_regret/meta metrics: {e}")
        
        # Compute grouped metrics by dataset
        grouped_metrics = self._compute_grouped_metrics(
            all_predictions, all_aggregated_probs, processed_labels,
            processed_dataset_indices, processed_example_local_indices, wagers_history, final_model_logits
        )
        
        results = {
            "predictions": all_predictions,
            "aggregated_probs": all_aggregated_probs,
            "labels": processed_labels,  # Use processed labels, not all labels
            "dataset_indices": processed_dataset_indices,
            "wagers_history": wagers_history,
            "val_nash_gap_history": np.array(val_nash_gap_history, dtype=np.float32),
            "val_d_regret_history": np.array(val_d_regret_history, dtype=np.float32),
            "val_accuracy_history": np.array(val_accuracy_history, dtype=np.float32),
            "val_nash_gap_history_epochs": np.array(val_nash_gap_history_epochs, dtype=np.int32),
            "batch_metrics": batch_metrics,
            "final_accuracy": accuracy,
            "final_nll": nll,
            "final_ece": ece,
            "final_auc": auc,
            "final_d_regret": d_regret,
            "final_brier_d_regret": brier_d_regret,
            "final_meta_acc": meta_acc,
            "final_meta_nll": meta_nll,
            "final_meta_auc": meta_auc,
            "grouped_metrics": grouped_metrics,
        }
        
        # Log grouped metrics
        display_names, slug_names = self._resolve_training_dataset_names()
        log.info("\n=== Training Metrics by Dataset ===")
        for dataset_idx, dataset_metrics in grouped_metrics.items():
            display_name = display_names[dataset_idx] if dataset_idx < len(display_names) else f"dataset_{dataset_idx}"
            log.info(f"{display_name}: accuracy={dataset_metrics['accuracy']:.4f}, "
                f"nll={dataset_metrics['nll']:.4f}, num_examples={dataset_metrics['num_examples']}")
        
        if self.enable_artifact_outputs:
            # Create analytics dataframe
            dataset_size = len(self.combined_dataset.x) if hasattr(self, 'combined_dataset') and self.combined_dataset is not None else None
            analytics_df = WageringAnalytics.create_training_analytics(
                wagering_method=self.wagering_method,
                aggregation_function=self.aggregation_function,
                models=self.models,
                datasets=self.datasets,
                shuffle_data=self.shuffle_data,
                shuffle_seed=self.shuffle_seed,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_criterion=self.early_stopping_criterion,
                use_brier_d_regret_for_early_stopping=self.use_brier_d_regret_for_early_stopping,
                use_min_kl_for_early_stopping=self.use_min_kl_for_early_stopping,
                save_every=self.save_every,
                results=results,
                metadata=self.metadata,
                checkpoint_dir=self.checkpoint_dir,
                dataset_size=dataset_size,
            )
            results["analytics_df"] = analytics_df

            # Save analytics dataframe to checkpoint directory
            if self.checkpoint_dir:
                analytics_path = self.checkpoint_dir / "analytics.csv"
                analytics_df.to_csv(analytics_path, index=False)
                log.debug(f"Saved analytics dataframe to {analytics_path}")

                if len(self.batch_metrics_history) > 0:
                    batch_metrics_df = pd.DataFrame(self.batch_metrics_history)
                    batch_metrics_path = self.checkpoint_dir / "batch_metrics.csv"
                    batch_metrics_df.to_csv(batch_metrics_path, index=False)
                    log.debug(f"Saved batch metrics dataframe to {batch_metrics_path}")
        else:
            results["analytics_df"] = None
        
        # Log final training metrics to wandb
        if self.wandb_logger:
            proposed_final_step = self.current_step + 1 if hasattr(self, 'current_step') else num_epochs * num_examples
            wandb_run_step = None
            if (
                hasattr(self.wandb_logger, 'run')
                and self.wandb_logger.run is not None
                and hasattr(self.wandb_logger.run, 'step')
            ):
                try:
                    run_step_value = self.wandb_logger.run.step
                    if run_step_value is not None:
                        wandb_run_step = int(run_step_value)
                except (TypeError, ValueError):
                    wandb_run_step = None

            final_step = (
                max(proposed_final_step, wandb_run_step + 1)
                if wandb_run_step is not None
                else proposed_final_step
            )
            wandb_final_dict = {
                "train/final/accuracy": accuracy,
                "train/final/nll": nll,
                "train/final/ece": ece if ece is not None and not np.isnan(ece) else None,
                "train/final/auc": auc if auc is not None and not np.isnan(auc) else None,
                "train/final/d_regret": d_regret if d_regret is not None and not np.isnan(d_regret) else None,
                "train/final/brier_d_regret": brier_d_regret if brier_d_regret is not None and not np.isnan(brier_d_regret) else None,
                "train/final/meta_acc": meta_acc if meta_acc is not None and not np.isnan(meta_acc) else None,
                "train/final/meta_nll": meta_nll if meta_nll is not None and not np.isnan(meta_nll) else None,
                "train/final/meta_auc": meta_auc if meta_auc is not None and not np.isnan(meta_auc) else None,
            }
            
            # Add grouped metrics
            _, slug_names = self._resolve_training_dataset_names()
            for dataset_idx, dataset_metrics in grouped_metrics.items():
                dataset_key = slug_names[dataset_idx] if dataset_idx < len(slug_names) else f"dataset_{dataset_idx}"
                wandb_final_dict.update({
                    f"train/final/{dataset_key}/accuracy": dataset_metrics["accuracy"],
                    f"train/final/{dataset_key}/nll": dataset_metrics["nll"],
                    f"train/final/{dataset_key}/ece": dataset_metrics.get("ece"),
                    f"train/final/{dataset_key}/auc": dataset_metrics.get("auc"),
                    f"train/final/{dataset_key}/d_regret": dataset_metrics.get("d_regret"),
                    f"train/final/{dataset_key}/brier_d_regret": dataset_metrics.get("brier_d_regret"),
                    f"train/final/{dataset_key}/num_examples": dataset_metrics["num_examples"],
                })
                
                for model_idx in range(len(self.models)):
                    wager_key = f"avg_wager_model_{model_idx}"
                    if wager_key in dataset_metrics:
                        wandb_final_dict[f"train/final/{dataset_key}/{wager_key}"] = dataset_metrics[wager_key]
            
            try:
                final_plot_step = final_step + 1
                if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                    self.wandb_logger.run.log(wandb_final_dict, step=final_step, commit=True)
                    self.wandb_logger.run.log(wandb_final_dict, step=final_plot_step, commit=True)
                else:
                    self.wandb_logger.log(wandb_final_dict, step=final_step, commit=True)
                    self.wandb_logger.log(wandb_final_dict, step=final_plot_step, commit=True)
                self.current_step = max(self.current_step, final_plot_step)
            except Exception as e:
                raise RuntimeError(f"Error logging train/final metrics to wandb: {e}") from e
        
        # Log final validation metrics to wandb
        if self.wandb_logger:
            final_val_metrics = {}
            final_val_nash_gap = None
            if self.validation_dataset is not None:
                final_val_metrics, final_val_nash_gap, _, _, _  = self._evaluate_validation()

            # Fallback to last available validation metrics if current evaluation is unavailable
            if not final_val_metrics and self.last_val_metrics:
                final_val_metrics = self.last_val_metrics

            if not final_val_metrics:
                if self.validation_dataset is None:
                    log.info("No validation dataset configured; skipping val/final logging.")
                else:
                    log.warning(
                        "Validation dataset is available but final validation metrics are missing; "
                        "skipping val/final logging."
                    )
            else:
                proposed_final_step = self.current_step + 1
                wandb_run_step = None
                if (
                    hasattr(self.wandb_logger, 'run')
                    and self.wandb_logger.run is not None
                    and hasattr(self.wandb_logger.run, 'step')
                ):
                    try:
                        run_step_value = self.wandb_logger.run.step
                        if run_step_value is not None:
                            wandb_run_step = int(run_step_value)
                    except (TypeError, ValueError):
                        wandb_run_step = None

                final_step = (
                    max(proposed_final_step, wandb_run_step + 1)
                    if wandb_run_step is not None
                    else proposed_final_step
                )
                
                wandb_val_final_dict = {
                    "val/final/accuracy": final_val_metrics.get("accuracy", 0.0),
                    "val/final/nll": final_val_metrics.get("nll", 0.0),
                }
                if final_val_metrics.get("ece") is not None and not np.isnan(final_val_metrics.get("ece", np.nan)):
                    wandb_val_final_dict["val/final/ece"] = final_val_metrics.get("ece")
                if final_val_metrics.get("auc") is not None and not np.isnan(final_val_metrics.get("auc", np.nan)):
                    wandb_val_final_dict["val/final/auc"] = final_val_metrics.get("auc")
                if final_val_metrics.get("d_regret") is not None and not np.isnan(final_val_metrics.get("d_regret", np.nan)):
                    wandb_val_final_dict["val/final/d_regret"] = final_val_metrics.get("d_regret")
                if final_val_metrics.get("brier_d_regret") is not None and not np.isnan(final_val_metrics.get("brier_d_regret", np.nan)):
                    wandb_val_final_dict["val/final/brier_d_regret"] = final_val_metrics.get("brier_d_regret")
                if final_val_metrics.get("meta_acc") is not None and not np.isnan(final_val_metrics.get("meta_acc", np.nan)):
                    wandb_val_final_dict["val/final/meta_acc"] = final_val_metrics.get("meta_acc")
                if final_val_metrics.get("meta_nll") is not None and not np.isnan(final_val_metrics.get("meta_nll", np.nan)):
                    wandb_val_final_dict["val/final/meta_nll"] = final_val_metrics.get("meta_nll")
                if final_val_metrics.get("meta_auc") is not None and not np.isnan(final_val_metrics.get("meta_auc", np.nan)):
                    wandb_val_final_dict["val/final/meta_auc"] = final_val_metrics.get("meta_auc")
                if final_val_nash_gap is not None:
                    final_val_nash_gap_max = float(np.max(final_val_nash_gap))
                    if not np.isnan(final_val_nash_gap_max):
                        wandb_val_final_dict["val/final/nash_gap_max"] = final_val_nash_gap_max
                
                if "grouped" in final_val_metrics:
                    _, slug_names = self._resolve_training_dataset_names()
                    for dataset_idx, dataset_metrics in final_val_metrics["grouped"].items():
                        dataset_key = slug_names[dataset_idx] if dataset_idx < len(slug_names) else f"dataset_{dataset_idx}"
                        wandb_val_final_dict.update({
                            f"val/final/{dataset_key}/accuracy": dataset_metrics["accuracy"],
                            f"val/final/{dataset_key}/nll": dataset_metrics["nll"],
                            f"val/final/{dataset_key}/num_examples": dataset_metrics["num_examples"],
                        })
                        if dataset_metrics.get("ece") is not None and not np.isnan(dataset_metrics.get("ece", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/ece"] = dataset_metrics.get("ece")
                        if dataset_metrics.get("auc") is not None and not np.isnan(dataset_metrics.get("auc", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/auc"] = dataset_metrics.get("auc")
                        if dataset_metrics.get("d_regret") is not None and not np.isnan(dataset_metrics.get("d_regret", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/d_regret"] = dataset_metrics.get("d_regret")
                        if dataset_metrics.get("brier_d_regret") is not None and not np.isnan(dataset_metrics.get("brier_d_regret", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/brier_d_regret"] = dataset_metrics.get("brier_d_regret")
                        if dataset_metrics.get("meta_acc") is not None and not np.isnan(dataset_metrics.get("meta_acc", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/meta_acc"] = dataset_metrics.get("meta_acc")
                        if dataset_metrics.get("meta_nll") is not None and not np.isnan(dataset_metrics.get("meta_nll", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/meta_nll"] = dataset_metrics.get("meta_nll")
                        if dataset_metrics.get("meta_auc") is not None and not np.isnan(dataset_metrics.get("meta_auc", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/meta_auc"] = dataset_metrics.get("meta_auc")
                
                try:
                    final_plot_step = final_step + 1
                    if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                        self.wandb_logger.run.log(wandb_val_final_dict, step=final_step, commit=True)
                        self.wandb_logger.run.log(wandb_val_final_dict, step=final_plot_step, commit=True)
                    else:
                        self.wandb_logger.log(wandb_val_final_dict, step=final_step, commit=True)
                        self.wandb_logger.log(wandb_val_final_dict, step=final_plot_step, commit=True)
                    self.current_step = max(self.current_step, final_plot_step)
                except Exception as e:
                    raise RuntimeError(f"Error logging val/final metrics to wandb: {e}") from e
        
        # Plot wagers over time
        if self.enable_artifact_outputs:
            self._plot_wagers_over_time(wagers_history, results)
            self._plot_val_nash_gap_relationships(
                val_nash_gap_history=np.array(val_nash_gap_history, dtype=np.float32),
                val_d_regret_history=np.array(val_d_regret_history, dtype=np.float32),
                val_accuracy_history=np.array(val_accuracy_history, dtype=np.float32),
                val_history_epochs=np.array(val_nash_gap_history_epochs, dtype=np.int32),
            )
        
        return results
    
    def _save_checkpoint(self, epoch: int):
        """Save checkpoint including hidden states for resuming.
        
        Note: Only saves training hidden states, not validation hidden states.
        Validation hidden states are recomputed on-demand and are not needed for resumption.
        """
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{self.current_step}.pt"
        checkpoint = {
            "epoch": epoch,
            "current_step": self.current_step,
            "wagering_method_state": self.wagering_method.state_dict(),
            "wagers_history": getattr(self, 'wagers_history', []),
            "best_d_regret": self.best_d_regret,
            "best_brier_d_regret": self.best_brier_d_regret,
            "best_kl_to_gold": self.best_kl_to_gold,
            "best_nash_gap": self.best_nash_gap,
            "best_batch_nash_gap": self.best_batch_nash_gap,
            "best_batch_brier_d_regret": self.best_batch_brier_d_regret,
            "epochs_since_improvement": self.epochs_since_improvement,
            "batches_since_improvement": self.batches_since_improvement,
            "best_epoch": self.best_epoch,
            "best_batch_step": self.best_batch_step,
        }
        
        # Save training logits if available (for resuming without recollecting)
        # if hasattr(self, 'all_model_logits') and self.all_model_logits is not None:
        #     checkpoint["all_model_logits"] = self.all_model_logits
        
        # # Do NOT save validation logits - they're recomputed on-demand during validation
        # # This saves significant disk space and memory
        
        # # Save training hidden states if available (for resuming without recollecting)
        # if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
        #     if isinstance(self.all_hidden_states, list):
        #         # Save as list of numpy arrays
        #         checkpoint["hidden_states"] = [hs for hs in self.all_hidden_states]
        #         checkpoint["hidden_states_format"] = "list"
        #     else:
        #         # Save as single numpy array
        #         checkpoint["hidden_states"] = self.all_hidden_states
        #         checkpoint["hidden_states_format"] = "array"
        #     log.info("Saved training hidden states to checkpoint for resuming")
        
        # Do NOT save validation hidden states - they're recomputed on-demand during batch validation
        # This saves significant disk space and memory, preventing OOM errors
        
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        save_error: Optional[Exception] = None

        # Retry with legacy serializer when zip writer fails on some filesystems.
        for use_new_zipfile_serialization in (True, False):
            try:
                torch.save(
                    checkpoint,
                    tmp_path,
                    _use_new_zipfile_serialization=use_new_zipfile_serialization,
                )
                tmp_path.replace(checkpoint_path)
                save_error = None
                break
            except Exception as e:
                save_error = e
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                if use_new_zipfile_serialization:
                    log.warning(
                        "Checkpoint save failed using zip serialization for %s. "
                        "Retrying with legacy serialization. Error: %s",
                        checkpoint_path,
                        e,
                    )

        if save_error is not None:
            usage_msg = ""
            try:
                usage = shutil.disk_usage(self.checkpoint_dir)
                usage_msg = (
                    f" (disk free near checkpoint dir: {usage.free / (1024**3):.2f} GiB, "
                    f"used: {usage.used / (1024**3):.2f} GiB)"
                )
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to save checkpoint to {checkpoint_path}{usage_msg}. "
                "This is often caused by filesystem quota/space limits or transient "
                "network filesystem write failures. Consider setting "
                "save_epoch_checkpoints=false or max_epoch_checkpoints to a small value (2-5)."
            ) from save_error

        self._prune_transition_checkpoints()

    def _prune_transition_checkpoints(self):
        """Delete older transition checkpoints if retention is configured."""
        if not self.checkpoint_dir or self.max_epoch_checkpoints is None:
            return

        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        stale_checkpoints = checkpoint_files[self.max_epoch_checkpoints :]
        for stale_path in stale_checkpoints:
            try:
                stale_path.unlink()
            except Exception as e:
                log.warning("Could not remove old checkpoint %s: %s", stale_path, e)
    
    def _plot_validation_wagers_by_dataset(
        self,
        val_wagers: np.ndarray,
        results: Dict[str, Any],
    ):
        """Plot average validation wagers grouped by dataset."""
        if "dataset_indices" not in results or self.checkpoint_dir is None:
            return
        
        dataset_indices = results["dataset_indices"]
        num_datasets = len(self.datasets)
        num_models = val_wagers.shape[1]
        
        # Resolve dataset names
        def _resolve_training_dataset_names():
            names = []
            if isinstance(self.metadata, dict):
                for key in ["training_datasets", "dataset_names", "datasets", "train_datasets"]:
                    v = self.metadata.get(key)
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        names = [str(x) for x in v][: len(self.datasets)]
                        break
                    if isinstance(v, str) and len(self.datasets) == 1:
                        names = [v]
                        break
            if not names:
                inferred = []
                for i, ds in enumerate(self.datasets):
                    ds_name = getattr(ds, "name", None) or getattr(ds, "dataset_name", None) or getattr(ds, "path", None)
                    inferred.append(str(ds_name) if ds_name else f"dataset_{i}")
                names = inferred[: len(self.datasets)]
            if len(names) != len(self.datasets):
                names = [f"dataset_{i}" for i in range(len(self.datasets))]
            return names
        
        # Get model names
        model_names = []
        if isinstance(self.metadata, dict) and "models" in self.metadata:
            raw_names = self.metadata["models"]
            if isinstance(raw_names, (list, tuple)):
                model_names = [str(name) for name in raw_names][:num_models]
        
        if len(model_names) != num_models and getattr(self, "models", None):
            inferred_names = []
            for i, model in enumerate(self.models):
                name = getattr(model, "model_path", None)
                if not name:
                    name = getattr(model, "model_name", None)
                if not name:
                    name = f"Model {i+1}"
                inferred_names.append(str(name))
            model_names = inferred_names[:num_models]
        
        if len(model_names) != num_models:
            model_names = [f"Model {i+1}" for i in range(num_models)]
        
        # Plot average validation wagers per dataset (bar plot)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        dataset_names = _resolve_training_dataset_names()
        x = np.arange(num_datasets)
        width = 0.8 / num_models
        
        for i in range(num_models):
            avg_wagers = []
            for dataset_idx in range(num_datasets):
                mask = dataset_indices == dataset_idx
                if np.any(mask):
                    avg_wager = np.mean(val_wagers[mask, i])
                else:
                    avg_wager = 0.0
                avg_wagers.append(avg_wager)
            
            ax.bar(x + i * width, avg_wagers, width, label=model_names[i], alpha=0.8)
        
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        ax.set_title("Average Wagers by Dataset (Validation)", fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels(dataset_names, rotation=20, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if self.checkpoint_dir:
            avg_save_path = self.checkpoint_dir / "validation_average_wagers_by_dataset.png"
            plt.savefig(avg_save_path, dpi=150, bbox_inches='tight')
            log.debug(f"Saved validation average wagers by dataset plot to {avg_save_path}")
            
            if self.wandb_logger:
                import wandb
                self._log_wandb_plot({"wagers_plot/val/average_by_dataset": wandb.Image(str(avg_save_path))})
        
        plt.close()
    
    def _plot_wagers_over_time(
        self,
        wagers_history: np.ndarray,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
    ):
        """Plot average wagers over time, overall and grouped by dataset."""
        num_examples, num_models = wagers_history.shape
        
        # Prefer human-readable model names from metadata (original config),
        # fall back to model objects' paths, and finally to generic names.
        model_names: List[str] = []
        if isinstance(self.metadata, dict) and "models" in self.metadata:
            raw_names = self.metadata["models"]
            if isinstance(raw_names, (list, tuple)):
                model_names = [str(name) for name in raw_names][:num_models]
        
        # If metadata is missing or length mismatch, try to infer from model objects
        if len(model_names) != num_models and getattr(self, "models", None):
            inferred_names: List[str] = []
            for i, model in enumerate(self.models):
                name = getattr(model, "model_path", None)
                if not name:
                    name = getattr(model, "model_name", None)
                if not name:
                    name = f"Model {i+1}"
                inferred_names.append(str(name))
            model_names = inferred_names[:num_models]
        
        # Final safety fallback
        if len(model_names) != num_models:
            model_names = [f"Model {i+1}" for i in range(num_models)]
        
        # Plot 1: Overall wagers over time
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        time_steps = np.arange(1, num_examples + 1)
        
        for i in range(num_models):
            ax.plot(time_steps, wagers_history[:, i], label=model_names[i], alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Wager (Weight)", fontsize=11)
        ax.set_title("Average Wagers Over Time (All Datasets)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path is None and self.checkpoint_dir:
            save_path = self.checkpoint_dir / "wagers_over_time.png"
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.debug(f"Saved overall wagers plot to {save_path}")
            
            if self.wandb_logger:
                import wandb
                self._log_wandb_plot({"wagers_plot/overall": wandb.Image(str(save_path))})
        
        plt.close()
        
        # Helper: resolve training dataset names for display
        def _resolve_training_dataset_names() -> List[str]:
            names: List[str] = []
            if isinstance(self.metadata, dict):
                for key in ["training_datasets", "dataset_names", "datasets", "train_datasets"]:
                    v = self.metadata.get(key)
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        names = [str(x) for x in v][: len(self.datasets)]
                        break
                    if isinstance(v, str) and len(self.datasets) == 1:
                        names = [v]
                        break
            if not names:
                inferred = []
                for i, ds in enumerate(self.datasets):
                    ds_name = getattr(ds, "name", None) or getattr(ds, "dataset_name", None) or getattr(ds, "path", None)
                    inferred.append(str(ds_name) if ds_name else f"dataset_{i}")
                names = inferred[: len(self.datasets)]
            # Final fallback
            if len(names) != len(self.datasets):
                names = [f"dataset_{i}" for i in range(len(self.datasets))]
            return names

        # Plot 2: Wagers grouped by dataset
        if "dataset_indices" in results:
            dataset_indices = results["dataset_indices"]
            num_datasets = len(self.datasets)
            dataset_names_disp = _resolve_training_dataset_names()
            
            # Create a subplot for each dataset
            fig, axes = plt.subplots(num_datasets, 1, figsize=(10, 4 * num_datasets))
            if num_datasets == 1:
                axes = [axes]
            
            for dataset_idx in range(num_datasets):
                ax = axes[dataset_idx]
                
                # Get mask for this dataset
                mask = dataset_indices == dataset_idx
                if not np.any(mask):
                    continue
                
                # Extract wagers for this dataset
                dataset_wagers = wagers_history[mask]
                dataset_steps = np.arange(1, len(dataset_wagers) + 1)
                
                # Plot wagers for each model
                for i in range(num_models):
                    ax.plot(dataset_steps, dataset_wagers[:, i], label=model_names[i], alpha=0.7, linewidth=1.5)
                
                dataset_name = dataset_names_disp[dataset_idx] if dataset_idx < len(dataset_names_disp) else f"dataset_{dataset_idx}"
                ax.set_xlabel("Training Step (within dataset)", fontsize=10)
                ax.set_ylabel("Wager (Weight)", fontsize=10)
                ax.set_title(f"Wagers Over Time - {dataset_name}", fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
            
            plt.tight_layout()
            
            if self.checkpoint_dir:
                grouped_save_path = self.checkpoint_dir / "wagers_over_time_by_dataset.png"
                plt.savefig(grouped_save_path, dpi=150, bbox_inches='tight')
                log.debug(f"Saved grouped wagers plot to {grouped_save_path}")
                
                if self.wandb_logger:
                    import wandb
                    self._log_wandb_plot({"wagers_plot/by_dataset": wandb.Image(str(grouped_save_path))})
            
            plt.close()
            
            # Plot 3: Average wagers per dataset (bar plot)
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            dataset_names = dataset_names_disp
            x = np.arange(num_datasets)
            width = 0.8 / num_models
            
            for i in range(num_models):
                avg_wagers = []
                for dataset_idx in range(num_datasets):
                    mask = dataset_indices == dataset_idx
                    if np.any(mask):
                        avg_wager = np.mean(wagers_history[mask, i])
                    else:
                        avg_wager = 0.0
                    avg_wagers.append(avg_wager)
                
                ax.bar(x + i * width, avg_wagers, width, label=model_names[i], alpha=0.8)
            
            ax.set_xlabel("Dataset", fontsize=11)
            ax.set_ylabel("Average Wager (Weight)", fontsize=11)
            ax.set_title("Average Wagers by Dataset", fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (num_models - 1) / 2)
            ax.set_xticklabels(dataset_names, rotation=20, ha='right')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.05])
            
            plt.tight_layout()
            
            if self.checkpoint_dir:
                avg_save_path = self.checkpoint_dir / "average_wagers_by_dataset.png"
                plt.savefig(avg_save_path, dpi=150, bbox_inches='tight')
                log.debug(f"Saved average wagers by dataset plot to {avg_save_path}")
                
                if self.wandb_logger:
                    import wandb
                    self._log_wandb_plot({"wagers_plot/average_by_dataset": wandb.Image(str(avg_save_path))})
            
            plt.close()

    def _plot_val_nash_gap_relationships(
        self,
        val_nash_gap_history: np.ndarray,
        val_d_regret_history: np.ndarray,
        val_accuracy_history: np.ndarray,
        val_history_epochs: np.ndarray,
    ):
        """Plot validation Nash-gap relationships against d_regret and accuracy."""
        if self.checkpoint_dir is None:
            return

        if val_nash_gap_history.size == 0:
            log.debug("Skipping val_nash_gap plots: no validation Nash-gap history available")
            return

        # Plot 1: val_nash_gap (x) vs d_regret (y)
        d_regret_mask = np.isfinite(val_nash_gap_history) & np.isfinite(val_d_regret_history)
        if np.any(d_regret_mask):
            x_vals = val_nash_gap_history[d_regret_mask]
            y_vals = val_d_regret_history[d_regret_mask]

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.scatter(x_vals, y_vals, s=32, alpha=0.85)
            ax.set_xlabel("Validation Nash Gap (Mean)", fontsize=11)
            ax.set_ylabel("Validation D-Regret", fontsize=11)
            ax.set_title("Validation D-Regret vs Mean Nash Gap", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path = self.checkpoint_dir / "validation_nash_gap_vs_d_regret.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.debug(f"Saved validation nash_gap vs d_regret plot to {save_path}")

            if self.wandb_logger:
                import wandb
                self._log_wandb_plot({"wagers_plot/val/nash_gap_vs_d_regret": wandb.Image(str(save_path))})

            plt.close()
        else:
            log.debug("Skipping val_nash_gap vs d_regret plot: no finite paired points")

        # Plot 2: val_nash_gap (x) vs accuracy (y)
        accuracy_mask = np.isfinite(val_nash_gap_history) & np.isfinite(val_accuracy_history)
        if np.any(accuracy_mask):
            x_vals = val_nash_gap_history[accuracy_mask]
            y_vals = val_accuracy_history[accuracy_mask]

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.scatter(x_vals, y_vals, s=32, alpha=0.85)
            ax.set_xlabel("Validation Nash Gap (Mean)", fontsize=11)
            ax.set_ylabel("Validation Accuracy", fontsize=11)
            ax.set_title("Validation Accuracy vs Mean Nash Gap", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.0, 1.05])
            plt.tight_layout()

            save_path = self.checkpoint_dir / "validation_nash_gap_vs_accuracy.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.debug(f"Saved validation nash_gap vs accuracy plot to {save_path}")

            if self.wandb_logger:
                import wandb
                self._log_wandb_plot({"wagers_plot/val/nash_gap_vs_accuracy": wandb.Image(str(save_path))})

            plt.close()
        else:
            log.debug("Skipping val_nash_gap vs accuracy plot: no finite paired points")

        if val_history_epochs.size > 0:
            log.debug(
                f"Tracked validation mean Nash gap for {val_history_epochs.size} epochs "
                f"(first epoch={int(val_history_epochs[0])}, last epoch={int(val_history_epochs[-1])})"
            )

    def _plot_val_wagers_vs_score_diff_for_epoch(
        self,
        val_wagers: np.ndarray,
        val_score_diffs: np.ndarray,
        model_brier_scores: Optional[np.ndarray],
        context_assignment_mask: Optional[np.ndarray],
        context_assignment_kind: Optional[str],
        epoch: int,
        batch_step: Optional[int] = None,
        plot_tag: Optional[str] = None,
    ):
        """Scatter plot of validation wagers vs score differences for a specific epoch."""
        self._plot_validation_pair_scatter(
            x_values=val_wagers,
            y_values=val_score_diffs,
            epoch=epoch,
            batch_step=batch_step,
            plot_tag=plot_tag,
            x_label="Validation Wagers (all models × val samples)",
            y_label="Validation Score Diff",
            title_prefix="Validation Score Diff vs Wagers",
            filename_suffix="wagers_vs_score_diff",
            wandb_suffix="wagers_vs_score_diff",
            missing_msg="wagers or score_diff",
            add_diagonal=True,
            model_brier_scores=model_brier_scores,
            context_assignment_mask=context_assignment_mask,
            context_assignment_kind=context_assignment_kind,
        )

    def _plot_val_estimated_score_diff_vs_wagers_for_epoch(
        self,
        val_wagers: np.ndarray,
        val_estimated_score_diffs: np.ndarray,
        model_brier_scores: Optional[np.ndarray],
        context_assignment_mask: Optional[np.ndarray],
        context_assignment_kind: Optional[str],
        epoch: int,
        batch_step: Optional[int] = None,
        plot_tag: Optional[str] = None,
    ):
        """Scatter plot of estimated_score_diff vs wagers for a specific epoch."""
        self._plot_validation_pair_scatter(
            x_values=val_wagers,
            y_values=val_estimated_score_diffs,
            epoch=epoch,
            batch_step=batch_step,
            plot_tag=plot_tag,
            x_label="Validation Wagers (all models × val samples)",
            y_label="Validation Estimated Score Diff",
            title_prefix="Validation Estimated Score Diff vs Wagers",
            filename_suffix="estimated_score_diff_vs_wagers",
            wandb_suffix="estimated_score_diff_vs_wagers",
            missing_msg="wagers or estimated_score_diff",
            add_diagonal=True,
            model_brier_scores=model_brier_scores,
            context_assignment_mask=context_assignment_mask,
            context_assignment_kind=context_assignment_kind,
        )

    def _plot_val_own_score_vs_estimated_score_for_epoch(
        self,
        val_own_scores: np.ndarray,
        val_estimated_scores: np.ndarray,
        model_brier_scores: Optional[np.ndarray],
        context_assignment_mask: Optional[np.ndarray],
        context_assignment_kind: Optional[str],
        epoch: int,
        batch_step: Optional[int] = None,
        plot_tag: Optional[str] = None,
    ):
        """Scatter plot of own scores vs estimated own scores for a specific epoch."""
        self._plot_validation_pair_scatter(
            x_values=val_own_scores,
            y_values=val_estimated_scores,
            epoch=epoch,
            batch_step=batch_step,
            plot_tag=plot_tag,
            x_label="Validation Own Scores",
            y_label="Validation Estimated Own Scores",
            title_prefix="Validation Own Scores vs Estimated Own Scores",
            filename_suffix="own_scores_vs_estimated_score",
            wandb_suffix="own_scores_vs_estimated_score",
            missing_msg="scores or estimated_score",
            add_diagonal=True,
            model_brier_scores=model_brier_scores,
            context_assignment_mask=context_assignment_mask,
            context_assignment_kind=context_assignment_kind,
        )

    def _plot_val_average_score_vs_estimated_average_score_for_epoch(
        self,
        val_average_scores: np.ndarray,
        val_estimated_average_scores: np.ndarray,
        model_brier_scores: Optional[np.ndarray],
        context_assignment_mask: Optional[np.ndarray],
        context_assignment_kind: Optional[str],
        epoch: int,
        batch_step: Optional[int] = None,
        plot_tag: Optional[str] = None,
    ):
        """Scatter plot of average scores vs estimated average scores for a specific epoch."""
        self._plot_validation_pair_scatter(
            x_values=val_average_scores,
            y_values=val_estimated_average_scores,
            epoch=epoch,
            batch_step=batch_step,
            plot_tag=plot_tag,
            x_label="Validation Average Scores",
            y_label="Validation Estimated Average Scores",
            title_prefix="Validation Average Scores vs Estimated Average Scores",
            filename_suffix="average_scores_vs_estimated_average_scores",
            wandb_suffix="average_scores_vs_estimated_average_scores",
            missing_msg="average_scores or estimated_average_scores",
            add_diagonal=True,
            model_brier_scores=model_brier_scores,
            context_assignment_mask=context_assignment_mask,
            context_assignment_kind=context_assignment_kind,
        )

    def _get_model_names_for_plot(self, num_models: int) -> List[str]:
        model_names: List[str] = []
        if isinstance(self.metadata, dict) and "models" in self.metadata:
            raw_names = self.metadata["models"]
            if isinstance(raw_names, (list, tuple)):
                model_names = [str(name) for name in raw_names][:num_models]

        if len(model_names) != num_models and getattr(self, "models", None):
            inferred_names: List[str] = []
            for i, model in enumerate(self.models):
                name = getattr(model, "model_path", None)
                if not name:
                    name = getattr(model, "model_name", None)
                if not name:
                    name = f"Model {i+1}"
                inferred_names.append(str(name))
            model_names = inferred_names[:num_models]

        if len(model_names) != num_models:
            model_names = [f"Model {i+1}" for i in range(num_models)]
        return model_names

    def _get_validation_context_assignment_mask(
        self,
        num_examples: int,
        num_models_total: int,
        dataset_indices: Optional[np.ndarray] = None,
        local_indices: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Return per-example context assignment mask for mixed-context datasets.

        This is used for visualization only (do not drop points). For PubMedQA,
        points without assigned context are shown in gray, while assigned-context
        points are shown in model color.

        Returns:
            (assignment_mask, assignment_kind)
            - assignment_mask: [num_examples, num_models_total] bool, True when model_idx is assigned
            - assignment_kind: "pubmedqa" | "race" | None
        """
        if dataset_indices is None:
            dataset_indices = getattr(self, "validation_dataset_indices", None)
        if local_indices is None:
            local_indices = getattr(self, "validation_example_local_indices", None)
        if dataset_indices is None or local_indices is None:
            return None, None

        dataset_indices_arr = np.asarray(dataset_indices)
        local_indices_arr = np.asarray(local_indices)
        if dataset_indices_arr.shape[0] != num_examples or local_indices_arr.shape[0] != num_examples:
            log.debug(
                "Skipping context-aware masking due to shape mismatch: "
                f"num_examples={num_examples}, dataset_indices={dataset_indices_arr.shape}, "
                f"local_indices={local_indices_arr.shape}"
            )
            return None, None

        assignment_mask = np.ones((num_examples, num_models_total), dtype=bool)
        has_mixed_context_dataset = False
        has_pubmedqa = False
        has_race = False

        for dataset_idx in range(len(self.datasets)):
            dataset_row_mask = dataset_indices_arr == dataset_idx
            if not np.any(dataset_row_mask):
                continue

            dataset = self.datasets[dataset_idx]
            assignment_list = None
            if hasattr(dataset, "pubmedqa_context_assignment_by_example"):
                assignment_list = getattr(dataset, "pubmedqa_context_assignment_by_example", None)
                has_pubmedqa = True
            elif hasattr(dataset, "race_context_assignment_by_example"):
                assignment_list = getattr(dataset, "race_context_assignment_by_example", None)
                has_race = True

            if not isinstance(assignment_list, list) or len(assignment_list) == 0:
                continue

            has_mixed_context_dataset = True

            try:
                assignments = np.asarray(assignment_list, dtype=np.int32)
            except Exception:
                continue

            if assignments.ndim != 1:
                continue

            row_indices = np.flatnonzero(dataset_row_mask)
            row_local_indices = local_indices_arr[row_indices]
            valid_local_idx_mask = (row_local_indices >= 0) & (row_local_indices < assignments.shape[0])
            if not np.any(valid_local_idx_mask):
                continue

            mapped_rows = row_indices[valid_local_idx_mask]
            mapped_models = assignments[row_local_indices[valid_local_idx_mask]]
            valid_model_idx_mask = (mapped_models >= 0) & (mapped_models < num_models_total)
            if not np.any(valid_model_idx_mask):
                continue

            mapped_rows = mapped_rows[valid_model_idx_mask]
            mapped_models = mapped_models[valid_model_idx_mask]
            assignment_mask[mapped_rows, :] = False
            assignment_mask[mapped_rows, mapped_models] = True

        if not has_mixed_context_dataset:
            return None, None
        if has_pubmedqa:
            return assignment_mask, "pubmedqa"
        if has_race:
            return assignment_mask, "race"
        return assignment_mask, None

    def _plot_validation_pair_scatter(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        epoch: int,
        batch_step: Optional[int],
        plot_tag: Optional[str],
        x_label: str,
        y_label: str,
        title_prefix: str,
        filename_suffix: str,
        wandb_suffix: str,
        missing_msg: str,
        add_diagonal: bool = False,
        model_brier_scores: Optional[np.ndarray] = None,
        context_assignment_mask: Optional[np.ndarray] = None,
        context_assignment_kind: Optional[str] = None,
    ):
        if self.checkpoint_dir is None:
            return

        if x_values is None or y_values is None:
            log.debug(f"Skipping epoch {epoch + 1} {filename_suffix} plot: missing {missing_msg}")
            return

        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)
        if x_values.ndim != 2 or y_values.ndim != 2 or x_values.shape != y_values.shape:
            log.debug(
                f"Skipping epoch {epoch + 1} {filename_suffix} plot: shape mismatch "
                f"x={x_values.shape}, y={y_values.shape}"
            )
            return

        num_models = x_values.shape[1]
        model_names = self._get_model_names_for_plot(num_models)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plotted_any = False

        def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            y = np.asarray(y, dtype=np.float64).reshape(-1)
            m = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(m)) < 2:
                return float("nan")
            x = x[m]
            y = y[m]
            x = x - float(np.mean(x))
            y = y - float(np.mean(y))
            denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
            if denom == 0.0:
                return float("nan")
            return float(np.sum(x * y) / denom)

        finite_xy_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if model_brier_scores is not None:
            model_brier_scores = np.asarray(model_brier_scores)
            if model_brier_scores.shape != y_values.shape:
                log.debug(
                    f"Ignoring model_brier_scores for {filename_suffix} plot due to shape mismatch: "
                    f"brier={model_brier_scores.shape}, y={y_values.shape}"
                )
                model_brier_scores = None

        use_pubmedqa_context_coloring = (
            context_assignment_kind == "pubmedqa"
            and context_assignment_mask is not None
            and np.asarray(context_assignment_mask).shape == y_values.shape
        )
        if use_pubmedqa_context_coloring:
            context_assignment_mask = np.asarray(context_assignment_mask, dtype=bool)
        elif model_brier_scores is not None:
            # For non-PubMedQA datasets: color points that achieve best (lowest) per-example Brier score.
            finite_brier_mask = np.isfinite(model_brier_scores)
            per_example_min_brier = np.min(np.where(finite_brier_mask, model_brier_scores, np.inf), axis=1)
            # Use isclose to handle floating precision/ties.
            best_brier_mask = (
                np.isfinite(per_example_min_brier)[:, np.newaxis]
                & finite_brier_mask
                & np.isclose(model_brier_scores, per_example_min_brier[:, np.newaxis], rtol=1e-6, atol=1e-12)
            )
        else:
            best_brier_mask = None

        if use_pubmedqa_context_coloring:
            colored_xy_mask = finite_xy_mask & context_assignment_mask
        elif best_brier_mask is not None:
            colored_xy_mask = finite_xy_mask & best_brier_mask
        else:
            colored_xy_mask = finite_xy_mask

        for model_idx in range(num_models):
            model_x = x_values[:, model_idx]
            model_y = y_values[:, model_idx]
            finite_mask = np.isfinite(model_x) & np.isfinite(model_y)
            if not np.any(finite_mask):
                continue

            if use_pubmedqa_context_coloring:
                assigned_mask = finite_mask & context_assignment_mask[:, model_idx]
                unassigned_mask = finite_mask & (~context_assignment_mask[:, model_idx])

                if np.any(assigned_mask):
                    ax.scatter(
                        model_x[assigned_mask],
                        model_y[assigned_mask],
                        s=14,
                        alpha=0.55,
                        label=model_names[model_idx],
                    )
                if np.any(unassigned_mask):
                    ax.scatter(
                        model_x[unassigned_mask],
                        model_y[unassigned_mask],
                        s=14,
                        color="lightgray",
                        alpha=0.2,
                        label=None,
                    )
            else:
                if best_brier_mask is None:
                    # Fallback: if we have no Brier scores, show everything in model color.
                    ax.scatter(
                        model_x[finite_mask],
                        model_y[finite_mask],
                        s=14,
                        alpha=0.55,
                        label=model_names[model_idx],
                    )
                else:
                    best_mask = finite_mask & best_brier_mask[:, model_idx]
                    non_best_mask = finite_mask & (~best_brier_mask[:, model_idx])

                    if np.any(best_mask):
                        ax.scatter(
                            model_x[best_mask],
                            model_y[best_mask],
                            s=14,
                            alpha=0.55,
                            label=model_names[model_idx],
                        )
                    if np.any(non_best_mask):
                        ax.scatter(
                            model_x[non_best_mask],
                            model_y[non_best_mask],
                            s=14,
                            color="lightgray",
                            alpha=0.2,
                            label=None,
                        )

            plotted_any = True

        if not plotted_any:
            plt.close()
            log.debug(f"Skipping epoch {epoch + 1} {filename_suffix} plot: no finite points")
            return

        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if plot_tag is not None:
            plot_title = f"{title_prefix} ({plot_tag.capitalize()}, Epoch {epoch + 1})"
        elif batch_step is not None:
            plot_title = f"{title_prefix} (Epoch {epoch + 1}, Batch {batch_step})"
        else:
            plot_title = f"{title_prefix} (Epoch {epoch + 1})"
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        pearson_colored = _pearson_r(x_values[colored_xy_mask], y_values[colored_xy_mask])
        pearson_all = _pearson_r(x_values[finite_xy_mask], y_values[finite_xy_mask])
        corr_text = f"Pearson r (colored): {pearson_colored:.3f}"
        fig.suptitle(plot_title, fontsize=12, fontweight='bold', y=0.985)
        fig.text(0.5, 0.942, corr_text, ha="center", va="top", fontsize=11)

        if add_diagonal:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            lo = min(xlim[0], ylim[0])
            hi = max(xlim[1], ylim[1])
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="lightgrey", linewidth=1.2, zorder=0)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        plt.tight_layout()

        if plot_tag is not None:
            save_path = self.checkpoint_dir / f"validation_epoch_{epoch + 1:04d}_{plot_tag}_{filename_suffix}.png"
        elif batch_step is not None:
            save_path = self.checkpoint_dir / (
                f"validation_epoch_{epoch + 1:04d}_batch_{batch_step:07d}_{filename_suffix}.png"
            )
        else:
            save_path = self.checkpoint_dir / f"validation_epoch_{epoch + 1:04d}_{filename_suffix}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.debug(f"Saved epoch {epoch + 1} {filename_suffix} plot to {save_path}")

        if self.wandb_logger:
            import wandb

            if plot_tag is not None:
                wandb_key = f"wagers_plot/val/{wandb_suffix}/{plot_tag}"
            elif batch_step is not None:
                wandb_key = f"wagers_plot/val/{wandb_suffix}/batch_{batch_step}"
            else:
                wandb_key = f"wagers_plot/val/{wandb_suffix}/epoch_{epoch + 1}"
            self._log_wandb_plot({wandb_key: wandb.Image(str(save_path))})

        plt.close()
    
    def save_final_checkpoint(self, save_dir: str) -> str:
        """Save final checkpoint and return the path.
        
        Returns:
            str: Path to the saved checkpoint directory
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure we save the best epoch state if available
        if self.best_wagering_method_state is not None:
            log.debug("Loading best checkpoint state before saving final checkpoint")
            self.wagering_method.load_state_dict(self.best_wagering_method_state)

        # Save wagering method (contains best epoch state if early stopping occurred)
        self.wagering_method.save_pretrained(str(save_dir))
        
        if self.best_epoch is not None:
            log.debug(f"Saved final checkpoint to {save_dir} (best epoch: {self.best_epoch + 1})")
        else:
            log.debug(f"Saved final checkpoint to {save_dir}")
        return str(save_dir)

