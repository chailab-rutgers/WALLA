"""Shared wandb step tracking and logging for training and evaluation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from wagering.utils.wagering_metrics import (
    ECE,
    compute_inverse_hhi,
    compute_meta_metrics,
    compute_model_brier_scores,
)

log = logging.getLogger("wagering")


def get_run_step(logger: Any) -> Optional[int]:
    """Return the active wandb run step, if available."""
    if not logger:
        return None
    if hasattr(logger, "run") and logger.run is not None:
        run = logger.run
        if hasattr(run, "step") and run.step is not None:
            return int(run.step)
    return None


def log_to_wandb(
    logger: Any,
    metrics: Dict[str, Any],
    step: int,
    *,
    commit: bool = False,
) -> None:
    """Log metrics to wandb at the given step."""
    if not logger:
        return
    if hasattr(logger, "run") and hasattr(logger.run, "log"):
        logger.run.log(metrics, step=step, commit=commit)
    elif hasattr(logger, "log"):
        logger.log(metrics, step=step, commit=commit)
    else:
        raise RuntimeError(f"wandb_logger has no log method: {type(logger)}")


def advance_step(logger: Any, step: int) -> int:
    """Return the next monotonic step (at least step+1 and run.step+1)."""
    next_step = step + 1
    run_step = get_run_step(logger)
    if run_step is not None:
        next_step = max(next_step, run_step + 1)
    return next_step


def sync_step_from_run(logger: Any, step: int) -> int:
    """Keep local step aligned with wandb run.step after commit=True logs."""
    run_step = get_run_step(logger)
    if run_step is not None:
        return max(step, run_step)
    return step


def resolve_initial_step(logger: Any, starting_step: Optional[int] = None) -> int:
    """Initial monotonic step when resuming training or evaluation."""
    if starting_step is not None:
        step = int(starting_step)
        run_step = get_run_step(logger)
        return max(step, run_step) if run_step is not None else step
    run_step = get_run_step(logger)
    return run_step if run_step is not None else 0


def log_final_metrics(logger: Any, step: int, metrics: Dict[str, Any]) -> int:
    """Log final metrics at two consecutive steps; return the last step used."""
    proposed = step + 1
    run_step = get_run_step(logger)
    final_step = max(proposed, run_step + 1) if run_step is not None else proposed
    final_plot_step = final_step + 1
    log_to_wandb(logger, metrics, final_step, commit=True)
    log_to_wandb(logger, metrics, final_plot_step, commit=True)
    return final_plot_step


def log_plot_payload(logger: Any, step: int, payload: Dict[str, Any]) -> int:
    """Log a plot/image payload with a monotonically increasing step."""
    if not logger:
        return step
    step = advance_step(logger, step)
    log_to_wandb(logger, payload, step, commit=True)
    return sync_step_from_run(logger, step)


def _maybe_metric(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


def _val_epoch_wandb_metrics(val_metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "val/epoch/accuracy": val_metrics.get("accuracy", 0.0),
        "val/epoch/nll": val_metrics.get("nll", 0.0),
    }
    for key, wandb_key in (
        ("ece", "val/epoch/ece"),
        ("auc", "val/epoch/auc"),
        ("d_regret", "val/epoch/d_regret"),
        ("kendall_tau", "val/epoch/kendall_tau"),
        ("best_model_mrr", "val/epoch/best_model_mrr"),
    ):
        v = val_metrics.get(key)
        if v is not None and not np.isnan(v):
            out[wandb_key] = v
    return out


def log_train_epoch(
    logger: Any,
    step: int,
    *,
    epoch: int,
    epoch_accuracy: float,
    epoch_nll: float,
    epoch_predictions: Sequence[int],
    epoch_probs: Sequence[np.ndarray],
    epoch_labels: Sequence[int],
    all_model_logits: np.ndarray,
    wagers_history: Sequence,
    val_metrics: Optional[Dict[str, Any]],
    validation_dataset_configured: bool,
) -> None:
    """Compute and log per-epoch train/val metrics to wandb."""
    if not logger or len(epoch_predictions) == 0:
        return

    epoch_probs_array = np.stack(epoch_probs)
    ece_metric = ECE(n_bins=20)
    confidences = epoch_probs_array.max(axis=1)
    correctness = (np.array(epoch_predictions) == epoch_labels).astype(float)
    epoch_ece = ece_metric(confidences.tolist(), correctness.tolist())

    max_probs = epoch_probs_array.max(axis=1)
    correctness_int = (np.array(epoch_predictions) == epoch_labels).astype(int)
    epoch_auc = roc_auc_score(correctness_int, max_probs)

    epoch_model_logits_transposed = all_model_logits[:, : len(epoch_predictions), :]
    epoch_model_logits = np.transpose(epoch_model_logits_transposed, (1, 0, 2))
    epoch_wagers_array = np.array(wagers_history)

    epoch_model_brier_scores = compute_model_brier_scores(epoch_model_logits, epoch_labels)
    meta_metrics = compute_meta_metrics(
        epoch_wagers_array, epoch_model_brier_scores
    )

    wandb_epoch_dict = {
        "train/epoch/accuracy": epoch_accuracy,
        "train/epoch/nll": epoch_nll,
        "train/epoch/ece": _maybe_metric(epoch_ece),
        "train/epoch/auc": _maybe_metric(epoch_auc),
        "train/epoch/kendall_tau": _maybe_metric(meta_metrics["kendall_tau"]),
        "train/epoch/best_model_mrr": _maybe_metric(meta_metrics["best_model_mrr"]),
        "train/epoch": epoch + 1,
    }

    if val_metrics:
        wandb_epoch_dict.update(_val_epoch_wandb_metrics(val_metrics))
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
    elif validation_dataset_configured:
        raise RuntimeError(
            f"Validation dataset is configured but val_metrics is empty for epoch {epoch + 1}"
        )

    log_to_wandb(logger, wandb_epoch_dict, step)


def log_train_final(
    logger: Any,
    step: int,
    *,
    accuracy: float,
    nll: float,
    ece: Any,
    auc: Any,
    brier_d_regret: Any,
) -> int:
    if not logger:
        return step
    return log_final_metrics(
        logger,
        step,
        {
            "train/final/accuracy": accuracy,
            "train/final/nll": nll,
            "train/final/ece": _maybe_metric(ece),
            "train/final/auc": _maybe_metric(auc),
            "train/final/brier_d_regret": _maybe_metric(brier_d_regret),
        },
    )


def log_val_final(
    logger: Any,
    step: int,
    *,
    validation_dataset: Any,
    get_val_metrics: Callable[[], Tuple[Dict[str, Any], Any, Any, Any]],
) -> int:
    if not logger:
        return step
    if validation_dataset is None:
        log.info("No validation dataset configured; skipping val/final logging.")
        return step

    final_val_metrics, _, _, _ = get_val_metrics()
    if not final_val_metrics:
        raise RuntimeError(
            "Validation dataset is configured but final validation metrics are missing"
        )

    wandb_val_final_dict = {
        "val/final/accuracy": final_val_metrics.get("accuracy", 0.0),
        "val/final/nll": final_val_metrics.get("nll", 0.0),
    }
    for key, wandb_key in (
        ("ece", "val/final/ece"),
        ("auc", "val/final/auc"),
        ("d_regret", "val/final/d_regret"),
        ("brier_d_regret", "val/final/brier_d_regret"),
    ):
        v = final_val_metrics.get(key)
        if v is not None and not np.isnan(v):
            wandb_val_final_dict[wandb_key] = v

    return log_final_metrics(logger, step, wandb_val_final_dict)


def eval_log_prefix(dataset_name: str) -> str:
    return "ood" if dataset_name.startswith("ood_") else "test"


def log_eval_batch(
    tracker: WandbStepTracker,
    *,
    dataset_name: str,
    batch_correct: np.ndarray,
    batch_nll: np.ndarray,
    batch_wagers: np.ndarray,
    running_accuracy: float,
    running_nll: float,
    inference_time_s: float,
) -> None:
    if not tracker.active():
        return

    log_prefix = eval_log_prefix(dataset_name)
    batch_inverse_hhi = compute_inverse_hhi(batch_wagers)

    wandb_log_dict = {
        f"{log_prefix}/{dataset_name}/batch/accuracy": float(np.mean(batch_correct)),
        f"{log_prefix}/{dataset_name}/batch/nll": float(np.mean(batch_nll)),
        f"{log_prefix}/{dataset_name}/batch/running_accuracy": running_accuracy,
        f"{log_prefix}/{dataset_name}/batch/running_nll": running_nll,
        f"{log_prefix}/{dataset_name}/batch/inference_time_s": float(inference_time_s),
        f"{log_prefix}/{dataset_name}/batch/inverse_hhi": batch_inverse_hhi,
    }
    for i in range(batch_wagers.shape[1]):
        wandb_log_dict[f"{log_prefix}/{dataset_name}/batch/wager_model_{i}"] = float(
            np.mean(batch_wagers[:, i])
        )
    tracker.log(wandb_log_dict, step=tracker.advance())


def log_eval_final(
    tracker: WandbStepTracker,
    *,
    dataset_name: str,
    inverse_hhi: float,
    avg_inference_time_per_batch_s: float,
    accuracy: float,
    nll: float,
    brier: Any,
    bernoulli_kl: Any,
    bernoulli_tv: Any,
    auc: Any,
    ece: Any,
    brier_d_regret: Any,
    kendall_tau: Any,
    best_model_mrr: Any,
    brier_best_wager_prob_mean: Any,
    brier_best_wager_prob_var: Any,
    wagers_history: np.ndarray,
    wager_prob_mean_per_model: Optional[np.ndarray],
    wager_prob_var_per_model: Optional[np.ndarray],
) -> None:
    if not tracker.active():
        return

    log_prefix = eval_log_prefix(dataset_name)
    log.debug(
        "Logging final metrics to wandb: prefix=%s, dataset_name=%s",
        log_prefix,
        dataset_name,
    )

    metric_values = {
        "inverse_hhi": inverse_hhi,
        "avg_inference_time_per_batch_s": avg_inference_time_per_batch_s,
        "accuracy": accuracy,
        "nll": nll,
        "brier": _maybe_metric(brier),
        "bernoulli_kl": _maybe_metric(bernoulli_kl),
        "bernoulli_tv": _maybe_metric(bernoulli_tv),
        "auc": _maybe_metric(auc),
        "ece": _maybe_metric(ece),
        "brier_d_regret": _maybe_metric(brier_d_regret),
        "kendall_tau": _maybe_metric(kendall_tau),
        "best_model_mrr": _maybe_metric(best_model_mrr),
        "brier_best_wager_prob_mean": brier_best_wager_prob_mean,
        "brier_best_wager_prob_var": brier_best_wager_prob_var,
    }

    primary_final_prefix = f"{log_prefix}/{dataset_name}/final"
    alias_final_prefix = f"{log_prefix}/final/{dataset_name}"
    wandb_metrics = {f"{primary_final_prefix}/{k}": v for k, v in metric_values.items()}
    wandb_metrics.update({f"{alias_final_prefix}/{k}": v for k, v in metric_values.items()})

    avg_wagers = np.mean(wagers_history, axis=0)
    for model_idx, avg_wager in enumerate(avg_wagers):
        wager_value = float(avg_wager)
        wandb_metrics[f"{primary_final_prefix}/avg_wager_model_{model_idx}"] = wager_value
        wandb_metrics[f"{alias_final_prefix}/avg_wager_model_{model_idx}"] = wager_value

    if wager_prob_mean_per_model is not None and wager_prob_var_per_model is not None:
        for model_idx in range(len(wager_prob_mean_per_model)):
            m = float(wager_prob_mean_per_model[model_idx])
            v = float(wager_prob_var_per_model[model_idx])
            wandb_metrics[f"{primary_final_prefix}/wager_prob_mean_model_{model_idx}"] = m
            wandb_metrics[f"{alias_final_prefix}/wager_prob_mean_model_{model_idx}"] = m
            wandb_metrics[f"{primary_final_prefix}/wager_prob_var_model_{model_idx}"] = v
            wandb_metrics[f"{alias_final_prefix}/wager_prob_var_model_{model_idx}"] = v

    tracker.log_final(wandb_metrics)


def log_eval_image(tracker: WandbStepTracker, key: str, image_path: str, *, advance: bool = True) -> None:
    if not tracker.active():
        return
    import wandb

    payload = {key: wandb.Image(str(image_path))}
    if advance:
        tracker.log_plot(payload)
    else:
        tracker.log(payload, step=tracker.step)


def log_eval_dataset_plot(tracker: WandbStepTracker, dataset_name: str, plot_name: str, image_path: str) -> None:
    log_prefix = eval_log_prefix(dataset_name)
    log_eval_image(tracker, f"{log_prefix}/{dataset_name}/{plot_name}", image_path)


def log_eval_multi_dataset_plot(tracker: WandbStepTracker, eval_type: str, image_path: str) -> None:
    if not tracker.active():
        return
    import wandb

    tracker.log_plot(
        {
            f"wagers_plot/{eval_type}/average_by_dataset": wandb.Image(str(image_path)),
            f"wagers_plot/average_by_dataset/{eval_type}": wandb.Image(str(image_path)),
        }
    )


class WandbStepTracker:
    """Monotonic step counter tied to a wandb logger."""

    def __init__(self, logger: Any, *, initial_step: int = 0) -> None:
        self.logger = logger
        self.step = initial_step

    def active(self) -> bool:
        return self.logger is not None

    def run_step(self) -> Optional[int]:
        return get_run_step(self.logger)

    def align_to_run(self) -> None:
        run_step = self.run_step()
        if run_step is not None and self.step < run_step:
            log.info(
                "Aligning wandb step from %d to active run step %d to keep logging monotonic",
                self.step,
                run_step,
            )
            self.step = run_step

    def advance(self) -> int:
        self.step = advance_step(self.logger, self.step)
        return self.step

    def log(self, metrics: Dict[str, Any], *, step: Optional[int] = None, commit: bool = False) -> None:
        if not self.active():
            return
        log_to_wandb(self.logger, metrics, self.step if step is None else step, commit=commit)

    def log_plot(self, metrics: Dict[str, Any]) -> None:
        if not self.active():
            return
        plot_step = self.advance()
        log_to_wandb(self.logger, metrics, plot_step, commit=True)
        self.step = sync_step_from_run(self.logger, self.step)

    def log_final(self, metrics: Dict[str, Any]) -> None:
        if not self.active():
            return
        self.step = log_final_metrics(self.logger, self.step, metrics)
