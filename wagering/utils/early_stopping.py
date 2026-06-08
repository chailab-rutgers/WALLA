"""Early stopping for wagering training."""

from __future__ import annotations

import copy
import logging
from collections import deque
from typing import Any, Deque, Optional, Tuple

import numpy as np

log = logging.getLogger("wagering")


def _rolling_weighted_mean(window: Deque[Tuple[float, int]]) -> float:
    weighted_sum = 0.0
    total_weight = 0
    for value, weight in window:
        weighted_sum += float(value) * int(weight)
        total_weight += int(weight)
    return weighted_sum / float(max(total_weight, 1))


class WageringEarlyStopping:
    """Validation-epoch and online-learning batch early stopping."""

    def __init__(
        self,
        *,
        patience: int = 10,
        criterion: str = "validation",
        use_brier_d_regret: bool = True,
        use_min_kl: bool = False,
    ):
        self.patience = patience
        self.criterion = str(criterion).strip().lower()
        if self.criterion not in {"validation", "online_learning"}:
            raise ValueError(
                "early_stopping_criterion must be one of {'validation', 'online_learning'}, "
                f"got: {criterion}"
            )
        self.use_brier_d_regret_for_early_stopping = bool(use_brier_d_regret)
        self.use_min_kl_for_early_stopping = bool(use_min_kl)
        if self.use_min_kl_for_early_stopping and self.use_brier_d_regret_for_early_stopping:
            raise ValueError(
                "Only one early-stopping metric override may be enabled at a time. "
                "Set at most one of use_brier_d_regret_for_early_stopping / use_min_kl_for_early_stopping."
            )
        if self.use_min_kl_for_early_stopping and self.criterion not in {"validation", "online_learning"}:
            raise ValueError(
                "use_min_kl_for_early_stopping=True requires early_stopping_criterion in "
                "{'validation', 'online_learning'}"
            )

        self.best_d_regret = float("inf")
        self.best_brier_d_regret = float("inf")
        self.best_kl_to_gold = float("inf")
        self.best_batch_brier_d_regret = float("inf")
        self.best_batch_kl_to_gold = float("inf")
        self.epochs_since_improvement = 0
        self.batches_since_improvement = 0
        self.early_stopped = False
        self.best_wagering_method_state: Optional[dict] = None
        self.best_epoch: Optional[int] = None
        self.best_batch_step: Optional[int] = None

        self._online_window_batches = 1
        self._online_window_target_examples = 1
        self._validation_examples_for_log = 0
        self._online_metric_window: Deque[Tuple[float, int]] = deque(maxlen=1)

    def setup_online_window(
        self,
        *,
        num_training_examples: int,
        batch_size: int,
        validation_split_ratio: float,
        validation_dataset_size: Optional[int],
    ) -> None:
        if self.criterion != "online_learning":
            return

        if validation_dataset_size is not None:
            validation_examples = validation_dataset_size
        else:
            validation_examples = int(round(num_training_examples * validation_split_ratio))

        if validation_examples <= 0:
            validation_examples = batch_size

        self._validation_examples_for_log = validation_examples
        self._online_window_batches = max(1, int(round(validation_examples / float(batch_size))))
        self._online_window_target_examples = self._online_window_batches * batch_size
        self._online_metric_window = deque(maxlen=self._online_window_batches)

    def log_enabled(self) -> None:
        if self.patience <= 0:
            return

        if self.criterion == "online_learning":
            metric_name = (
                "rolling-window kl_to_gold"
                if self.use_min_kl_for_early_stopping
                else (
                    "rolling-window brier_d_regret"
                    if self.use_brier_d_regret_for_early_stopping
                    else "rolling-window metric (unsupported)"
                )
            )
            log.info(
                "Early stopping enabled: criterion=online_learning, metric=%s, "
                "patience=%d batches, window=%d batches (~%d examples, validation=%d)",
                metric_name,
                self.patience,
                self._online_window_batches,
                self._online_window_target_examples,
                self._validation_examples_for_log,
            )
            return

        if self.use_min_kl_for_early_stopping:
            metric_name = "validation kl_to_gold"
        else:
            metric_name = (
                "validation brier_d_regret"
                if self.use_brier_d_regret_for_early_stopping
                else "validation d_regret"
            )
        log.info(
            "Early stopping enabled: criterion=validation, metric=%s, patience=%d epochs",
            metric_name,
            self.patience,
        )

    def _monitored_metric_name(self) -> str:
        if self.use_min_kl_for_early_stopping:
            return "kl_to_gold"
        if self.use_brier_d_regret_for_early_stopping:
            return "brier_d_regret"
        return "d_regret"

    def _best_validation_metric(self) -> float:
        if self.use_min_kl_for_early_stopping:
            return self.best_kl_to_gold
        if self.use_brier_d_regret_for_early_stopping:
            return self.best_brier_d_regret
        return self.best_d_regret

    def _set_best_validation_metric(self, value: float) -> None:
        if self.use_min_kl_for_early_stopping:
            self.best_kl_to_gold = value
        elif self.use_brier_d_regret_for_early_stopping:
            self.best_brier_d_regret = value
        else:
            self.best_d_regret = value

    def require_finite_validation_kl(self, val_kl_to_gold: Optional[float]) -> None:
        if not self.use_min_kl_for_early_stopping:
            return
        if val_kl_to_gold is None or not np.isfinite(float(val_kl_to_gold)):
            raise RuntimeError(
                "use_min_kl_for_early_stopping=True requires a finite validation "
                "kl_to_gold metric. This metric is only computed for datasets with "
                "soft probabilistic labels (probability_label_column / dataset.probabilistic_labels)."
            )
    def should_track_online_batch(self) -> bool:
        return self.criterion == "online_learning" and self.patience > 0

    def update_online_batch(
        self,
        *,
        epoch: int,
        batch_end: int,
        num_examples: int,
        batch_size_actual: int,
        batch_brier_d_regret: Optional[float],
        batch_kl_to_gold: Optional[float],
        batch_soft_label_count: int,
        checkpoint_state: dict,
    ) -> bool:
        """Update online-learning early stopping. Returns True if training should stop."""
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

            self._online_metric_window.append((float(batch_kl_to_gold), int(batch_soft_label_count)))
            if len(self._online_metric_window) < self._online_window_batches:
                improved = False
                current_batch_metric = None
            else:
                current_batch_metric = _rolling_weighted_mean(self._online_metric_window)
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

            self._online_metric_window.append((float(batch_brier_d_regret), batch_size_actual))
            if len(self._online_metric_window) < self._online_window_batches:
                improved = False
                current_batch_metric = None
            else:
                current_batch_metric = _rolling_weighted_mean(self._online_metric_window)
                improved = current_batch_metric < self.best_batch_brier_d_regret
            if improved:
                self.best_batch_brier_d_regret = current_batch_metric
        else:
            raise RuntimeError("Not implemented")

        if current_batch_metric is None:
            return False

        if improved:
            self.batches_since_improvement = 0
            self.best_wagering_method_state = copy.deepcopy(checkpoint_state)
            self.best_epoch = epoch
            self.best_batch_step = epoch * num_examples + batch_end
        else:
            self.batches_since_improvement += 1

        if self.batches_since_improvement < self.patience:
            return False

        if self.use_min_kl_for_early_stopping:
            log.info(
                "Early stopping (online_learning): rolling-window kl_to_gold "
                "(window=%d batches) did not improve for %d batches. "
                "Best rolling kl_to_gold: %.6f%s",
                self._online_window_batches,
                self.patience,
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
                self._online_window_batches,
                self.patience,
                self.best_batch_brier_d_regret,
                (
                    f" (epoch {self.best_epoch + 1}, step {self.best_batch_step})"
                    if self.best_epoch is not None and self.best_batch_step is not None
                    else ""
                ),
            )
        self.early_stopped = True
        return True

    def should_track_validation_epoch(
        self,
        *,
        val_d_regret: Optional[float],
        val_brier_d_regret: Optional[float],
        val_kl_to_gold: Optional[float],
    ) -> bool:
        if self.criterion != "validation" or self.patience <= 0:
            return False
        if self.use_min_kl_for_early_stopping:
            return val_kl_to_gold is not None
        if self.use_brier_d_regret_for_early_stopping:
            return val_brier_d_regret is not None
        return val_d_regret is not None

    def update_validation_epoch(
        self,
        *,
        epoch: int,
        val_d_regret: Optional[float],
        val_brier_d_regret: Optional[float],
        val_kl_to_gold: Optional[float],
        wagering_method: Any,
        checkpoint_state: dict,
    ) -> bool:
        """Update validation early stopping. Returns True if training should stop."""
        monitored_metric_name = self._monitored_metric_name()
        if self.use_min_kl_for_early_stopping:
            monitored_metric_value = float(val_kl_to_gold)
        elif self.use_brier_d_regret_for_early_stopping:
            monitored_metric_value = float(val_brier_d_regret)
        else:
            monitored_metric_value = float(val_d_regret)

        best_metric_value = self._best_validation_metric()

        if monitored_metric_value < best_metric_value:
            self._set_best_validation_metric(monitored_metric_value)
            self.epochs_since_improvement = 0
            self.best_wagering_method_state = copy.deepcopy(checkpoint_state)
            self.best_epoch = epoch
            log.debug(
                "Saving best checkpoint state dict keys: %s",
                list(self.best_wagering_method_state.keys()),
            )
            log.debug(
                "New best %s: %.4f at epoch %d",
                monitored_metric_name,
                self._best_validation_metric(),
                epoch + 1,
            )
        else:
            self.epochs_since_improvement += 1

        if self.epochs_since_improvement < self.patience:
            return False

        best_metric_for_log = self._best_validation_metric()
        log.info(
            "Early stopping: No improvement on validation set for %d epochs. "
            "Best validation %s: %.4f (from epoch %d)",
            self.patience,
            monitored_metric_name,
            best_metric_for_log,
            self.best_epoch + 1,
        )
        self.early_stopped = True
        if self.best_wagering_method_state is not None:
            log.info(
                "Loading best checkpoint from epoch %d (%s=%.4f)",
                self.best_epoch + 1,
                monitored_metric_name,
                best_metric_for_log,
            )
            log.debug(
                "State dict keys before load: %s",
                list(wagering_method.state_dict().keys()),
            )
            wagering_method.load_state_dict(self.best_wagering_method_state)
            log.debug(
                "State dict keys after load: %s",
                list(wagering_method.state_dict().keys()),
            )
        return True

    def restore_best_checkpoint(self, wagering_method: Any) -> None:
        if self.best_wagering_method_state is None:
            return
        if not self.early_stopped:
            log.debug(
                "Training completed without early stopping. Loading best checkpoint state "
                "for final checkpoint saving and evaluation."
            )
        elif self.criterion == "online_learning":
            log.debug("Loading best checkpoint state after online-learning early stopping.")
        wagering_method.load_state_dict(self.best_wagering_method_state)
