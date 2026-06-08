"""
Training pipeline for multi-LLM wagering methods.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence
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
from wagering.utils.cache_manager import (
    collect_stacked_model_artifacts,
    compute_all_prompt_perplexities,
    unload_ensemble_whitebox_models,
)
from wagering.utils.prompt_manager import (
    get_mixed_context_dataset_type,
    get_model_specific_prompts,
)
from wagering.utils.dataset_utils import apply_shuffling

log = logging.getLogger("wagering")

import re

from sklearn.metrics import roc_auc_score

from wagering.utils.wagering_plots import WageringPlotter, get_validation_context_assignment_mask
from wagering.utils.early_stopping import WageringEarlyStopping
from wagering.utils.wandb_logging import (
    get_run_step,
    log_plot_payload,
    log_train_epoch,
    log_train_final,
    log_val_final,
)
from wagering.utils.wagering_metrics import (
    ECE,
    build_gold_label_distribution_for_rows,
    compute_brier_dynamic_regret,
    compute_mean_kl_to_gold_distribution,
    compute_meta_metrics,
    compute_model_bernoulli_kl_to_gt_scores,
    compute_model_brier_scores,
    is_cluster_saturation_dataset_name,
    resolve_positive_option_index,
)


class WageringTrainer:
    """
    Trainer for multi-LLM wagering methods.
    
    Handles training loop, logging, checkpointing, and evaluation.
    """
    
    def __init__(
        self,
        models: List[WhiteboxModel],
        dataset: Dataset,
        wagering_method: WageringMethod,
        aggregation_function: AggregationFunction,
        option_tokens: List[str] = ["A", "B", "C", "D"],
        checkpoint_dir: Optional[str] = None,
        wandb_logger: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        shuffle_data: bool = True,
        shuffle_seed: int = 42,
        early_stopping_patience: int = 10,
        batch_size: int = 100,  # Batch size for training loop
        validation_split_ratio: float = 0.1,  # Fraction of data to use for validation (default: 10%)
        early_stopping_criterion: str = "validation",
        use_brier_d_regret_for_early_stopping: bool = True,
        use_min_kl_for_early_stopping: bool = False,
        wager_score_plot_every: Optional[int] = None,
        logit_calibrator: Optional[Any] = None,
        max_training_batches: Optional[int] = None,
        model_configs_for_sequential_perplexity: Optional[List[Dict[str, Any]]] = None,
        perplexity_load_cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            models: List of WhiteboxModel instances
            dataset: Training Dataset instance
            wagering_method: WageringMethod instance
            aggregation_function: AggregationFunction instance
            option_tokens: List of option tokens (e.g., ["A", "B", "C", "D"])
            checkpoint_dir: Directory for saving checkpoints
            wandb_logger: Optional wandb logger
            early_stopping_patience: Number of non-improving intervals before stopping.
                Uses epochs for ``validation`` criterion and batches for ``online_learning`` criterion.
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
            max_training_batches: If set, stop after this many training-loop batches
                (optimizer steps) across epochs in this ``train()`` call.
            model_configs_for_sequential_perplexity: Merged per-model YAML dicts; used
                to load one model at a time for prompt perplexity when visible GPUs
                are fewer than ensemble slots.
            perplexity_load_cache_kwargs: Optional kwargs (e.g. ``cache_dir``) for
                those sequential loads.
        """
        self.models = models
        self.dataset = dataset
        self.wagering_method = wagering_method
        self.aggregation_function = aggregation_function
        self.option_tokens = option_tokens
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.wandb_logger = wandb_logger
        self.metadata = metadata or {}
        self.shuffle_data = shuffle_data
        self.shuffle_seed = shuffle_seed
        self.early_stopping = WageringEarlyStopping(
            patience=early_stopping_patience,
            criterion=early_stopping_criterion,
            use_brier_d_regret=use_brier_d_regret_for_early_stopping,
            use_min_kl=use_min_kl_for_early_stopping,
        )
        self.batch_size = batch_size
        self.validation_split_ratio = validation_split_ratio
        self.wager_score_plot_every = (
            int(wager_score_plot_every) if wager_score_plot_every is not None else None
        )
        if self.wager_score_plot_every is not None and self.wager_score_plot_every <= 0:
            raise ValueError(
                f"wager_score_plot_every must be positive, got {wager_score_plot_every}"
            )
        self.logit_calibrator = logit_calibrator
        self.max_training_batches = (
            int(max_training_batches) if max_training_batches is not None else None
        )
        if self.max_training_batches is not None and self.max_training_batches <= 0:
            raise ValueError(
                f"max_training_batches must be positive, got {max_training_batches}"
            )
        self.requires_hidden_states = bool(getattr(self.wagering_method, "requires_hidden_states", True))
        self.method_requires_model_perplexities = bool(
            getattr(self.wagering_method, "requires_model_perplexities", False)
        )
        self._model_configs_for_sequential_perplexity = model_configs_for_sequential_perplexity
        self._perplexity_load_cache_kwargs = perplexity_load_cache_kwargs or {}
        self._router_prompts_per_model: Optional[List[List[str]]] = None

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._plotter = WageringPlotter(
            checkpoint_dir=self.checkpoint_dir,
            metadata=self.metadata,
            dataset=self.dataset,
            models=self.models,
            log_wandb_plot=self._log_wandb_plot,
        )

        # Training state
        self.current_step = 0
        self.wagers_history = []
        

        # Cache the most recent validation metrics (for final logging fallback)
        self.last_val_metrics: Optional[Dict[str, Any]] = None
        
        run_step = get_run_step(self.wandb_logger)
        if run_step is not None and self.current_step < run_step:
            log.info(
                "Aligning trainer current_step from %d to active wandb run step %d to keep logging monotonic",
                self.current_step,
                run_step,
            )
            self.current_step = run_step
        
        # Collect per-dataset cached logits/hidden states first, then combine datasets and shuffle
        self._collect_logits()
        if self.requires_hidden_states and (
            not hasattr(self, "all_hidden_states") or self.all_hidden_states is None
        ):
            raise RuntimeError(
                "Hidden states required but not collected with logits in _collect_logits."
            )
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
        (
            self.combined_dataset,
            self.labels,
            self.example_local_indices,
            self.validation_dataset,
            self.validation_labels,
            self.validation_example_local_indices,
            self.all_model_logits,
            self.all_model_val_logits,
            self.all_hidden_states,
            self.all_val_hidden_states,
        ) = apply_shuffling(
            self.combined_dataset,
            self.labels,
            self.example_local_indices,
            shuffle_data=self.shuffle_data,
            shuffle_seed=self.shuffle_seed,
            validation_split_ratio=self.validation_split_ratio,
            dataset=self.dataset,
            all_model_logits=getattr(self, "all_model_logits", None),
            all_hidden_states=getattr(self, "all_hidden_states", None),
        )
        self._prepare_model_perplexities()

    def _prepare_model_perplexities(self) -> None:
        """Precompute train/validation prompt perplexities when required by method."""
        self.model_prompt_perplexities = None
        self.validation_model_prompt_perplexities = None

        if not self.method_requires_model_perplexities:
            return

        ppl_kwargs = {
            "model_configs_for_sequential": self._model_configs_for_sequential_perplexity,
            "load_cache_kwargs": self._perplexity_load_cache_kwargs,
            "group_identical_model_configs": False,
        }
        self.model_prompt_perplexities = compute_all_prompt_perplexities(
            self.models, self.combined_dataset, **ppl_kwargs
        )
        if self.validation_dataset is not None:
            self.validation_model_prompt_perplexities = compute_all_prompt_perplexities(
                self.models, self.validation_dataset, **ppl_kwargs
            )

        if any(isinstance(m, WhiteboxModel) for m in self.models):
            self.models = unload_ensemble_whitebox_models(self.models)
            log.info("Unloaded language-model weights after prompt perplexity precompute.")

        log.info(
            "Computed prompt perplexities for training method: train_shape=%s%s",
            self.model_prompt_perplexities.shape,
            "" if self.validation_model_prompt_perplexities is None else f", val_shape={self.validation_model_prompt_perplexities.shape}",
        )

    def _log_wandb_plot(self, payload: Dict[str, Any]) -> None:
        self.current_step = log_plot_payload(self.wandb_logger, self.current_step, payload)

    def _prepare_datasets(self):
        """Build label arrays from the training dataset (after cache collection).

        Shuffling and train/validation split happen in :func:`apply_shuffling`.
        """
        if len(self.dataset.x) <= 0:
            raise ValueError("Cannot build training set: dataset is empty.")

        labels = []
        for y in self.dataset.y:
            if isinstance(y, str):
                labels.append(self.option_tokens.index(y))
            else:
                labels.append(int(y))

        self.combined_dataset = self.dataset
        self.labels = np.array(labels, dtype=np.int32)
        self.example_local_indices = np.arange(len(self.dataset.x), dtype=np.int32)

    def _router_wagering_question_kwargs_for_batch(
        self,
        base_questions: List[str],
        batch_start: int,
        batch_end: int,
        *,
        validation: bool,
    ) -> Dict[str, Any]:
        """
        Build ``questions`` / optional ``questions_per_model`` kwargs for compute_wagers.

        Router methods with ``expects_per_model_router_prompts`` receive per-model prompt
        variants on mixed-context datasets (PubMedQA and pubmedqa-routed CSV data).
        """
        kwargs: Dict[str, Any] = {"questions": base_questions}
        if not bool(getattr(self.wagering_method, "expects_per_model_router_prompts", False)):
            return kwargs

        if validation:
            local_indices = getattr(self, "validation_example_local_indices", None)
        else:
            local_indices = getattr(self, "example_local_indices", None)

        if local_indices is None:
            return kwargs

        batch_local_indices = np.asarray(local_indices[batch_start:batch_end], dtype=np.int32)
        if batch_local_indices.shape[0] != len(base_questions):
            return kwargs

        num_models = len(self.models)
        if num_models <= 0:
            return kwargs

        ds = self.dataset
        questions_per_model: List[List[str]] = [[] for _ in range(num_models)]
        force_without_context = bool(getattr(self.wagering_method, "pubmedqa_strip_context", False))
        for row_idx, fallback_question in enumerate(base_questions):
            local_idx = int(batch_local_indices[row_idx])
            if get_mixed_context_dataset_type(ds) is None:
                for mi in range(num_models):
                    questions_per_model[mi].append(fallback_question)
                continue

            if force_without_context:
                without_ctx = getattr(ds, "pubmedqa_without_context_x", None)
                if isinstance(without_ctx, list) and 0 <= local_idx < len(without_ctx):
                    prompt = without_ctx[local_idx]
                else:
                    prompt = fallback_question
                for mi in range(num_models):
                    questions_per_model[mi].append(prompt)
                continue

            if self._router_prompts_per_model is None:
                self._router_prompts_per_model = [
                    get_model_specific_prompts(ds, model_index=mi) for mi in range(num_models)
                ]

            per_model_lists = self._router_prompts_per_model
            if local_idx < 0 or local_idx >= len(per_model_lists[0]):
                for mi in range(num_models):
                    questions_per_model[mi].append(fallback_question)
                continue

            for mi in range(num_models):
                questions_per_model[mi].append(per_model_lists[mi][local_idx])

        kwargs["questions_per_model"] = questions_per_model
        return kwargs

    def _evaluate_validation(self) -> Tuple[Dict[str, Any], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """
        Evaluate the wagering method on the validation set using batch processing.
        
        Returns:
            Tuple of:
                - metrics dictionary (accuracy, nll, ece, auc, ...)
                - val_score_diffs per validation sample and model (or None)
                - val_wagers per validation sample and model
                - val_sigmoid_wagers per validation sample and model (or None)
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
        
        val_predictions = []
        val_probs = []
        val_wagers = []  # Track wagers for each example
        num_val_examples = len(self.validation_dataset.x)
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
            }
            wagering_kwargs.update(
                self._router_wagering_question_kwargs_for_batch(
                    batch_questions,
                    batch_start,
                    batch_end,
                    validation=True,
                )
            )
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
            batch_score_diff = res_dict.get("score_diff", None)
            # Aggregate predictions for batch
            batch_aggregated_log_probs, batch_aggregated_probs = self.aggregation_function.aggregate(
                batch_logits_transposed, batch_wagers
            )  # [batch_size, num_options] each
            
            batch_predictions = np.argmax(batch_aggregated_probs, axis=1)  # [batch_size]
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
        if val_score_diffs is not None:
            val_score_diffs = np.asarray(val_score_diffs, dtype=np.float32)
        # Compute metrics
        val_accuracy = np.mean(val_predictions == self.validation_labels)
        
        # Compute NLL
        correct_class_probs = val_probs[np.arange(len(self.validation_labels)), self.validation_labels]
        val_nll = -np.mean(np.log(correct_class_probs + 1e-10))
        
        ece_metric = ECE(n_bins=20)
        confidences = val_probs.max(axis=1)
        correctness = (val_predictions == self.validation_labels).astype(float)
        val_ece = ece_metric(confidences.tolist(), correctness.tolist())

        max_probs = val_probs.max(axis=1)
        correctness_int = (val_predictions == self.validation_labels).astype(int)
        val_auc = roc_auc_score(correctness_int, max_probs)

        val_model_logits = np.transpose(self.all_model_val_logits, (1, 0, 2))

        val_gold_dist = build_gold_label_distribution_for_rows(
            self.validation_labels,
            self.validation_example_local_indices,
            self.dataset,
            self.option_tokens,
            int(val_probs.shape[1]),
        )
        dataset_name = getattr(self.dataset, "cache_dataset_name", None)
        has_soft_labels = (
            is_cluster_saturation_dataset_name(dataset_name)
            and hasattr(self.dataset, "probabilistic_labels")
        )
        val_kl_to_gold = None
        if has_soft_labels:
            val_kl_to_gold = compute_mean_kl_to_gold_distribution(
                val_gold_dist,
                val_probs,
            )
        val_brier_d_regret = compute_brier_dynamic_regret(
            val_model_logits,
            val_probs,
            self.validation_labels,
            gold_label_distribution=val_gold_dist,
        )
        val_model_brier_scores = compute_model_brier_scores(
            val_model_logits,
            self.validation_labels,
        )
        meta_metrics = compute_meta_metrics(
            val_wagers,
            val_model_brier_scores,
        )
        val_kendall_tau = meta_metrics["kendall_tau"]
        val_best_model_mrr = meta_metrics["best_model_mrr"]
        
        # Set back to train mode
        self.wagering_method.train_mode()
        
        metrics = {
            "accuracy": val_accuracy,
            "nll": val_nll,
            "ece": val_ece if val_ece is not None and not np.isnan(val_ece) else None,
            "auc": val_auc if val_auc is not None and not np.isnan(val_auc) else None,
            "brier_d_regret": val_brier_d_regret if val_brier_d_regret is not None and not np.isnan(val_brier_d_regret) else None,
            "kl_to_gold": val_kl_to_gold if val_kl_to_gold is not None and not np.isnan(val_kl_to_gold) else None,
            "kendall_tau": val_kendall_tau if val_kendall_tau is not None and not np.isnan(val_kendall_tau) else None,
            "best_model_mrr": val_best_model_mrr if val_best_model_mrr is not None and not np.isnan(val_best_model_mrr) else None,
        }
        
        self._plotter.plot_validation_wagers_by_dataset(val_wagers, {})
        
        return metrics, val_score_diffs, val_wagers, val_sigmoid_wagers

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
                }
                wagering_kwargs.update(
                    self._router_wagering_question_kwargs_for_batch(
                        batch_questions,
                        batch_start,
                        batch_end,
                        validation=True,
                    )
                )
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
                    np.asarray(compute_model_brier_scores(batch_logits_transposed, batch_labels), dtype=np.float32)
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
        context_assignment_mask, context_assignment_kind = get_validation_context_assignment_mask(
            self.dataset,
            num_examples=num_plot_examples,
            num_models_total=int(result["wagers"].shape[1]),
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

        Note: Validation split happens AFTER cache loading in apply_shuffling(), so this
        only collects logits and hidden states for the full unshuffled datasets.

        TODO: Methods that update LLMs during training should disable caching.
        """
        collect_wagering_hidden_states = self.requires_hidden_states
        collect_calibration_hidden_states = self.logit_calibrator is not None
        collect_any_hidden_states = collect_wagering_hidden_states or collect_calibration_hidden_states

        if collect_any_hidden_states:
            log.info(
                "Collecting logits and hidden states from all models "
                "(per-model, per-dataset cache, unshuffled)..."
            )
        else:
            log.info(
                "Collecting logits from all models "
                "(hidden states disabled for this wagering method)..."
            )

        artifacts = collect_stacked_model_artifacts(
            self.models,
            self.dataset,
            self.option_tokens,
            collect_hidden_states=collect_any_hidden_states,
            mixed_context_error_message=(
                "Mixed-context dataset missing per-example context assignments. "
                "Ensure assign_pubmedqa_context_models ran before cache collection."
            ),
        )
        self.all_model_logits = artifacts.logits
        log.debug(f"All training logits shape: {self.all_model_logits.shape}")

        if artifacts.context_assignments is not None:
            context_assignments = artifacts.context_assignments
        else:
            context_assignments = np.full((len(self.dataset.x),), -1, dtype=np.int64)

        if (
            context_assignments.shape[0] == self.all_model_logits.shape[1]
            and np.any(context_assignments >= 0)
        ):
            self.all_calibration_context_assignments = context_assignments
        else:
            self.all_calibration_context_assignments = None

        combined_hidden_states = (
            artifacts.combined_hidden_states() if collect_any_hidden_states else None
        )
        if collect_wagering_hidden_states:
            self.all_hidden_states = combined_hidden_states
        else:
            self.all_hidden_states = None
        if collect_calibration_hidden_states:
            self.all_calibration_hidden_states = combined_hidden_states

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
        apply_kwargs = {}
        if context_assignments is not None:
            apply_kwargs["context_model_index_by_example"] = context_assignments
        self.all_model_logits = self.logit_calibrator.apply_to_stacked_logits(
            self.all_model_logits,
            calibration_hidden_states,
            **apply_kwargs,
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

        num_epochs = int(num_epochs)

        # Inference-only methods with no trainable parameters produce identical
        # per-batch outputs on repeated epochs over the same frozen cached logits.
        # Run one epoch and reuse those metrics for remaining epochs.
        if num_epochs > 1:
            has_trainable_params = bool(self.wagering_method.get_trainable_parameters())

            if not has_trainable_params:
                num_epochs = 1
        
        num_examples = len(self.combined_dataset.x)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size

        
        # Track epoch-level metrics for early stopping
        epoch_accuracies = []

        val_d_regret_history = []
        val_accuracy_history = []

        # Initialize these lists (will be reset each epoch to only keep final epoch's predictions)
        all_predictions = []
        all_aggregated_probs = []
        wagers_history = []
        stop_training_now = False

        last_completed_batches = 0
        validation_dataset_size = (
            len(self.validation_dataset.x) if self.validation_dataset is not None else None
        )
        self.early_stopping.setup_online_window(
            num_training_examples=num_examples,
            batch_size=self.batch_size,
            validation_split_ratio=self.validation_split_ratio,
            validation_dataset_size=validation_dataset_size,
        )
        self.early_stopping.log_enabled()

        batches_processed = 0
        for epoch in range(num_epochs):
            
            # Reset predictions/probs/wagers at start of each epoch
            # We only want to keep the final epoch's predictions for evaluation
            all_predictions = []
            all_aggregated_probs = []
            wagers_history = []
            
            epoch_predictions = []
            epoch_probs = []
            epoch_correct = 0
            epoch_nll_sum = 0.0
            
            for batch_idx in range(num_batches):
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
                batch_local_indices = np.asarray(
                    self.example_local_indices[batch_start:batch_end], dtype=np.int32
                )
                num_options = int(batch_logits_transposed.shape[2])
                batch_gold_label_distribution = build_gold_label_distribution_for_rows(
                    batch_labels,
                    batch_local_indices,
                    self.dataset,
                    self.option_tokens,
                    num_options,
                ).astype(np.float32)
                wagering_kwargs = {
                    "model_logits": batch_logits_transposed,
                    "gold_label": batch_labels,
                    "hidden_states_list": batch_hidden_states,
                }
                wagering_kwargs.update(
                    self._router_wagering_question_kwargs_for_batch(
                        batch_questions,
                        batch_start,
                        batch_end,
                        validation=False,
                    )
                )
                wagering_kwargs["gold_label_distribution"] = batch_gold_label_distribution
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

                self.wagering_method.update(
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
                batch_brier_d_regret = None
                batch_kl_to_gold = None
                batch_soft_label_count = 0
                
                batch_brier_d_regret = compute_brier_dynamic_regret(
                    batch_logits_transposed,
                    batch_aggregated_probs,
                    batch_labels,
                    gold_label_distribution=np.asarray(
                        batch_gold_label_distribution, dtype=np.float64
                    ),
                )

                dataset_name = getattr(self.dataset, "cache_dataset_name", None)
                has_soft_labels = (
                    is_cluster_saturation_dataset_name(dataset_name)
                    and hasattr(self.dataset, "probabilistic_labels")
                )
                if has_soft_labels:
                    batch_soft_label_count = int(batch_size_actual)
                    batch_kl_to_gold = compute_mean_kl_to_gold_distribution(
                        np.asarray(batch_gold_label_distribution, dtype=np.float64),
                        np.asarray(batch_aggregated_probs, dtype=np.float64),
                    )

                if self.early_stopping.should_track_online_batch():
                    if self.early_stopping.update_online_batch(
                        epoch=epoch,
                        batch_end=batch_end,
                        num_examples=num_examples,
                        batch_size_actual=batch_size_actual,
                        batch_brier_d_regret=batch_brier_d_regret,
                        batch_kl_to_gold=batch_kl_to_gold,
                        batch_soft_label_count=batch_soft_label_count,
                        checkpoint_state=self.wagering_method.state_dict(),
                    ):
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

                self.current_step = global_step

                batches_processed += 1
                if self.max_training_batches is not None and batches_processed >= self.max_training_batches:
                    log.info(
                        "Stopping after %d training batch(es) (max_training_batches).",
                        self.max_training_batches,
                    )
                    stop_training_now = True
                if stop_training_now:
                    break
            

            # Compute epoch-level metrics
            epoch_labels = self.labels[: len(epoch_predictions)]
            epoch_accuracy = np.mean(np.array(epoch_predictions) == epoch_labels)
            epoch_nll = epoch_nll_sum / len(epoch_predictions)
            
            # Increment current_step to ensure epoch-level logging uses a step after batch logs
            # This prevents wandb warnings about logging to an already-used step
            self.current_step += 1
            
            epoch_accuracies.append(epoch_accuracy)

            if stop_training_now:
                break
            
            # Evaluate on validation set if available
            val_metrics = {}
            if self.validation_dataset is not None:
                val_metrics, val_score_diff, val_wagers, val_sigmoid_wagers = self._evaluate_validation()
                val_d_regret = val_metrics.get("d_regret", None)
                val_brier_d_regret = val_metrics.get("brier_d_regret", None)
                val_kl_to_gold = val_metrics.get("kl_to_gold", None)
                if val_metrics:
                    self.last_val_metrics = val_metrics
                self.early_stopping.require_finite_validation_kl(val_kl_to_gold)
            else:
                val_d_regret = None
                val_brier_d_regret = None
                val_kl_to_gold = None

            if self.early_stopping.should_track_validation_epoch(
                val_d_regret=val_d_regret,
                val_brier_d_regret=val_brier_d_regret,
                val_kl_to_gold=val_kl_to_gold,
            ):
                if self.early_stopping.update_validation_epoch(
                    epoch=epoch,
                    val_d_regret=val_d_regret,
                    val_brier_d_regret=val_brier_d_regret,
                    val_kl_to_gold=val_kl_to_gold,
                    wagering_method=self.wagering_method,
                    checkpoint_state=self.wagering_method.state_dict(),
                ):
                    break
            
            log_train_epoch(
                self.wandb_logger,
                self.current_step,
                epoch=epoch,
                epoch_accuracy=epoch_accuracy,
                epoch_nll=epoch_nll,
                epoch_predictions=epoch_predictions,
                epoch_probs=epoch_probs,
                epoch_labels=epoch_labels,
                all_model_logits=self.all_model_logits,
                wagers_history=wagers_history,
                val_metrics=val_metrics,
                validation_dataset_configured=self.validation_dataset is not None,
            )

        self.early_stopping.restore_best_checkpoint(self.wagering_method)

        if self.wager_score_plot_every is not None and self.validation_dataset is not None:
            plot_arrays = self._collect_validation_plot_arrays(max_examples=1000)
            if plot_arrays is not None:
                final_batch_step = max(1, last_completed_batches)
                final_epoch = (
                    self.early_stopping.best_epoch
                    if self.early_stopping.best_epoch is not None
                    else max(0, epoch)
                )
                log.info(
                    "Logging final wager plots using best available checkpoint state (epoch=%d, step=%d).",
                    final_epoch + 1,
                    final_batch_step,
                )
                self._plotter.plot_val_wagers_vs_score_diff_for_epoch(
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
                    self._plotter.plot_val_estimated_score_diff_vs_wagers_for_epoch(
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
                    self._plotter.plot_val_own_score_vs_estimated_score_for_epoch(
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
                    self._plotter.plot_val_average_score_vs_estimated_average_score_for_epoch(
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
                raise RuntimeError(
                    "wager_score_plot_every is set but validation plot arrays could not be collected"
                )

        # Convert to arrays
        all_predictions = np.array(all_predictions, dtype=np.int32)
        all_aggregated_probs = np.stack(all_aggregated_probs, axis=0)
        wagers_history = np.stack(wagers_history, axis=0)  # [num_examples, num_models]
        
        num_processed = len(all_predictions)
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

        processed_example_local_indices = self.example_local_indices[:num_processed]
        
        # Compute final metrics
        accuracy = np.mean(all_predictions == processed_labels)
        
        # Compute NLL (negative log likelihood) for correct classes
        correct_class_probs = all_aggregated_probs[np.arange(len(processed_labels)), processed_labels]
        nll = -np.mean(np.log(correct_class_probs + 1e-10))
        
        ece_metric = ECE(n_bins=20)
        confidences = all_aggregated_probs.max(axis=1)
        correctness = (all_predictions == processed_labels).astype(float)
        ece = ece_metric(confidences.tolist(), correctness.tolist())

        max_probs = all_aggregated_probs.max(axis=1)
        correctness_int = (all_predictions == processed_labels).astype(int)
        auc = roc_auc_score(correctness_int, max_probs)

        final_model_logits_transposed = self.all_model_logits[:, :num_processed, :]
        final_model_logits = np.transpose(final_model_logits_transposed, (1, 0, 2))

        final_gold_dist = build_gold_label_distribution_for_rows(
            processed_labels,
            processed_example_local_indices,
            self.dataset,
            self.option_tokens,
            int(all_aggregated_probs.shape[1]),
        )
        brier_d_regret = compute_brier_dynamic_regret(
            final_model_logits,
            all_aggregated_probs,
            processed_labels,
            gold_label_distribution=final_gold_dist,
        )

        
        results = {
            "predictions": all_predictions,
            "aggregated_probs": all_aggregated_probs,
            "labels": processed_labels,
            "wagers_history": wagers_history,
            "val_d_regret_history": np.array(val_d_regret_history, dtype=np.float32),
            "val_accuracy_history": np.array(val_accuracy_history, dtype=np.float32),
            "final_accuracy": accuracy,
            "final_nll": nll,
            "final_ece": ece,
            "final_auc": auc,
            "final_brier_d_regret": brier_d_regret,
        }
        
        # Log metrics
        log.info("\n=== Training Metrics ===")
        # Create analytics dataframe
        dataset_size = len(self.combined_dataset.x) if hasattr(self, 'combined_dataset') and self.combined_dataset is not None else None
        analytics_df = WageringAnalytics.create_training_analytics(
            wagering_method=self.wagering_method,
            aggregation_function=self.aggregation_function,
            models=self.models,
            dataset=self.dataset,
            shuffle_data=self.shuffle_data,
            shuffle_seed=self.shuffle_seed,
            early_stopping_patience=self.early_stopping.patience,
            early_stopping_criterion=self.early_stopping.criterion,
            use_brier_d_regret_for_early_stopping=self.early_stopping.use_brier_d_regret_for_early_stopping,
            use_min_kl_for_early_stopping=self.early_stopping.use_min_kl_for_early_stopping,
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

        self.current_step = log_train_final(
            self.wandb_logger,
            self.current_step,
            accuracy=accuracy,
            nll=nll,
            ece=ece,
            auc=auc,
            brier_d_regret=brier_d_regret,
        )
        self.current_step = log_val_final(
            self.wandb_logger,
            self.current_step,
            validation_dataset=self.validation_dataset,
            get_val_metrics=self._evaluate_validation,
        )
        
        # Plot wagers over time
        self._plotter.plot_wagers_over_time(wagers_history, results)
        
        return results
    
    def save_final_checkpoint(self, save_dir: str) -> str:
        """Save final checkpoint and return the path.
        
        Returns:
            str: Path to the saved checkpoint directory
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure we save the best epoch state if available
        if self.early_stopping.best_wagering_method_state is not None:
            self.wagering_method.load_state_dict(self.early_stopping.best_wagering_method_state)

        # Save wagering method (contains best epoch state if early stopping occurred)
        self.wagering_method.save_pretrained(str(save_dir))
        return str(save_dir)

