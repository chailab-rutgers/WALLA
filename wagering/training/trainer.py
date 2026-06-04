"""
Training pipeline for multi-LLM wagering methods.
"""

import logging
import os
import sys
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
    get_model_specific_prompts,
    get_model_prompt_variant,
    get_cached_logits_and_hidden_states_for_model,
    resolve_hidden_state_layers_for_model,
    set_cached_logits_and_hidden_states_for_model,
    _get_mixed_context_dataset_type,
)

log = logging.getLogger("wagering")

import re

from sklearn.metrics import roc_auc_score

from wagering.core.metrics import ECE
from wagering.utils.wagering_plots import WageringPlotter, get_validation_context_assignment_mask
from wagering.utils.wagering_metrics import (
    build_gold_label_distribution_for_rows,
    compute_brier_dynamic_regret,
    compute_dynamic_regret,
    compute_mean_kl_to_gold_distribution,
    compute_meta_metrics,
    compute_model_bernoulli_kl_to_gt_scores,
    compute_model_brier_scores,
    is_cluster_saturation_dataset_name,
    resolve_positive_option_index,
)


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
        metadata: Optional[Dict[str, Any]] = None,
        shuffle_data: bool = True,
        shuffle_seed: int = 42,
        early_stopping_patience: int = 10,
        batch_size: int = 100,  # Batch size for training loop
        validation_split_ratio: float = 0.1,  # Fraction of data to use for validation (default: 10%)
        balance_training_datasets: bool = True,
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
            datasets: List of Dataset instances (will be concatenated for training)
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
            balance_training_datasets: If True, randomly subsample each training
                dataset to the minimum dataset size before concatenation.
            max_training_batches: If set, stop after this many training-loop batches
                (optimizer steps) across epochs in this ``train()`` call.
            model_configs_for_sequential_perplexity: Merged per-model YAML dicts; used
                to load one model at a time for prompt perplexity when visible GPUs
                are fewer than ensemble slots.
            perplexity_load_cache_kwargs: Optional kwargs (e.g. ``cache_dir``) for
                those sequential loads.
        """
        self.models = models
        self.datasets = datasets
        self.wagering_method = wagering_method
        self.aggregation_function = aggregation_function
        self.option_tokens = option_tokens
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.wandb_logger = wandb_logger
        self.metadata = metadata or {}
        self.shuffle_data = shuffle_data
        self.shuffle_seed = shuffle_seed
        self.early_stopping_patience = early_stopping_patience
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
        self.hidden_state_layers = getattr(self.wagering_method, "hidden_state_layers", None)
        self.hidden_state_layers_per_model = getattr(self.wagering_method, "hidden_state_layers_per_model", None)
        self.method_requires_model_perplexities = bool(
            getattr(self.wagering_method, "requires_model_perplexities", False)
        )
        self._model_configs_for_sequential_perplexity = model_configs_for_sequential_perplexity
        self._perplexity_load_cache_kwargs = perplexity_load_cache_kwargs or {}
        self._router_prompts_per_model_by_dataset: Dict[int, List[List[str]]] = {}

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._plotter = WageringPlotter(
            checkpoint_dir=self.checkpoint_dir,
            metadata=self.metadata,
            datasets=self.datasets,
            models=self.models,
            log_wandb_plot=self._log_wandb_plot,
        )

        # Training state
        self.current_step = 0
        self.wagers_history = []
        self.metrics_history = []
        self.batch_metrics_history: List[Dict[str, Any]] = []
        
        # Running average tracker for last 5 batches (for wandb logging)
        self.running_avg_window = 10
        self.batch_metrics_buffer = deque(maxlen=self.running_avg_window)  # Store last N batches of metrics

        # Cache the most recent validation metrics (for final logging fallback)
        self.last_val_metrics: Optional[Dict[str, Any]] = None
        
        # Early stopping state
        # Now epoch-based instead of step-based
        # Note: best_d_regret tracks validation d_regret if validation set exists, otherwise training d_regret
        # d_regret is a loss metric, so lower is better (initialized to infinity)
        self.best_d_regret = float('inf')
        self.best_brier_d_regret = float('inf')
        self.best_kl_to_gold = float("inf")
        self.best_batch_brier_d_regret = float('inf')
        self.best_batch_kl_to_gold = float("inf")
        self.epochs_since_improvement = 0
        self.batches_since_improvement = 0
        self.early_stopped = False
        self.best_wagering_method_state = None  # Store the best checkpoint state
        self.best_epoch = None  # Track which epoch had the best checkpoint
        self.best_batch_step = None  # Track global step for online-learning best checkpoint

        # If logging into an already-active wandb run, ensure training steps never
        # move backward relative to run.step.
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

            all_perplexities[:, model_index] = self._compute_prompt_perplexities_for_model(
                model=model,
                prompts=model_prompts,
                batch_size=max(1, int(dataset.batch_size)),
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
                return int(run.step)

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
        variants on mixed-context datasets (PubMedQA/RACE and pubmedqa-routed CSV data).
        """
        kwargs: Dict[str, Any] = {"questions": base_questions}
        if not bool(getattr(self.wagering_method, "expects_per_model_router_prompts", False)):
            return kwargs

        if validation:
            dataset_indices = getattr(self, "validation_dataset_indices", None)
            local_indices = getattr(self, "validation_example_local_indices", None)
        else:
            dataset_indices = getattr(self, "dataset_indices", None)
            local_indices = getattr(self, "example_local_indices", None)

        if dataset_indices is None or local_indices is None:
            return kwargs

        batch_dataset_indices = np.asarray(dataset_indices[batch_start:batch_end], dtype=np.int32)
        batch_local_indices = np.asarray(local_indices[batch_start:batch_end], dtype=np.int32)
        if (
            batch_dataset_indices.shape[0] != len(base_questions)
            or batch_local_indices.shape[0] != len(base_questions)
        ):
            return kwargs

        num_models = len(self.models)
        if num_models <= 0:
            return kwargs

        questions_per_model: List[List[str]] = [[] for _ in range(num_models)]
        force_without_context = bool(getattr(self.wagering_method, "pubmedqa_strip_context", False))
        for row_idx, fallback_question in enumerate(base_questions):
            dataset_idx = int(batch_dataset_indices[row_idx])
            local_idx = int(batch_local_indices[row_idx])
            if dataset_idx < 0 or dataset_idx >= len(self.datasets):
                for mi in range(num_models):
                    questions_per_model[mi].append(fallback_question)
                continue

            ds = self.datasets[dataset_idx]
            if _get_mixed_context_dataset_type(ds) is None:
                for mi in range(num_models):
                    questions_per_model[mi].append(fallback_question)
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
                    questions_per_model[mi].append(prompt)
                continue

            if dataset_idx not in self._router_prompts_per_model_by_dataset:
                self._router_prompts_per_model_by_dataset[dataset_idx] = [
                    get_model_specific_prompts(ds, model_index=mi) for mi in range(num_models)
                ]

            per_model_lists = self._router_prompts_per_model_by_dataset[dataset_idx]
            if local_idx < 0 or local_idx >= len(per_model_lists[0]):
                for mi in range(num_models):
                    questions_per_model[mi].append(fallback_question)
                continue

            for mi in range(num_models):
                questions_per_model[mi].append(per_model_lists[mi][local_idx])

        kwargs["questions_per_model"] = questions_per_model
        return kwargs

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
        val_d_regret, best_expert_ids = compute_dynamic_regret(
            val_model_logits, val_probs, self.validation_labels
        )
        val_gold_dist = build_gold_label_distribution_for_rows(
            self.validation_labels,
            self.validation_dataset_indices,
            self.validation_example_local_indices,
            self.datasets,
            self.option_tokens,
            int(val_probs.shape[1]),
        )
        ds_ix = np.asarray(self.validation_dataset_indices, dtype=np.int32)
        soft_mask = np.zeros((int(ds_ix.shape[0]),), dtype=bool)
        for dataset_idx in np.unique(ds_ix).tolist():
            ds_idx = int(dataset_idx)
            if ds_idx < 0 or ds_idx >= len(self.datasets):
                continue
            ds = self.datasets[ds_idx]
            dataset_name = getattr(ds, "cache_dataset_name", None)
            if not is_cluster_saturation_dataset_name(dataset_name):
                continue
            if not hasattr(ds, "probabilistic_labels"):
                continue
            soft_mask |= ds_ix == ds_idx
        val_kl_to_gold = None
        if np.any(soft_mask):
            val_kl_to_gold = compute_mean_kl_to_gold_distribution(
                val_gold_dist,
                val_probs,
                mask=soft_mask,
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
            best_expert_ids,
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
            "d_regret": val_d_regret if val_d_regret is not None and not np.isnan(val_d_regret) else None,
            "brier_d_regret": val_brier_d_regret if val_brier_d_regret is not None and not np.isnan(val_brier_d_regret) else None,
            "kl_to_gold": val_kl_to_gold if val_kl_to_gold is not None and not np.isnan(val_kl_to_gold) else None,
            "kendall_tau": val_kendall_tau if val_kendall_tau is not None and not np.isnan(val_kendall_tau) else None,
            "best_model_mrr": val_best_model_mrr if val_best_model_mrr is not None and not np.isnan(val_best_model_mrr) else None,
        }
        
        # Compute metrics by dataset
        if hasattr(self, 'validation_dataset_indices') and self.validation_dataset_indices is not None:
            # Plot validation wagers by dataset
            val_results = {
                "dataset_indices": self.validation_dataset_indices,
            }
            self._plotter.plot_validation_wagers_by_dataset(val_wagers, val_results)
        
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
            self.datasets,
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
            has_trainable_params = bool(self.wagering_method.get_trainable_parameters())

            if not has_trainable_params:
                effective_num_epochs = 1
                reuse_static_epoch_results = self.max_training_batches is None
        
        num_examples = len(self.combined_dataset.x)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size
        
        # Training loop
        batch_metrics = []
        
        # Track epoch-level metrics for early stopping
        epoch_accuracies = []

        val_d_regret_history = []
        val_accuracy_history = []

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

        if self.early_stopping_patience > 0:
            if self.early_stopping_criterion == "online_learning":
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
                        else "validation d_regret"
                    )
                log.info(
                    "Early stopping enabled: criterion=validation, metric=%s, "
                    "patience=%d epochs",
                    metric_name,
                    self.early_stopping_patience,
                )
        
        batches_processed = 0
        for epoch in range(effective_num_epochs):
            
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
                batch_dataset_indices = np.asarray(
                    self.dataset_indices[batch_start:batch_end], dtype=np.int32
                )
                batch_local_indices = np.asarray(
                    self.example_local_indices[batch_start:batch_end], dtype=np.int32
                )
                num_options = int(batch_logits_transposed.shape[2])
                batch_gold_label_distribution = build_gold_label_distribution_for_rows(
                    batch_labels,
                    batch_dataset_indices,
                    batch_local_indices,
                    self.datasets,
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
                batch_sigmoid_wagers = res_dict.get("sigmoid_wagers", None)
                batch_total_payout_values = res_dict.get("total_payout", None)

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
                    sw = np.asarray(batch_sigmoid_wagers, dtype=np.float64)
                    if sw.ndim != 2:
                        raise ValueError(
                            f"sigmoid_wagers must be shape [batch, num_models], got {sw.shape}"
                        )
                    batch_total_wagers = float(sw.sum(axis=1).mean())

                if batch_total_payout_values is not None:
                    tp = np.asarray(batch_total_payout_values, dtype=np.float64)
                    if tp.ndim != 2:
                        raise ValueError(
                            f"total_payout must be shape [batch, num_models], got {tp.shape}"
                        )
                    batch_total_payout = float(tp.sum(axis=1).mean())

                batch_d_regret, batch_best_expert_ids = compute_dynamic_regret(
                    batch_logits_transposed,
                    batch_aggregated_probs,
                    batch_labels,
                )
                batch_brier_d_regret = compute_brier_dynamic_regret(
                    batch_logits_transposed,
                    batch_aggregated_probs,
                    batch_labels,
                    gold_label_distribution=np.asarray(
                        batch_gold_label_distribution, dtype=np.float64
                    ),
                )
                batch_model_brier_scores = compute_model_brier_scores(
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

                soft_mask = np.zeros((int(batch_size_actual),), dtype=bool)
                for dataset_idx in np.unique(batch_dataset_indices).tolist():
                    ds_idx = int(dataset_idx)
                    if ds_idx < 0 or ds_idx >= len(self.datasets):
                        continue
                    ds = self.datasets[ds_idx]
                    dataset_name = getattr(ds, "cache_dataset_name", None)
                    if not is_cluster_saturation_dataset_name(dataset_name):
                        continue
                    if not hasattr(ds, "probabilistic_labels"):
                        continue
                    soft_mask |= batch_dataset_indices == ds_idx
                batch_soft_label_count = int(np.sum(soft_mask))
                if batch_soft_label_count > 0:
                    batch_kl_to_gold = compute_mean_kl_to_gold_distribution(
                        np.asarray(batch_gold_label_distribution, dtype=np.float64),
                        np.asarray(batch_aggregated_probs, dtype=np.float64),
                        mask=soft_mask,
                    )

                if (
                    self.early_stopping_criterion == "online_learning"
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
                        raise RuntimeError("Not implemented")

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
                ece_metric = ECE(n_bins=20)
                batch_confidences = batch_aggregated_probs.max(axis=1)
                batch_correctness = batch_correct.astype(float)
                batch_ece = ece_metric(batch_confidences.tolist(), batch_correctness.tolist())

                batch_max_probs = batch_aggregated_probs.max(axis=1)
                batch_binary_correct = batch_correct.astype(int)
                batch_auc = roc_auc_score(batch_binary_correct, batch_max_probs)

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
                }

                # Add wagering-specific batch summaries for offline analysis/plotting.
                # Keep these mirrored with wandb keys where possible.
                if batch_total_wagers is not None and np.isfinite(batch_total_wagers):
                    batch_record["total_wagers"] = float(batch_total_wagers)
                if batch_total_payout is not None and np.isfinite(batch_total_payout):
                    batch_record["total_payout"] = float(batch_total_payout)
                if batch_wagers is not None and batch_wagers.shape[1] > 0:
                    for i in range(batch_wagers.shape[1]):
                        batch_record[f"wager_model_{i}"] = float(np.mean(batch_wagers[:, i]))
                if batch_sigmoid_wagers is not None:
                    sw = np.asarray(batch_sigmoid_wagers, dtype=np.float64)
                    for i in range(sw.shape[1]):
                        batch_record[f"sigmoid_wager_model_{i}"] = float(sw[:, i].mean())
                if batch_total_payout_values is not None:
                    payout_arr = np.asarray(batch_total_payout_values, dtype=np.float64)
                    for i in range(payout_arr.shape[1]):
                        batch_record[f"net_payout_model_{i}"] = float(payout_arr[:, i].mean())
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
            # log.info(f"Epoch {epoch+1} training accuracy: {epoch_accuracy:.4f}, NLL: {epoch_nll:.4f}")

            if stop_training_now:
                break
            
            # Evaluate on validation set if available
            val_metrics = {}
            val_score_diff = None
            val_wagers = None
            val_sigmoid_wagers = None
            if self.validation_dataset is not None:
                val_metrics, val_score_diff, val_wagers, val_sigmoid_wagers = self._evaluate_validation()
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

            # Early stopping: check for improvement on validation set after each epoch
            # d_regret is a loss metric, so lower is better
            if (
                self.early_stopping_criterion == "validation"
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
                
                ece_metric = ECE(n_bins=20)
                confidences = epoch_probs_array.max(axis=1)
                correctness = (np.array(epoch_predictions) == epoch_labels).astype(float)
                epoch_ece = ece_metric(confidences.tolist(), correctness.tolist())

                max_probs = epoch_probs_array.max(axis=1)
                correctness_int = (np.array(epoch_predictions) == epoch_labels).astype(int)
                epoch_auc = roc_auc_score(correctness_int, max_probs)

                epoch_model_logits_transposed = self.all_model_logits[:, : len(epoch_predictions), :]
                epoch_model_logits = np.transpose(epoch_model_logits_transposed, (1, 0, 2))
                epoch_wagers_array = np.array(wagers_history)

                epoch_d_regret, best_expert_ids = compute_dynamic_regret(
                    epoch_model_logits, epoch_probs_array, epoch_labels
                )
                epoch_model_brier_scores = compute_model_brier_scores(
                    epoch_model_logits,
                    epoch_labels,
                )
                meta_metrics = compute_meta_metrics(
                    epoch_wagers_array,
                    best_expert_ids,
                    epoch_model_brier_scores,
                )
                epoch_kendall_tau = meta_metrics["kendall_tau"]
                epoch_best_model_mrr = meta_metrics["best_model_mrr"]

                # Log epoch metrics
                wandb_epoch_dict = {
                    "train/epoch/accuracy": epoch_accuracy,
                    "train/epoch/nll": epoch_nll,
                    "train/epoch/ece": epoch_ece if epoch_ece is not None and not np.isnan(epoch_ece) else None,
                    "train/epoch/auc": epoch_auc if epoch_auc is not None and not np.isnan(epoch_auc) else None,
                    "train/epoch/d_regret": epoch_d_regret if epoch_d_regret is not None and not np.isnan(epoch_d_regret) else None,
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
                    if val_metrics.get("kendall_tau") is not None and not np.isnan(val_metrics.get("kendall_tau", np.nan)):
                        val_dict_update["val/epoch/kendall_tau"] = val_metrics.get("kendall_tau")
                    if val_metrics.get("best_model_mrr") is not None and not np.isnan(val_metrics.get("best_model_mrr", np.nan)):
                        val_dict_update["val/epoch/best_model_mrr"] = val_metrics.get("best_model_mrr")
                    
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
                    
                elif self.validation_dataset is not None:
                    raise RuntimeError(
                        f"Validation dataset is configured but val_metrics is empty for epoch {epoch + 1}"
                    )

                if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                    self.wandb_logger.run.log(wandb_epoch_dict, step=self.current_step)
                elif hasattr(self.wandb_logger, 'log'):
                    self.wandb_logger.log(wandb_epoch_dict, step=self.current_step)
                else:
                    raise RuntimeError(
                        f"wandb_logger doesn't have 'log' method. Type: {type(self.wandb_logger)}"
                    )

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

        processed_dataset_indices = self.dataset_indices[:num_processed]
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

        
        results = {
            "predictions": all_predictions,
            "aggregated_probs": all_aggregated_probs,
            "labels": processed_labels,  # Use processed labels, not all labels
            "dataset_indices": processed_dataset_indices,
            "wagers_history": wagers_history,
            "val_d_regret_history": np.array(val_d_regret_history, dtype=np.float32),
            "val_accuracy_history": np.array(val_accuracy_history, dtype=np.float32),
            "batch_metrics": batch_metrics,
            "final_accuracy": accuracy,
            "final_nll": nll,
            "final_ece": ece,
            "final_auc": auc,
            "final_brier_d_regret": brier_d_regret,
        }
        
        # Log metrics
        log.info("\n=== Training Metrics by Dataset ===")
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
        
        # Log final training metrics to wandb
        if self.wandb_logger:
            proposed_final_step = self.current_step + 1 if hasattr(self, 'current_step') else num_epochs * num_examples
            wandb_run_step = None
            wandb_run_step = self._get_wandb_run_step()
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
                "train/final/brier_d_regret": brier_d_regret if brier_d_regret is not None and not np.isnan(brier_d_regret) else None,
            }

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
            if self.validation_dataset is not None:
                final_val_metrics, _, _, _ = self._evaluate_validation()

            if not final_val_metrics:
                if self.validation_dataset is None:
                    log.info("No validation dataset configured; skipping val/final logging.")
                else:
                    raise RuntimeError(
                        "Validation dataset is configured but final validation metrics are missing"
                    )
            else:
                proposed_final_step = self.current_step + 1
                wandb_run_step = self._get_wandb_run_step()

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

