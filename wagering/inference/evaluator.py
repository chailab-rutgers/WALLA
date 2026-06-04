"""
Inference/evaluation pipeline for multi-LLM wagering methods.
"""

import logging
import pickle
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Ensure local project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wagering.core.model import WhiteboxModel
from wagering.core.dataset import Dataset
from wagering.core.metrics import ECE, bernoulli_kl_divergence, bernoulli_tv_distance

# Local wagering imports
from wagering.methods.base import WageringMethod
from wagering.training.analytics import WageringAnalytics
from wagering.training.trainer import (
    WageringTrainer,
    _compute_model_brier_scores,
    compute_brier_dynamic_regret,
    compute_dynamic_regret,
    compute_meta_metrics,
    compute_normalized_wager_probability_stats,
)
from wagering.aggregation.base import AggregationFunction
from wagering.utils.multi_llm_ensemble import (
    collect_option_logits_and_hidden_states_for_model,
    extract_hidden_state_features,
    get_model_prompt_variant,
    get_model_specific_prompts,
    get_cached_logits_and_hidden_states_for_model,
    set_cached_logits_and_hidden_states_for_model,
    _get_mixed_context_dataset_type,
)

log = logging.getLogger("wagering")

from sklearn.metrics import roc_auc_score


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


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Stable softmax for arrays with last dim = classes."""
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    z = x - m
    ez = np.exp(z)
    denom = np.sum(ez, axis=-1, keepdims=True)
    return ez / (denom + 1e-20)


def _compute_model_brier_scores_soft_binary(
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

    model_probs = _softmax_np(logits)  # [N, M, 2]
    return np.sum((model_probs - y[:, np.newaxis, :]) ** 2, axis=2)


def _debug_log_eval_batch_prob_align(
    *,
    dataset: Dataset,
    dataset_name: str,
    option_tokens: List[str],
    batch_start: int,
    batch_end: int,
    batch_model_probs: np.ndarray,  # [B, M, K]
    batch_labels: np.ndarray,  # [B]
) -> None:
    """
    Inference-time analogue of the training debug logs.

    Uses `dataset.probability_labels` (from `probability_label_column`) when present to build
    soft ground-truth distributions (binary only), otherwise uses one-hot labels.

    Logs:
      - which model is best per row by |P(pos)-Q(pos)|
      - mean KL(q||p_best)
      - mean over rows of mean KL(q||p_nonbest) (nonbest averaged across models first)
      - pct_best_model_is_context_slot for |P(pos)-Q(pos)|, TV, and argmin KL
    """
    p_all = np.asarray(batch_model_probs, dtype=np.float64)
    bsz, num_models, num_options = p_all.shape
    if num_options != 2:
        log.info(
            "eval_debug_prob_align dataset=%s rows=[%d:%d): skipped (num_options=%d != 2)",
            dataset_name,
            int(batch_start),
            int(batch_end),
            int(num_options),
        )
        return

    pos_marker = getattr(dataset, "positive_label", None)
    pos_idx = 0
    if pos_marker is not None:
        try:
            pos_idx = int(option_tokens.index(str(pos_marker).strip()))
        except ValueError:
            pos_idx = 0

    prob_labs = getattr(dataset, "probability_labels", None)
    if prob_labs is None and getattr(dataset, "probabilistic_labels", None) is not None:
        raise ValueError(
            "Dataset provides `probabilistic_labels` but not `probability_labels`. "
            "KL/TV must use `probability_labels` (configured via `probability_label_column`)."
        )

    if isinstance(prob_labs, (list, tuple)) and len(prob_labs) == len(dataset.x):
        target_vec = np.asarray(prob_labs[batch_start:batch_end], dtype=np.float64)
    else:
        target_vec = (np.asarray(batch_labels, dtype=np.int64) == int(pos_idx)).astype(np.float64)

    q = np.zeros((bsz, 2), dtype=np.float64)
    q[:, int(pos_idx)] = target_vec
    q[:, int(1 - int(pos_idx))] = 1.0 - target_vec

    p_pos = p_all[:, :, int(pos_idx)]
    q_pos = q[:, int(pos_idx)]
    abs_err = np.abs(p_pos - q_pos[:, np.newaxis])
    best_m = np.argmin(abs_err, axis=1)

    tv_bm = _tv_distance_per_model(p_all, q)
    best_m_tv = np.argmin(tv_bm, axis=1)

    kl_all = np.empty((bsz, num_models), dtype=np.float64)
    for m in range(num_models):
        kl_all[:, m] = _kl_qp_categorical_rows(q, p_all[:, m, :])
    best_m_kl = np.argmin(kl_all, axis=1)

    p_best = p_all[np.arange(bsz), best_m, :]
    kl_best = _kl_qp_categorical_rows(q, p_best)

    other_mask = np.ones((bsz, num_models), dtype=bool)
    other_mask[np.arange(bsz), best_m] = False
    n_other = np.maximum(other_mask.sum(axis=1), 1)
    kl_other_mean_per_ex = (kl_all * other_mask).sum(axis=1) / n_other

    # Context routing assignment (if present) is stored directly on the dataset.
    ctx_model = np.full(bsz, -1, dtype=np.int64)
    dataset_type = _get_mixed_context_dataset_type(dataset)
    if dataset_type is not None:
        raw = getattr(dataset, f"{dataset_type}_context_assignment_by_example", None)
        if isinstance(raw, list) and len(raw) == len(dataset.x):
            try:
                ctx_model = np.asarray(raw, dtype=np.int64)[batch_start:batch_end]
            except Exception:
                ctx_model = np.full(bsz, -1, dtype=np.int64)

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

    counts = np.bincount(best_m, minlength=num_models)
    log.info(
        "eval_debug_prob_align dataset=%s rows=[%d:%d) |best_model_counts|=%s | "
        "mean_KL_best=%.6f mean_of_mean_KL_nonbest=%.6f | "
        "pct_best_model_is_context_slot=%s pct_best_model_is_context_slot_tv=%s "
        "pct_best_model_is_context_slot_kl=%s (over %d/%d rows with assignment; TV=d_TV=0.5*L1; KL=KL(q||p))",
        dataset_name,
        int(batch_start),
        int(batch_end),
        counts.tolist(),
        float(np.mean(kl_best)),
        float(np.mean(kl_other_mean_per_ex)),
        ctx_pct_str,
        ctx_pct_tv_str,
        ctx_pct_kl_str,
        int(np.count_nonzero(ctx_ok)),
        int(bsz),
    )


class WageringEvaluator:
    """
    Evaluator for multi-LLM wagering methods.
    
    Evaluates on test splits and OOD datasets, computes accuracy, AUC, ECE, and Brier score.
    """
    
    def __init__(
        self,
        models: List[Union[WhiteboxModel, str]],
        wagering_method: WageringMethod,
        aggregation_function: AggregationFunction,
        option_tokens: List[str] = ["A", "B", "C", "D"],
        wandb_logger: Optional[Any] = None,
        checkpoint_dir: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
        training_checkpoint_path: Optional[str] = None,
        seed: Optional[int] = None,
        wandb_starting_step: Optional[int] = None,
        logit_calibrator: Optional[Any] = None,
        model_configs_for_sequential_perplexity: Optional[List[Dict[str, Any]]] = None,
        perplexity_load_cache_kwargs: Optional[Dict[str, Any]] = None,
        debug_batch_prob_alignment: bool = False,
    ):
        """
        Initialize the evaluator.
        
        Args:
            models: ``WhiteboxModel`` instances and/or string checkpoint paths
            model_configs_for_sequential_perplexity: Per-slot merged YAML configs for
                one-at-a-time perplexity loads when VRAM is insufficient for the ensemble.
            perplexity_load_cache_kwargs: Optional kwargs for sequential model loads.
            wagering_method: WageringMethod instance (should be in eval mode)
            aggregation_function: AggregationFunction instance
            option_tokens: List of option tokens (e.g., ["A", "B", "C", "D"])
            wandb_logger: Optional wandb logger for logging metrics
            checkpoint_dir: Optional directory to save/load evaluation checkpoints
            metadata: Optional metadata dict with model_names, training_datasets, etc.
            training_checkpoint_path: Optional path to the training checkpoint used
            seed: Optional random seed used for this run
            wandb_starting_step: Optional starting step for wandb logging (useful when resuming from training)
        """
        self.models = models
        self.wagering_method = wagering_method
        self.aggregation_function = aggregation_function
        self.option_tokens = option_tokens
        self.wandb_logger = wandb_logger
        self.checkpoint_dir = checkpoint_dir
        self.metadata = metadata or {}
        self.training_checkpoint_path = training_checkpoint_path
        self.seed = seed
        self.logit_calibrator = logit_calibrator
        self.hidden_state_layers = getattr(self.wagering_method, "hidden_state_layers", None)
        self.method_requires_model_perplexities = bool(
            getattr(self.wagering_method, "requires_model_perplexities", False)
        )
        self._model_configs_for_sequential_perplexity = model_configs_for_sequential_perplexity
        self._perplexity_load_cache_kwargs = perplexity_load_cache_kwargs or {}
        self.debug_batch_prob_alignment = bool(debug_batch_prob_alignment)

        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.wagering_method.eval_mode()

        # Global step counter for wandb logging across all datasets.
        # This ensures each dataset gets unique step numbers and time series work correctly.
        # Continue from training's last step if provided to avoid wandb step ordering warnings.
        if wandb_starting_step is not None:
            # Use provided starting step (from training's last step).
            self._global_wandb_step = int(wandb_starting_step)
            run_step = self._get_wandb_run_step()
            if run_step is not None:
                self._global_wandb_step = max(self._global_wandb_step, run_step)
            log.info(
                "Initialized wandb step counter to %s (continuing from training)",
                self._global_wandb_step,
            )
        elif self.wandb_logger:
            try:
                # Get current step from wandb run if it exists (when resuming).
                # When wandb resumes a run, we need to continue from where training left off.
                if hasattr(self.wandb_logger, "run") and self.wandb_logger.run is not None:
                    try:
                        run = self.wandb_logger.run
                        if hasattr(run, "step") and run.step is not None:
                            self._global_wandb_step = int(run.step)
                        else:
                            self._global_wandb_step = 0
                            log.warning(
                                "Could not determine wandb step from run. Starting from 0. "
                                "If this is a resumed run, step ordering warnings may occur."
                            )
                    except Exception as e:
                        self._global_wandb_step = 0
                        log.warning("Error getting wandb step: %s, starting from 0", e)
                    else:
                        if self._global_wandb_step > 0:
                            log.info(
                                "Initialized wandb step counter to %s (resumed from training)",
                                self._global_wandb_step,
                            )
                else:
                    self._global_wandb_step = 0
                    log.info("Initialized wandb step counter to 0 (wandb run not available)")
            except Exception as e:
                # If anything goes wrong, start from 0.
                self._global_wandb_step = 0
                log.warning("Could not get wandb step, starting from 0: %s", e)
        else:
            self._global_wandb_step = 0

    def _maybe_load_model_for_collection(
        self, model: Union[WhiteboxModel, str], model_index: int
    ) -> Tuple[WhiteboxModel, bool]:
        """
        Ensure we have a loaded WhiteboxModel when we need to collect logits/hidden states.

        In memory-saving evaluation modes (e.g. sequential prompt-perplexity), ``self.models``
        may contain string paths. If a logits cache entry is missing, we can lazily load
        the model for that index *only* to build the missing cache artifact.

        Returns:
            (whitebox_model, was_loaded_here)
        """
        if not isinstance(model, str):
            return model, False

        cfgs = self._model_configs_for_sequential_perplexity
        if cfgs is None or model_index < 0 or model_index >= len(cfgs):
            raise RuntimeError(
                f"Cache miss for model path {model}. Model must be loaded to collect logits."
            )

        from wagering.utils.model_utils import load_models_from_config

        loaded, _ = load_models_from_config(
            [cfgs[model_index]],
            cache_kwargs=self._perplexity_load_cache_kwargs,
            share_identical_models=True,
        )
        return loaded[0], True

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

    def _advance_wandb_step(self) -> int:
        """Advance internal wandb step while staying monotonic with run.step."""
        next_step = self._global_wandb_step + 1
        run_step = self._get_wandb_run_step()
        if run_step is not None:
            next_step = max(next_step, run_step + 1)
        self._global_wandb_step = next_step
        return self._global_wandb_step

    def _log_wandb_plot(self, payload: Dict[str, Any]) -> None:
        """Log plot payloads to wandb using a safe monotonically increasing step."""
        if not self.wandb_logger:
            return

        log_step = self._advance_wandb_step()
        if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
            self.wandb_logger.run.log(payload, step=log_step, commit=True)
        else:
            self.wandb_logger.log(payload, step=log_step, commit=True)

    @staticmethod
    def _compute_prompt_perplexities_for_model(
        model: WhiteboxModel,
        prompts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        """
        Compute teacher-forced prompt perplexity per example for one model.

        Returns:
            np.ndarray of shape [num_examples].
        """
        import torch

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
        import gc
        import json

        import torch
        from wagering.utils.model_utils import load_models_from_config

        num_examples = len(dataset.x)
        num_models = len(self.models)
        cfgs = self._model_configs_for_sequential_perplexity
        if cfgs is None or len(cfgs) != num_models:
            raise RuntimeError("Sequential eval perplexity requires model configs matching ensemble size")

        all_perplexities = np.empty((num_examples, num_models), dtype=np.float32)
        batch_size = max(1, int(dataset.batch_size))
        log.info(
            "Computing eval prompt perplexities sequentially (%d models; %d visible CUDA device(s))",
            num_models,
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        )

        def _shared_model_key(model_cfg: Dict[str, Any]) -> str:
            try:
                return json.dumps(model_cfg, sort_keys=True, default=str)
            except TypeError:
                return repr(model_cfg)

        # Group ensemble slots by identical model configs so we only load a given
        # set of weights once, while still computing per-slot perplexities for
        # potentially different prompts (e.g. mixed-context prompt routing).
        key_to_indices: Dict[str, List[int]] = {}
        for model_index, cfg in enumerate(cfgs):
            key_to_indices.setdefault(_shared_model_key(cfg), []).append(model_index)

        for shared_key, model_indices in key_to_indices.items():
            loaded, _ = load_models_from_config(
                [cfgs[model_indices[0]]],
                cache_kwargs=self._perplexity_load_cache_kwargs,
                share_identical_models=True,
            )
            wb = loaded[0]
            try:
                for model_index in model_indices:
                    model_prompts = get_model_specific_prompts(dataset, model_index=model_index)
                    if len(model_prompts) != num_examples:
                        raise ValueError(
                            "Prompt/label length mismatch while computing eval prompt perplexities. "
                            f"prompts={len(model_prompts)}, examples={num_examples}"
                        )
                    all_perplexities[:, model_index] = WageringTrainer._compute_prompt_perplexities_for_model(
                        wb,
                        model_prompts,
                        batch_size,
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
                    "Prompt-perplexity wagering requires loaded model objects, "
                    f"but model at index {model_index} is a string path: {model}. "
                    "With more models than visible GPUs, pass model_configs_for_sequential_perplexity."
                )

            model_prompts = get_model_specific_prompts(dataset, model_index=model_index)
            if len(model_prompts) != num_examples:
                raise ValueError(
                    "Prompt/label length mismatch while computing eval prompt perplexities. "
                    f"prompts={len(model_prompts)}, examples={num_examples}"
                )

            all_perplexities[:, model_index] = self._compute_prompt_perplexities_for_model(
                model=model,
                prompts=model_prompts,
                batch_size=max(1, int(dataset.batch_size)),
            )

        return all_perplexities
    
    def evaluate(
        self,
        dataset: Dataset,
        dataset_name: str = "test",
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset.
        
        Uses shared cache to avoid recomputing logits and hidden states for the same models and datasets
        across different wagering methods. This is the default behavior since LLMs are not updated.
        
        TODO: Methods that update LLMs during evaluation should disable caching.
        
        Args:
            dataset: Dataset to evaluate on
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Dictionary with evaluation results and metrics
        """
        log.info(f"Evaluating on {dataset_name} ({len(dataset.x)} examples)")
        
        # Check if wagering method requires LLM hidden states for routing.
        # RouteLLMBertWagers routes on BERT-encoded prompts only (arXiv:2406.18665); skip HS unless calibrating logits.
        wagering_method_name = type(self.wagering_method).__name__
        needs_hidden_states = (
            wagering_method_name not in [
                "EqualWagers",
                "ZeroOneWagers",
                "OneZeroWagers",
                "RouteLLMBertWagers",
                "RouterDCWagers",
            ]
            or self.logit_calibrator is not None
        )
        
        # Check cache per model and collect if needed
        all_model_logits_list = []
        all_model_hidden_states_list = [] if needs_hidden_states else None
        all_model_calibration_hidden_states_list = [] if self.logit_calibrator is not None else None
        labels = None
        
        for i, model in enumerate(self.models):
            # Try to load from cache for this model
            model_path = model if isinstance(model, str) else model.model_path
            prompt_variant = get_model_prompt_variant(dataset, model_index=i)
            cached_logits, cached_hidden_states, cached_labels = get_cached_logits_and_hidden_states_for_model(
                model_path,
                dataset,
                self.option_tokens,
                prompt_variant=prompt_variant,
                model_index=i,
                hidden_state_layers=self.hidden_state_layers,
            )
            
            if cached_logits is not None:
                log.debug(f"Model {i+1}/{len(self.models)}: Using cached logits")
                all_model_logits_list.append(cached_logits)
                
                if needs_hidden_states:
                    if cached_hidden_states is not None:
                        log.info(f"Model {i+1}/{len(self.models)}: Using cached hidden states")
                        all_model_hidden_states_list.append(cached_hidden_states)
                        if all_model_calibration_hidden_states_list is not None:
                            calibration_hidden_states = get_cached_logits_and_hidden_states_for_model(
                                model_path,
                                dataset,
                                self.option_tokens,
                                prompt_variant=prompt_variant,
                                model_index=i,
                                hidden_state_layers=[-1],
                            )[1]
                            if calibration_hidden_states is None:
                                raise RuntimeError(
                                    "Temperature calibration requires last-layer hidden states in cache"
                                )
                            all_model_calibration_hidden_states_list.append(calibration_hidden_states)
                    else:
                        log.info(f"Model {i+1}/{len(self.models)}: Hidden states not cached - will collect")
                        # Need to collect for this model
                        wb_model, loaded_here = self._maybe_load_model_for_collection(model, i)
                        model_logits, model_hidden_states_all_layers, model_labels = collect_option_logits_and_hidden_states_for_model(
                            wb_model,
                            dataset,
                            self.option_tokens,
                            model_identifier=str(model_path),
                            model_index=i,
                            hidden_state_layers=self.hidden_state_layers,
                            model_prompts=get_model_specific_prompts(dataset, model_index=i),
                            prompt_variant=prompt_variant,
                        )
                        # Update cache with hidden states
                        set_cached_logits_and_hidden_states_for_model(
                            wb_model,
                            dataset,
                            self.option_tokens,
                            model_logits,
                            model_hidden_states_all_layers,
                            model_labels,
                            prompt_variant=prompt_variant,
                            model_index=i,
                            hidden_state_layers=self.hidden_state_layers,
                        )
                        if loaded_here:
                            try:
                                import gc
                                import torch

                                del wb_model.model
                                del wb_model.tokenizer
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                        model_hidden_states = extract_hidden_state_features(
                            model_hidden_states_all_layers,
                            self.hidden_state_layers,
                        )
                        if model_hidden_states is None:
                            raise RuntimeError(
                                "Hidden-state cache is in legacy format and cannot satisfy hidden_state_layers; recache is required"
                            )
                        all_model_logits_list[-1] = model_logits  # Use freshly collected logits
                        all_model_hidden_states_list.append(model_hidden_states)
                        if all_model_calibration_hidden_states_list is not None:
                            calibration_hidden_states = extract_hidden_state_features(
                                model_hidden_states_all_layers,
                                [-1],
                            )
                            if calibration_hidden_states is None:
                                raise RuntimeError(
                                    "Temperature calibration requires last-layer hidden states"
                                )
                            all_model_calibration_hidden_states_list.append(calibration_hidden_states)
                        labels = model_labels
                
                # Set labels from cache if not already set
                if labels is None:
                    labels = cached_labels
            else:
                # Cache miss - collect both logits and hidden states
                log.info(f"Model {i+1}/{len(self.models)}: Cache miss - collecting logits and hidden states")
                wb_model, loaded_here = self._maybe_load_model_for_collection(model, i)
                model_logits, model_hidden_states_all_layers, model_labels = collect_option_logits_and_hidden_states_for_model(
                    wb_model,
                    dataset,
                    self.option_tokens,
                    model_identifier=str(model_path),
                    model_index=i,
                    hidden_state_layers=self.hidden_state_layers,
                    model_prompts=get_model_specific_prompts(dataset, model_index=i),
                    prompt_variant=prompt_variant,
                )
                
                # Cache the results for this model
                set_cached_logits_and_hidden_states_for_model(
                    wb_model,
                    dataset,
                    self.option_tokens,
                    model_logits,
                    model_hidden_states_all_layers,
                    model_labels,
                    prompt_variant=prompt_variant,
                    model_index=i,
                    hidden_state_layers=self.hidden_state_layers,
                )
                if loaded_here:
                    try:
                        import gc
                        import torch

                        del wb_model.model
                        del wb_model.tokenizer
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                model_hidden_states = extract_hidden_state_features(
                    model_hidden_states_all_layers,
                    self.hidden_state_layers,
                )
                if model_hidden_states is None:
                    raise RuntimeError(
                        "Hidden-state cache is in legacy format and cannot satisfy hidden_state_layers; recache is required"
                    )
                
                all_model_logits_list.append(model_logits)
                if needs_hidden_states:
                    all_model_hidden_states_list.append(model_hidden_states)
                if all_model_calibration_hidden_states_list is not None:
                    calibration_hidden_states = extract_hidden_state_features(
                        model_hidden_states_all_layers,
                        [-1],
                    )
                    if calibration_hidden_states is None:
                        raise RuntimeError(
                            "Temperature calibration requires last-layer hidden states"
                        )
                    all_model_calibration_hidden_states_list.append(calibration_hidden_states)
                labels = model_labels
        
        # Stack into final arrays
        all_model_logits = np.stack(all_model_logits_list, axis=0)  # [num_models, num_examples, num_options]
        
        if needs_hidden_states and all_model_hidden_states_list:
            # Check if all hidden states have the same shape
            hidden_dims = [hs.shape[-1] for hs in all_model_hidden_states_list]
            if len(set(hidden_dims)) == 1:
                # All same dimension - stack into single array
                all_model_hidden_states = np.stack(all_model_hidden_states_list, axis=0)
                log.info(f"Stacked hidden states: shape {all_model_hidden_states.shape}")
            else:
                # Different dimensions - keep as list
                log.info(f"Models have different hidden dimensions: {hidden_dims}. Keeping as list.")
                all_model_hidden_states = all_model_hidden_states_list
        else:
            all_model_hidden_states = None
        
        # Ensure labels are numpy array
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, dtype=np.int32)

        if self.logit_calibrator is not None:
            if all_model_calibration_hidden_states_list is None:
                raise RuntimeError("Temperature calibration requires cached hidden states during evaluation")

            calibration_hidden_dims = [hs.shape[-1] for hs in all_model_calibration_hidden_states_list]
            if len(set(calibration_hidden_dims)) == 1:
                calibration_hidden_states = np.stack(all_model_calibration_hidden_states_list, axis=0)
            else:
                calibration_hidden_states = all_model_calibration_hidden_states_list

            context_assignments = None
            dataset_type = _get_mixed_context_dataset_type(dataset)
            if dataset_type is not None:
                raw = getattr(dataset, f"{dataset_type}_context_assignment_by_example", None)
                if isinstance(raw, list) and len(raw) == len(dataset.x):
                    context_assignments = np.asarray(raw, dtype=np.int64)

            try:
                if context_assignments is not None:
                    all_model_logits = self.logit_calibrator.apply_to_stacked_logits(
                        all_model_logits,
                        calibration_hidden_states,
                        context_model_index_by_example=context_assignments,
                    )
                else:
                    all_model_logits = self.logit_calibrator.apply_to_stacked_logits(
                        all_model_logits,
                        calibration_hidden_states,
                    )
            except TypeError:
                all_model_logits = self.logit_calibrator.apply_to_stacked_logits(
                    all_model_logits,
                    calibration_hidden_states,
                )
            log.info("Applied frozen temperature scaling to cached evaluation logits")

        num_examples = all_model_logits.shape[1]
        model_perplexities = None
        if self.method_requires_model_perplexities:
            model_perplexities = self._compute_prompt_perplexities(dataset)
            if model_perplexities.shape != (num_examples, len(self.models)):
                raise RuntimeError(
                    "Computed model_perplexities has unexpected shape: "
                    f"got {model_perplexities.shape}, expected {(num_examples, len(self.models))}"
                )
            log.info(
                "Computed evaluation prompt perplexities with shape %s",
                model_perplexities.shape,
            )
        
        # Evaluate on all examples in batches for efficiency
        all_predictions = []
        all_aggregated_probs = []
        wagers_history = []  # Track wagers for each example
        total_payout_history = []  # Track per-example per-model net payout when available
        sigmoid_wagers_history = []  # Track unnormalized wagers when provided

        # Per-batch inference timing (seconds) for compute_wagers + aggregation.
        inference_times_s: List[float] = []
        
        # Running metrics for per-step logging
        running_correct = 0
        running_nll_sum = 0.0
        
        eval_batch_size = 100  # Process evaluation in batches of 100
        
        for batch_start in range(0, num_examples, eval_batch_size):
            batch_end = min(batch_start + eval_batch_size, num_examples)
            batch_size_actual = batch_end - batch_start
            
            # Get batch of logits
            batch_logits = all_model_logits[:, batch_start:batch_end, :]  # [num_models, batch_size, num_options]
            batch_logits_transposed = np.transpose(batch_logits, (1, 0, 2))  # [batch_size, num_models, num_options]
            batch_labels = labels[batch_start:batch_end]  # [batch_size]

            if self.debug_batch_prob_alignment:
                # Per-model predicted class distributions for this batch.
                batch_model_probs = _softmax_np(batch_logits_transposed)
                _debug_log_eval_batch_prob_align(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    option_tokens=self.option_tokens,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    batch_model_probs=batch_model_probs,
                    batch_labels=batch_labels,
                )
            
            # Get questions for batch (for wagering methods that need them)
            batch_questions = dataset.x[batch_start:batch_end]  # List of question strings
            batch_questions_per_model = None
            if bool(getattr(self.wagering_method, "expects_per_model_router_prompts", False)):
                if _get_mixed_context_dataset_type(dataset) is not None:
                    if bool(getattr(self.wagering_method, "pubmedqa_strip_context", False)):
                        without_ctx = getattr(dataset, "pubmedqa_without_context_x", None)
                        if isinstance(without_ctx, list) and len(without_ctx) >= batch_end:
                            slice_without = without_ctx[batch_start:batch_end]
                        else:
                            slice_without = batch_questions
                        batch_questions_per_model = [list(slice_without) for _ in range(len(self.models))]
                    else:
                        batch_questions_per_model = [
                            get_model_specific_prompts(dataset, model_index=mi)[batch_start:batch_end]
                            for mi in range(len(self.models))
                        ]
            
            # Prepare hidden states for batch if available
            batch_hidden_states = None
            if all_model_hidden_states is not None:
                if isinstance(all_model_hidden_states, list):
                    # List of arrays with different dimensions - keep as list
                    batch_hidden_states = []
                    for i in range(len(all_model_hidden_states)):
                        model_hs = all_model_hidden_states[i][batch_start:batch_end, :]  # [batch_size, hidden_dim_i]
                        batch_hidden_states.append(model_hs)
                else:
                    # Stacked array: [num_models, num_examples, hidden_dim]
                    batch_hidden_states_array = all_model_hidden_states[:, batch_start:batch_end, :]
                    # Convert to list of [num_models] arrays, each [batch_size, hidden_dim]
                    batch_hidden_states = [batch_hidden_states_array[i, :, :] for i in range(batch_hidden_states_array.shape[0])]
            
            # Compute wagers for batch
            wagering_kwargs: Dict[str, Any] = {}
            if model_perplexities is not None:
                wagering_kwargs["model_perplexities"] = model_perplexities[batch_start:batch_end]

            t0 = time.perf_counter()
            res_dict = self.wagering_method.compute_wagers(
                model_logits=batch_logits_transposed,
                gold_label=batch_labels,
                hidden_states_list=batch_hidden_states,
                questions=batch_questions,
                questions_per_model=batch_questions_per_model,
                **wagering_kwargs,
            )  # [batch_size, num_models]
            batch_wagers = res_dict["wagers"]
            batch_total_payout = res_dict.get("total_payout", None)
            batch_sigmoid_wagers = res_dict.get("sigmoid_wagers", None)
            # Aggregate predictions for batch
            batch_aggregated_log_probs, batch_aggregated_probs = self.aggregation_function.aggregate(
                batch_logits_transposed, batch_wagers
            )  # [batch_size, num_options] each
            inference_times_s.append(float(time.perf_counter() - t0))
            
            batch_predictions = np.argmax(batch_aggregated_probs, axis=1)  # [batch_size]
            
            all_predictions.extend(batch_predictions.tolist())
            all_aggregated_probs.extend(batch_aggregated_probs.tolist())
            wagers_history.extend(batch_wagers.tolist())
            if batch_total_payout is not None:
                try:
                    total_payout_history.extend(np.asarray(batch_total_payout).tolist())
                except Exception:
                    pass
            if batch_sigmoid_wagers is not None:
                try:
                    sigmoid_wagers_history.extend(np.asarray(batch_sigmoid_wagers).tolist())
                except Exception:
                    pass
            
            # Compute batch metrics using vectorized operations
            batch_correct = (batch_predictions == batch_labels)
            batch_nll = -np.log(batch_aggregated_probs[np.arange(batch_size_actual), batch_labels] + 1e-10)
            
            # Update running metrics
            running_correct += int(np.sum(batch_correct))
            running_nll_sum += np.sum(batch_nll)
            running_accuracy = running_correct / (batch_end)
            running_nll = running_nll_sum / batch_end
            
            # Log batch-level metrics to wandb
            if self.wandb_logger:
                log_prefix = "test" if not dataset_name.startswith("ood_") else "ood"
                
                # Inverse HHI (effective number of models) per example, then averaged over the batch.
                # N_eff = (sum_i w_i)^2 / (sum_i w_i^2). For normalized weights, this is 1 / sum_i w_i^2.
                try:
                    w = np.asarray(batch_wagers, dtype=np.float64)
                    sum_w = np.sum(w, axis=1)
                    sum_w2 = np.sum(w * w, axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        n_eff = np.divide(sum_w * sum_w, sum_w2)
                    batch_inverse_hhi = float(np.nanmean(n_eff))
                except Exception:
                    batch_inverse_hhi = float("nan")

                # Log batch average metrics
                wandb_log_dict = {
                    f"{log_prefix}/{dataset_name}/batch/accuracy": float(np.mean(batch_correct)),
                    f"{log_prefix}/{dataset_name}/batch/nll": float(np.mean(batch_nll)),
                    f"{log_prefix}/{dataset_name}/batch/running_accuracy": running_accuracy,
                    f"{log_prefix}/{dataset_name}/batch/running_nll": running_nll,
                    f"{log_prefix}/{dataset_name}/batch/inference_time_s": float(inference_times_s[-1]),
                    f"{log_prefix}/{dataset_name}/batch/inverse_hhi": batch_inverse_hhi,
                }
                
                # Add average wager statistics
                for i in range(batch_wagers.shape[1]):
                    wandb_log_dict[f"{log_prefix}/{dataset_name}/batch/wager_model_{i}"] = float(np.mean(batch_wagers[:, i]))
                
                # Use global step counter to ensure unique steps across all datasets
                log_step = self._advance_wandb_step()
                try:
                    # Use same API pattern as trainer for consistency
                    if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                        self.wandb_logger.run.log(wandb_log_dict, step=log_step)
                    else:
                        self.wandb_logger.log(wandb_log_dict, step=log_step)
                except Exception as e:
                    raise Exception(f"✗ Error logging batch metrics to wandb: {e}", exc_info=True)
        
        # Convert to arrays
        all_predictions = np.array(all_predictions, dtype=np.int32)
        all_aggregated_probs = np.stack(all_aggregated_probs, axis=0)
        wagers_history = np.stack(wagers_history, axis=0)  # [num_examples, num_models]
        if total_payout_history:
            try:
                total_payout_history = np.asarray(total_payout_history, dtype=np.float32)
            except Exception:
                total_payout_history = None
        else:
            total_payout_history = None
        if sigmoid_wagers_history:
            try:
                sigmoid_wagers_history = np.asarray(sigmoid_wagers_history, dtype=np.float32)
            except Exception:
                sigmoid_wagers_history = None
        else:
            sigmoid_wagers_history = None

        # Effective number of models / inverse HHI over examples.
        inverse_hhi = float("nan")
        try:
            w = np.asarray(wagers_history, dtype=np.float64)
            sum_w = np.sum(w, axis=1)
            sum_w2 = np.sum(w * w, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                n_eff = np.divide(sum_w * sum_w, sum_w2)
            inverse_hhi = float(np.nanmean(n_eff))
        except Exception as e:
            log.warning("Could not compute inverse HHI: %s", e)

        # Average inference time per batch in seconds (compute_wagers + aggregation).
        avg_inference_time_per_batch_s = float("nan")
        if inference_times_s:
            try:
                avg_inference_time_per_batch_s = float(np.mean(np.asarray(inference_times_s, dtype=np.float64)))
            except Exception:
                avg_inference_time_per_batch_s = float("nan")
        
        # Compute metrics
        accuracy = np.mean(all_predictions == labels)
        
        # Compute NLL (negative log likelihood) for correct classes
        correct_class_probs = all_aggregated_probs[np.arange(len(labels)), labels]
        nll = -np.mean(np.log(correct_class_probs + 1e-10))

        # Compute multiclass Brier score: mean over examples of sum((p_k - y_k)^2)
        num_options = all_aggregated_probs.shape[1]
        one_hot_labels = np.eye(num_options, dtype=np.float64)[labels]
        brier = np.mean(np.sum((all_aggregated_probs - one_hot_labels) ** 2, axis=1))

        # Bernoulli KL / TV for binary tasks: D_KL(Bernoulli(target) || Bernoulli(pred)),
        # TV = mean |p_pred - p_target|. Target is soft if dataset provides probability labels.
        bernoulli_kl = float("nan")
        bernoulli_tv = float("nan")
        prob_labs = None
        if num_options == 2:
            pos_marker = getattr(dataset, "positive_label", None)
            pos_idx = 0
            if pos_marker is not None:
                try:
                    pos_idx = int(self.option_tokens.index(str(pos_marker).strip()))
                except ValueError:
                    pos_idx = 0
            pred_vec = all_aggregated_probs[:, pos_idx].astype(np.float64, copy=False)
            # Strict: use `probability_labels` (fed by `probability_label_column`, e.g. posterior_prob).
            prob_labs = getattr(dataset, "probability_labels", None)
            if prob_labs is None and getattr(dataset, "probabilistic_labels", None) is not None:
                raise ValueError(
                    "Dataset provides `probabilistic_labels` but not `probability_labels`. "
                    "KL/TV must use `probability_labels` (configured via `probability_label_column`)."
                )
            if (
                isinstance(prob_labs, (list, tuple))
                and len(prob_labs) == num_examples
            ):
                target_vec = np.asarray(prob_labs, dtype=np.float64)
            else:
                target_vec = (labels == pos_idx).astype(np.float64)
            try:
                # print(target_vec)
                # print("-"*50)
                # print(pred_vec)
                # 0/0
                bernoulli_kl = bernoulli_kl_divergence(pred_vec.tolist(), target_vec.tolist())
                bernoulli_tv = bernoulli_tv_distance(pred_vec.tolist(), target_vec.tolist())
            except ValueError as e_kl:
                log.warning("Could not compute Bernoulli KL/TV for %s: %s", dataset_name, e_kl)

            # If soft labels are present, also compute Brier against the soft distribution.
            if isinstance(prob_labs, (list, tuple)) and len(prob_labs) == num_examples:
                gt_pos = np.asarray(prob_labs, dtype=np.float64)
                y_soft = np.zeros((num_examples, 2), dtype=np.float64)
                y_soft[:, int(pos_idx)] = gt_pos
                y_soft[:, int(1 - int(pos_idx))] = 1.0 - gt_pos
                brier = float(np.mean(np.sum((all_aggregated_probs - y_soft) ** 2, axis=1)))
        
        # Compute AUC
        auc = None
        # Use max probability as confidence score
        max_probs = all_aggregated_probs.max(axis=1)
        correctness = (all_predictions == labels).astype(int)
        
        if len(np.unique(correctness)) >= 2:
            try:
                auc = roc_auc_score(correctness, max_probs)
            except ValueError:
                log.warning("Could not compute AUC (all predictions same class)")
                auc = np.nan
        else:
            auc = np.nan
        
        # Compute ECE
        ece = None
        try:
            # Confidences are already probabilities in [0, 1], so do not re-normalize.
            ece_metric = ECE(normalize=False, n_bins=20)
            confidences = all_aggregated_probs.max(axis=1)
            correctness = (all_predictions == labels).astype(float)

            finite_mask = np.isfinite(confidences) & np.isfinite(correctness)
            if not np.any(finite_mask):
                raise ValueError("No finite confidence/correctness pairs available for ECE")

            if np.sum(finite_mask) < len(confidences):
                dropped = int(len(confidences) - np.sum(finite_mask))
                log.warning(
                    "Dropping %d non-finite samples before ECE computation for dataset %s",
                    dropped,
                    dataset_name,
                )

            ece = ece_metric(
                confidences[finite_mask].tolist(),
                correctness[finite_mask].tolist(),
            )
        except Exception as e:
            log.warning(f"Could not compute ECE: {e}")
            ece = np.nan
        
        # Compute Dynamic Regret and Meta Metrics
        d_regret = None
        brier_d_regret = None
        meta_acc = None
        meta_nll = None
        meta_auc = None
        kendall_tau = None
        best_model_mrr = None
        wager_prob_mean_per_model = None
        wager_prob_var_per_model = None
        brier_best_wager_prob_mean = None
        brier_best_wager_prob_var = None
        try:
            # Get model logits in the right format [num_examples, num_models, num_options]
            model_logits = np.transpose(all_model_logits, (1, 0, 2))
            d_regret, best_expert_ids = compute_dynamic_regret(
                model_logits, all_aggregated_probs, labels
            )
            # For binary datasets with soft ground-truth (e.g. cluster_saturation_bayesX),
            # align Brier dynamic regret with training/validation by using probability labels
            # when available (otherwise fall back to one-hot via `labels`).
            if num_options == 2 and isinstance(prob_labs, (list, tuple)) and len(prob_labs) == num_examples:
                pos_marker = getattr(dataset, "positive_label", None)
                pos_idx = 0
                if pos_marker is not None:
                    try:
                        pos_idx = int(self.option_tokens.index(str(pos_marker).strip()))
                    except ValueError:
                        pos_idx = 0
                brier_d_regret = compute_brier_dynamic_regret(
                    model_logits,
                    all_aggregated_probs,
                    labels,
                    gt_positive_probs=np.asarray(prob_labs, dtype=np.float64),
                    positive_option_index=int(pos_idx),
                )
            else:
                brier_d_regret = compute_brier_dynamic_regret(
                    model_logits, all_aggregated_probs, labels
                )
            if num_options == 2 and isinstance(prob_labs, (list, tuple)) and len(prob_labs) == num_examples:
                model_brier_scores = _compute_model_brier_scores_soft_binary(
                    model_logits,
                    gt_positive_probs=np.asarray(prob_labs, dtype=np.float64),
                    positive_option_index=int(pos_idx),
                )
            else:
                model_brier_scores = _compute_model_brier_scores(model_logits, labels)

            meta_metrics = compute_meta_metrics(
                wagers_history,
                best_expert_ids,
                model_brier_scores,
            )
            meta_acc = meta_metrics["meta_acc"]
            meta_nll = meta_metrics["meta_nll"]
            meta_auc = meta_metrics["meta_auc"]
            kendall_tau = meta_metrics["kendall_tau"]
            best_model_mrr = meta_metrics["best_model_mrr"]

            try:
                brier_best_model_ids = np.argmin(model_brier_scores, axis=1)
                wager_prob_stats = compute_normalized_wager_probability_stats(
                    wagers_history, brier_best_model_ids
                )
                wager_prob_mean_per_model = wager_prob_stats["wager_prob_mean_per_model"]
                wager_prob_var_per_model = wager_prob_stats["wager_prob_var_per_model"]
                brier_best_wager_prob_mean = wager_prob_stats["brier_best_wager_prob_mean"]
                brier_best_wager_prob_var = wager_prob_stats["brier_best_wager_prob_var"]
            except Exception as e_wps:
                log.warning("Could not compute normalized wager probability stats: %s", e_wps)
        except Exception as e:
            log.warning(f"Could not compute d_regret/meta metrics: {e}")
        
        # Average wagers / net payout per model (when provided by the wagering method).
        avg_wager_per_model = None
        avg_wager_total = None
        try:
            avg_wager_per_model = np.mean(np.asarray(wagers_history, dtype=np.float64), axis=0).astype(np.float64)
            avg_wager_total = float(np.mean(np.sum(np.asarray(wagers_history, dtype=np.float64), axis=1)))
        except Exception:
            avg_wager_per_model = None
            avg_wager_total = None

        avg_sigmoid_wager_per_model = None
        avg_sigmoid_wager_total = None
        if sigmoid_wagers_history is not None:
            try:
                sw = np.asarray(sigmoid_wagers_history, dtype=np.float64)
                if sw.ndim == 2 and sw.shape[0] == wagers_history.shape[0]:
                    avg_sigmoid_wager_per_model = np.mean(sw, axis=0).astype(np.float64)
                    avg_sigmoid_wager_total = float(np.mean(np.sum(sw, axis=1)))
            except Exception:
                avg_sigmoid_wager_per_model = None
                avg_sigmoid_wager_total = None

        avg_net_payout_per_model = None
        avg_net_payout_total = None
        if total_payout_history is not None:
            try:
                payout_arr = np.asarray(total_payout_history, dtype=np.float64)
                if payout_arr.ndim == 2 and payout_arr.shape[0] == wagers_history.shape[0]:
                    avg_net_payout_per_model = np.mean(payout_arr, axis=0).astype(np.float64)
                    avg_net_payout_total = float(np.mean(np.sum(payout_arr, axis=1)))
            except Exception:
                avg_net_payout_per_model = None
                avg_net_payout_total = None

        results = {
            "dataset_name": dataset_name,
            "num_examples": num_examples,
            "predictions": all_predictions,
            "aggregated_probs": all_aggregated_probs,
            "labels": labels,
            "wagers_history": wagers_history,
            # NOTE: per-example payout arrays can be very large; we persist them to disk below when available.
            "avg_wager_per_model": avg_wager_per_model,
            "avg_wager_total": avg_wager_total,
            "avg_sigmoid_wager_per_model": avg_sigmoid_wager_per_model,
            "avg_sigmoid_wager_total": avg_sigmoid_wager_total,
            "avg_net_payout_per_model": avg_net_payout_per_model,
            "avg_net_payout_total": avg_net_payout_total,
            "inverse_hhi": inverse_hhi,
            "avg_inference_time_per_batch_s": avg_inference_time_per_batch_s,
            "accuracy": accuracy,
            "nll": nll,
            "brier": brier,
            "bernoulli_kl": bernoulli_kl,
            "bernoulli_tv": bernoulli_tv,
            "auc": auc,
            "ece": ece,
            "d_regret": d_regret,
            "brier_d_regret": brier_d_regret,
            "meta_acc": meta_acc,
            "meta_nll": meta_nll,
            "meta_auc": meta_auc,
            "kendall_tau": kendall_tau,
            "best_model_mrr": best_model_mrr,
            "wager_prob_mean_per_model": wager_prob_mean_per_model,
            "wager_prob_var_per_model": wager_prob_var_per_model,
            "brier_best_wager_prob_mean": brier_best_wager_prob_mean,
            "brier_best_wager_prob_var": brier_best_wager_prob_var,
        }

        # ------------------------------------------------------------------
        # Subset metrics: only examples where at least one base model is wrong.
        # (i.e., exclude examples where all models' argmax prediction is correct)
        # ------------------------------------------------------------------
        try:
            # model_logits is [num_examples, num_models, num_options] when available.
            if "model_logits" not in locals():
                model_logits = np.transpose(all_model_logits, (1, 0, 2))

            base_preds = np.argmax(model_logits, axis=2)  # [num_examples, num_models]
            base_correct = base_preds == labels[:, None]
            subset_mask = ~np.all(base_correct, axis=1)
            subset_n = int(np.sum(subset_mask))

            subset_metrics: Dict[str, Any] = {
                "subset_name": "any_model_wrong",
                "num_examples": subset_n,
            }

            if subset_n > 0:
                sub_labels = labels[subset_mask]
                sub_probs = all_aggregated_probs[subset_mask]
                sub_preds = all_predictions[subset_mask]
                sub_wagers = wagers_history[subset_mask]
                sub_model_logits = model_logits[subset_mask]

                subset_metrics["accuracy"] = float(np.mean(sub_preds == sub_labels))
                correct_probs = sub_probs[np.arange(len(sub_labels)), sub_labels]
                subset_metrics["nll"] = float(-np.mean(np.log(correct_probs + 1e-10)))

                num_options = sub_probs.shape[1]
                one_hot = np.eye(num_options, dtype=np.float64)[sub_labels]
                subset_metrics["brier"] = float(np.mean(np.sum((sub_probs - one_hot) ** 2, axis=1)))

                # AUC on correctness vs confidence (max prob)
                conf = sub_probs.max(axis=1)
                corr = (sub_preds == sub_labels).astype(int)
                if len(np.unique(corr)) >= 2:
                    try:
                        subset_metrics["auc"] = float(roc_auc_score(corr, conf))
                    except Exception:
                        subset_metrics["auc"] = float("nan")
                else:
                    subset_metrics["auc"] = float("nan")

                # ECE
                try:
                    ece_metric = ECE(normalize=False, n_bins=20)
                    finite_mask = np.isfinite(conf) & np.isfinite(corr.astype(float))
                    if np.any(finite_mask):
                        subset_metrics["ece"] = float(
                            ece_metric(conf[finite_mask].tolist(), corr.astype(float)[finite_mask].tolist())
                        )
                    else:
                        subset_metrics["ece"] = float("nan")
                except Exception:
                    subset_metrics["ece"] = float("nan")

                # Inverse HHI on subset
                try:
                    w = np.asarray(sub_wagers, dtype=np.float64)
                    sum_w = np.sum(w, axis=1)
                    sum_w2 = np.sum(w * w, axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        n_eff = np.divide(sum_w * sum_w, sum_w2)
                    subset_metrics["inverse_hhi"] = float(np.nanmean(n_eff))
                except Exception:
                    subset_metrics["inverse_hhi"] = float("nan")

                # Avg inference time per batch doesn't change with filtering; keep same.
                subset_metrics["avg_inference_time_per_batch_s"] = avg_inference_time_per_batch_s

                # Dynamic regret + meta metrics on subset
                try:
                    sub_d_regret, sub_best_expert_ids = compute_dynamic_regret(
                        sub_model_logits, sub_probs, sub_labels
                    )
                    subset_metrics["d_regret"] = sub_d_regret
                    if num_options == 2 and isinstance(prob_labs, (list, tuple)) and len(prob_labs) == num_examples:
                        pos_marker = getattr(dataset, "positive_label", None)
                        pos_idx = 0
                        if pos_marker is not None:
                            try:
                                pos_idx = int(self.option_tokens.index(str(pos_marker).strip()))
                            except ValueError:
                                pos_idx = 0
                        gt_pos = np.asarray(prob_labs, dtype=np.float64)[subset_mask]
                        subset_metrics["brier_d_regret"] = compute_brier_dynamic_regret(
                            sub_model_logits,
                            sub_probs,
                            sub_labels,
                            gt_positive_probs=gt_pos,
                            positive_option_index=int(pos_idx),
                        )
                    else:
                        subset_metrics["brier_d_regret"] = compute_brier_dynamic_regret(
                            sub_model_logits, sub_probs, sub_labels
                        )
                    if num_options == 2 and isinstance(prob_labs, (list, tuple)) and len(prob_labs) == num_examples:
                        pos_marker = getattr(dataset, "positive_label", None)
                        pos_idx = 0
                        if pos_marker is not None:
                            try:
                                pos_idx = int(self.option_tokens.index(str(pos_marker).strip()))
                            except ValueError:
                                pos_idx = 0
                        gt_pos = np.asarray(prob_labs, dtype=np.float64)[subset_mask]
                        sub_model_brier_scores = _compute_model_brier_scores_soft_binary(
                            sub_model_logits,
                            gt_positive_probs=gt_pos,
                            positive_option_index=int(pos_idx),
                        )
                    else:
                        sub_model_brier_scores = _compute_model_brier_scores(sub_model_logits, sub_labels)
                    sub_meta = compute_meta_metrics(sub_wagers, sub_best_expert_ids, sub_model_brier_scores)
                    subset_metrics["meta_acc"] = sub_meta.get("meta_acc")
                    subset_metrics["meta_nll"] = sub_meta.get("meta_nll")
                    subset_metrics["meta_auc"] = sub_meta.get("meta_auc")
                    subset_metrics["kendall_tau"] = sub_meta.get("kendall_tau")
                    subset_metrics["best_model_mrr"] = sub_meta.get("best_model_mrr")
                except Exception as e_sub:
                    log.warning("Could not compute subset d_regret/meta metrics: %s", e_sub)

                if sub_probs.shape[1] == 2:
                    sub_pos_marker = getattr(dataset, "positive_label", None)
                    sub_pos_idx = 0
                    if sub_pos_marker is not None:
                        try:
                            sub_pos_idx = int(self.option_tokens.index(str(sub_pos_marker).strip()))
                        except ValueError:
                            sub_pos_idx = 0
                    sub_pred_vec = sub_probs[:, sub_pos_idx].astype(np.float64, copy=False)
                    sub_pl = getattr(dataset, "probability_labels", None)
                    if sub_pl is None and getattr(dataset, "probabilistic_labels", None) is not None:
                        raise ValueError(
                            "Dataset provides `probabilistic_labels` but not `probability_labels`. "
                            "KL/TV must use `probability_labels` (configured via `probability_label_column`)."
                        )
                    if (
                        isinstance(sub_pl, (list, tuple))
                        and len(sub_pl) == num_examples
                    ):
                        sub_target_vec = np.asarray(sub_pl, dtype=np.float64)[subset_mask]
                    else:
                        sub_target_vec = (sub_labels == sub_pos_idx).astype(np.float64)
                    try:
                        subset_metrics["bernoulli_kl"] = bernoulli_kl_divergence(
                            sub_pred_vec.tolist(), sub_target_vec.tolist()
                        )
                        subset_metrics["bernoulli_tv"] = bernoulli_tv_distance(
                            sub_pred_vec.tolist(), sub_target_vec.tolist()
                        )
                    except ValueError:
                        subset_metrics["bernoulli_kl"] = float("nan")
                        subset_metrics["bernoulli_tv"] = float("nan")

            results["subset_any_model_wrong"] = subset_metrics

            # Console output
            if subset_n > 0:
                acc_s = subset_metrics.get("accuracy")
                nll_s = subset_metrics.get("nll")
                ece_s = subset_metrics.get("ece")
                dr_s = subset_metrics.get("d_regret")
                sub_kl = subset_metrics.get("bernoulli_kl")
                sub_tv = subset_metrics.get("bernoulli_tv")
                log.info(
                    "%s - Subset(any model wrong; n=%d) Accuracy: %s, NLL: %s, KL: %s, TV: %s, ECE: %s, DRegret: %s",
                    dataset_name,
                    subset_n,
                    f"{float(acc_s):.4f}" if acc_s is not None and np.isfinite(acc_s) else "N/A",
                    f"{float(nll_s):.4f}" if nll_s is not None and np.isfinite(nll_s) else "N/A",
                    f"{float(sub_kl):.4f}" if sub_kl is not None and np.isfinite(sub_kl) else "N/A",
                    f"{float(sub_tv):.4f}" if sub_tv is not None and np.isfinite(sub_tv) else "N/A",
                    f"{float(ece_s):.4f}" if ece_s is not None and np.isfinite(ece_s) else "N/A",
                    f"{float(dr_s):.4f}" if dr_s is not None and np.isfinite(dr_s) else "N/A",
                )
            else:
                log.info("%s - Subset(any model wrong) is empty (n=0); skipped subset metrics.", dataset_name)
        except Exception as e:
            log.warning("Could not compute subset(any model wrong) metrics: %s", e)
        
        brier_str = f"{brier:.4f}" if brier is not None and not np.isnan(brier) else "N/A"
        auc_str = f"{auc:.4f}" if auc is not None and not np.isnan(auc) else "N/A"
        ece_str = f"{ece:.4f}" if ece is not None and not np.isnan(ece) else "N/A"
        d_regret_str = f"{d_regret:.4f}" if d_regret is not None and not np.isnan(d_regret) else "N/A"
        brier_d_regret_str = f"{brier_d_regret:.4f}" if brier_d_regret is not None and not np.isnan(brier_d_regret) else "N/A"
        meta_acc_str = f"{meta_acc:.4f}" if meta_acc is not None and not np.isnan(meta_acc) else "N/A"
        kendall_tau_str = f"{kendall_tau:.4f}" if kendall_tau is not None and not np.isnan(kendall_tau) else "N/A"
        best_model_mrr_str = f"{best_model_mrr:.4f}" if best_model_mrr is not None and not np.isnan(best_model_mrr) else "N/A"
        bernoulli_kl_str = (
            f"{bernoulli_kl:.4f}" if bernoulli_kl is not None and not np.isnan(bernoulli_kl) else "N/A"
        )
        bernoulli_tv_str = (
            f"{bernoulli_tv:.4f}" if bernoulli_tv is not None and not np.isnan(bernoulli_tv) else "N/A"
        )
        log.info(
            f"{dataset_name} - Accuracy: {accuracy:.4f}, NLL: {nll:.4f}, Brier: {brier_str}, "
            f"KL: {bernoulli_kl_str}, TV: {bernoulli_tv_str}, AUC: {auc_str}, ECE: {ece_str}, "
            f"DRegret: {d_regret_str}, BrierDRegret: {brier_d_regret_str}, MetaAcc: {meta_acc_str}, "
            f"KendallTau: {kendall_tau_str}, BestModelMRR: {best_model_mrr_str}"
        )
        
        # Log average wagers per model
        avg_wagers = np.mean(wagers_history, axis=0)
        wager_info = ", ".join([f"Model {i}: {wager:.4f}" for i, wager in enumerate(avg_wagers)])
        log.info(f"{dataset_name} - Average Wagers: {wager_info}")
        if wager_prob_mean_per_model is not None and wager_prob_var_per_model is not None:
            n_models_wp = len(wager_prob_mean_per_model)
            wp_parts = [
                f"Model {i}: mean={wager_prob_mean_per_model[i]:.4f}, var={wager_prob_var_per_model[i]:.4f}"
                for i in range(n_models_wp)
            ]
            log.info(f"{dataset_name} - Normalized wager prob (mean, var over examples): {', '.join(wp_parts)}")
        if brier_best_wager_prob_mean is not None and brier_best_wager_prob_var is not None:
            log.info(
                f"{dataset_name} - Wager prob on Brier-best expert: "
                f"mean={brier_best_wager_prob_mean:.4f}, var={brier_best_wager_prob_var:.4f}"
            )
        
        # Create analytics dataframe for this evaluation
        training_datasets = self.metadata.get("training_datasets", [])
        if isinstance(training_datasets, str):
            training_datasets = [training_datasets]
        
        # Get dataset size (number of examples evaluated) - used to distinguish different settings
        dataset_size = len(dataset.x) if hasattr(dataset, 'x') and dataset.x is not None else None
        
        analytics_df = WageringAnalytics.create_evaluation_analytics(
            wagering_method=self.wagering_method,
            aggregation_function=self.aggregation_function,
            models=self.models,
            evaluation_dataset_name=dataset_name,
            training_datasets=training_datasets,
            results=results,
            metadata=self.metadata,
            checkpoint_path=self.training_checkpoint_path,
            seed=self.seed,
            dataset_size=dataset_size,
        )
        results["analytics_df"] = analytics_df
        
        # Save analytics dataframe to checkpoint directory
        if self.checkpoint_dir:
            analytics_path = self.checkpoint_dir / f"analytics_{dataset_name}.csv"
            analytics_df.to_csv(analytics_path, index=False)
            log.debug(f"Saved analytics dataframe to {analytics_path}")

            if total_payout_history is not None:
                try:
                    payout_path = self.checkpoint_dir / f"total_payout_history_{dataset_name}.npy"
                    np.save(payout_path, np.asarray(total_payout_history, dtype=np.float32))
                    results["total_payout_history_path"] = str(payout_path)
                    log.debug("Saved total payout history to %s", payout_path)
                except Exception as e:
                    log.warning("Could not save total payout history array: %s", e)

            if sigmoid_wagers_history is not None:
                try:
                    sw_path = self.checkpoint_dir / f"sigmoid_wagers_history_{dataset_name}.npy"
                    np.save(sw_path, np.asarray(sigmoid_wagers_history, dtype=np.float32))
                    results["sigmoid_wagers_history_path"] = str(sw_path)
                    log.debug("Saved sigmoid wagers history to %s", sw_path)
                except Exception as e:
                    log.warning("Could not save sigmoid wagers history array: %s", e)
        
        # Log final evaluation metrics to wandb separately from batch metrics
        # These represent the overall evaluation results, not per-batch metrics
        if self.wandb_logger:
            log_prefix = "test" if not dataset_name.startswith("ood_") else "ood"
            log.debug(f"Logging final metrics to wandb: prefix={log_prefix}, dataset_name={dataset_name}")
            # Use global step counter to ensure final metrics appear after all batch metrics
            final_step = self._advance_wandb_step()
            metric_values = {
                "inverse_hhi": inverse_hhi,
                "avg_inference_time_per_batch_s": avg_inference_time_per_batch_s,
                "accuracy": accuracy,
                "nll": nll,
                "brier": brier if brier is not None and not np.isnan(brier) else None,
                "bernoulli_kl": bernoulli_kl if bernoulli_kl is not None and not np.isnan(bernoulli_kl) else None,
                "bernoulli_tv": bernoulli_tv if bernoulli_tv is not None and not np.isnan(bernoulli_tv) else None,
                "auc": auc if auc is not None and not np.isnan(auc) else None,
                "ece": ece if ece is not None and not np.isnan(ece) else None,
                "d_regret": d_regret if d_regret is not None and not np.isnan(d_regret) else None,
                "brier_d_regret": brier_d_regret if brier_d_regret is not None and not np.isnan(brier_d_regret) else None,
                "meta_acc": meta_acc if meta_acc is not None and not np.isnan(meta_acc) else None,
                "meta_nll": meta_nll if meta_nll is not None and not np.isnan(meta_nll) else None,
                "meta_auc": meta_auc if meta_auc is not None and not np.isnan(meta_auc) else None,
                "kendall_tau": kendall_tau if kendall_tau is not None and not np.isnan(kendall_tau) else None,
                "best_model_mrr": best_model_mrr if best_model_mrr is not None and not np.isnan(best_model_mrr) else None,
                "brier_best_wager_prob_mean": brier_best_wager_prob_mean,
                "brier_best_wager_prob_var": brier_best_wager_prob_var,
            }

            primary_final_prefix = f"{log_prefix}/{dataset_name}/final"
            alias_final_prefix = f"{log_prefix}/final/{dataset_name}"
            wandb_metrics = {f"{primary_final_prefix}/{k}": v for k, v in metric_values.items()}
            wandb_metrics.update({f"{alias_final_prefix}/{k}": v for k, v in metric_values.items()})
            
            # Add average wagers per model
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
            
            try:
                final_plot_step = final_step + 1
                if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                    self.wandb_logger.run.log(wandb_metrics, step=final_step, commit=True)
                    self.wandb_logger.run.log(wandb_metrics, step=final_plot_step, commit=True)
                else:
                    self.wandb_logger.log(wandb_metrics, step=final_step, commit=True)
                    self.wandb_logger.log(wandb_metrics, step=final_plot_step, commit=True)
            except Exception as e:
                raise Exception(f"Error logging final metrics to wandb: {e}", exc_info=True)
        
        # Plot wagers for this evaluation
        self._plot_evaluation_wagers(results)
        
        return results
    
    def _plot_evaluation_wagers(self, results: Dict[str, Any]):
        """
        Plot wagers during evaluation for a dataset.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from typing import List
        
        if "wagers_history" not in results or self.checkpoint_dir is None:
            return
        
        wagers_history = results["wagers_history"]
        dataset_name = results["dataset_name"]
        num_examples, num_models = wagers_history.shape
        
        # Get model names from metadata
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
        
        # Plot wagers over time
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        time_steps = np.arange(1, num_examples + 1)
        
        for i in range(num_models):
            ax.plot(time_steps, wagers_history[:, i], label=model_names[i], alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel("Evaluation Step", fontsize=11)
        ax.set_ylabel("Wager (Weight)", fontsize=11)
        ax.set_title(f"Wagers Over Time - {dataset_name}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.checkpoint_dir / f"wagers_over_time_{dataset_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.debug(f"Saved wagers plot to {save_path}")
        
        if self.wandb_logger:
            import wandb
            log_prefix = "test" if not dataset_name.startswith("ood_") else "ood"
            try:
                self._log_wandb_plot({f"{log_prefix}/{dataset_name}/wagers_plot": wandb.Image(str(save_path))})
            except Exception as e:
                raise Exception(f"Error logging wagers plot: {e}")
        
        plt.close()
        
        # Plot average wagers per model
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        avg_wagers = np.mean(wagers_history, axis=0)
        
        bars = ax.bar(range(num_models), avg_wagers, alpha=0.7, color='steelblue')
        
        # Add value labels on bars
        for bar, wager in zip(bars, avg_wagers):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{wager:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        ax.set_title(f"Average Wagers by Model - {dataset_name}", fontsize=12, fontweight='bold')
        ax.set_xticks(range(num_models))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        avg_save_path = self.checkpoint_dir / f"average_wagers_{dataset_name}.png"
        plt.savefig(avg_save_path, dpi=150, bbox_inches='tight')
        log.debug(f"Saved average wagers plot to {avg_save_path}")
        
        if self.wandb_logger:
            import wandb
            log_prefix = "test" if not dataset_name.startswith("ood_") else "ood"
            try:
                self._log_wandb_plot({f"{log_prefix}/{dataset_name}/average_wagers_plot": wandb.Image(str(avg_save_path))})
            except Exception as e:
                raise Exception(f"Error logging average wagers plot: {e}")
        
        plt.close()
    
    def _save_checkpoint(self, all_results: Dict[str, Any], completed_datasets: List[str]):
        """Save evaluation checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint_file = self.checkpoint_dir / "eval_checkpoint.pkl"
        checkpoint_data = {
            "results": all_results,
            "completed_datasets": completed_datasets,
            "global_wandb_step": self._global_wandb_step,  # Save global step counter
        }
        
        try:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data, f)
            log.info(f"Saved evaluation checkpoint to {checkpoint_file}")
        except Exception as e:
            raise Exception(f"Failed to save checkpoint: {e}")
    
    def _plot_average_wagers_across_datasets(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        datasets_list: List[Tuple[Dataset, str]],
        eval_type: str = "test",
    ):
        """
        Plot average wagers grouped by dataset (aggregated across multiple evaluation datasets).
        
        Args:
            results_dict: Dictionary of results from evaluate() calls
            datasets_list: List of (dataset, name) tuples
            eval_type: Either "test" or "ood" for logging prefix
        """
        if self.checkpoint_dir is None or not results_dict:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Collect model names
        model_names: List[str] = []
        num_models = None
        
        # Get first result to determine number of models
        first_result = next(iter(results_dict.values())) if results_dict else None
        if first_result and "wagers_history" in first_result:
            num_models = first_result["wagers_history"].shape[1]
            
            # Try to get model names from metadata
            if isinstance(self.metadata, dict) and "model_names" in self.metadata:
                raw_names = self.metadata["model_names"]
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
        
        if num_models is None:
            log.warning("Could not determine number of models for plotting")
            return
        
        # Prepare data for plotting: average wagers per dataset
        num_datasets = len(datasets_list)
        dataset_names = [name for _, name in datasets_list]
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        x = np.arange(num_datasets)
        width = 0.8 / num_models
        
        # For each model, compute average wager across examples in each dataset
        for model_idx in range(num_models):
            avg_wagers_per_dataset = []
            
            for dataset_name in dataset_names:
                if dataset_name in results_dict and "wagers_history" in results_dict[dataset_name]:
                    wagers_history = results_dict[dataset_name]["wagers_history"]
                    avg_wager = np.mean(wagers_history[:, model_idx])
                else:
                    avg_wager = 0.0
                avg_wagers_per_dataset.append(avg_wager)
            
            ax.bar(x + model_idx * width, avg_wagers_per_dataset, width, label=model_names[model_idx], alpha=0.8)
        
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        ax.set_title(f"Average Wagers by Dataset ({eval_type.capitalize()})", fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels(dataset_names, rotation=20, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.checkpoint_dir / f"{eval_type}_average_wagers_by_dataset.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved {eval_type} average wagers by dataset plot to {save_path}")
        
        # Log to wandb
        if self.wandb_logger:
            import wandb
            try:
                if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                    self.wandb_logger.run.log(
                        {f"wagers_plot/{eval_type}/average_by_dataset": wandb.Image(str(save_path))},
                        step=self._global_wandb_step,
                    )
                else:
                    self.wandb_logger.log(
                        {f"wagers_plot/{eval_type}/average_by_dataset": wandb.Image(str(save_path))},
                        step=self._global_wandb_step,
                    )
            except Exception as e:
                raise Exception(f"Error logging plot to wandb: {e}")
        
        plt.close()
    
    def _plot_average_wagers_across_datasets(
        self,
        all_results: Dict[str, Any],
        eval_type: str = "test",
    ):
        """
        Plot average wagers across multiple test/OOD datasets.
        
        Args:
            all_results: Dictionary with results from all datasets, keyed by dataset name
            eval_type: Either "test", "ood", or "test_and_ood" for logging prefix and filtering
        """

        if self.checkpoint_dir is None or not all_results:
            log.warning(f"SKIPPING PLOT: checkpoint_dir={self.checkpoint_dir}, all_results={bool(all_results)}")
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from typing import List
        
        # Collect wagers history and dataset names from all results
        dataset_names = []
        all_wagers_list = []
        
        for dataset_name, result in all_results.items():
            # Filter based on eval_type
            is_ood = dataset_name.startswith("ood_")
            if eval_type == "test" and is_ood:
                continue
            elif eval_type == "ood" and not is_ood:
                continue
            # For "test_and_ood", include all datasets (no filtering)
            
            if "wagers_history" in result:
                wagers = result["wagers_history"]  # [num_examples, num_models]
                all_wagers_list.append(wagers)
                # Clean up dataset name for display
                display_name = dataset_name.replace("ood_", "")
                if is_ood and eval_type == "test_and_ood":
                    display_name = f"[OOD] {display_name}"
                dataset_names.append(display_name)
        
        if not all_wagers_list:
            log.warning(f"No wagers history found for {eval_type} evaluation")
            return
        
        # Compute average wagers per dataset
        num_datasets = len(all_wagers_list)
        num_models = all_wagers_list[0].shape[1]
        
        # Get model names
        model_names: List[str] = []
        if isinstance(self.metadata, dict) and "model_names" in self.metadata:
            raw_names = self.metadata["model_names"]
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
        
        # Plot: Average wagers per dataset
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        x = np.arange(num_datasets)
        width = 0.8 / num_models
        
        for i in range(num_models):
            avg_wagers = []
            for dataset_idx in range(num_datasets):
                wagers = all_wagers_list[dataset_idx]
                avg_wager = np.mean(wagers[:, i])
                avg_wagers.append(avg_wager)
            
            ax.bar(x + i * width, avg_wagers, width, label=model_names[i], alpha=0.8)
        
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        
        # Set title based on eval_type
        if eval_type == "test":
            title = "Average Wagers by Dataset (Test)"
        elif eval_type == "ood":
            title = "Average Wagers by Dataset (OOD)"
        else:  # test_and_ood
            title = "Average Wagers by Dataset (Test + OOD)"
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels(dataset_names, rotation=20, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.checkpoint_dir / f"average_wagers_by_dataset_{eval_type}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.debug(f"Saved wagers plot ({eval_type}) to {save_path}")
        
        if self.wandb_logger:
            import wandb
            plot_image_primary = wandb.Image(str(save_path))
            plot_image_alias = wandb.Image(str(save_path))
            self._log_wandb_plot(
                {
                    f"wagers_plot/{eval_type}/average_by_dataset": plot_image_primary,
                    f"wagers_plot/average_by_dataset/{eval_type}": plot_image_alias,
                }
            )
        
        plt.close(fig)
        
        plt.close()
    
    def evaluate_multiple(
        self,
        test_datasets: List[Tuple[Dataset, str]],
        ood_datasets: Optional[List[Tuple[Dataset, str]]] = None,
        ood_dataset: Optional[Tuple[Dataset, str]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate on multiple test datasets and optionally OOD datasets.
        
        Args:
            test_datasets: List of (dataset, name) tuples for test splits
            ood_datasets: Optional list of (dataset, name) tuples for OOD evaluation
            ood_dataset: Backward-compatible optional single (dataset, name) tuple for OOD evaluation
            resume: If True, attempt to resume from checkpoint if available (DISABLED - always evaluates from scratch)
            
        Returns:
            Dictionary with evaluation results for all datasets, including a combined analytics_df
        """
        all_results = {}
        completed_datasets = []
        all_analytics_dfs = []
        ood_datasets_to_eval: List[Tuple[Dataset, str]] = []
        if ood_datasets:
            ood_datasets_to_eval.extend(ood_datasets)
        if ood_dataset is not None:
            if not ood_datasets_to_eval:
                ood_datasets_to_eval.append(ood_dataset)
            else:
                log.warning(
                    "Both ood_datasets and ood_dataset were provided; evaluating all entries from ood_datasets."
                )
        
        # Evaluate on test splits
        for dataset, name in test_datasets:
            log.info(f"Evaluating test dataset: {name}")
            results = self.evaluate(dataset, name)
            all_results[name] = results
            completed_datasets.append(name)
            
            # Collect analytics dataframe
            if "analytics_df" in results:
                all_analytics_dfs.append(results["analytics_df"])
            
            # Save checkpoint after each dataset
            # self._save_checkpoint(all_results, completed_datasets)
        
        # Evaluate on OOD datasets if provided
        for dataset, name in ood_datasets_to_eval:
            ood_name = f"ood_{name}"
            
            log.info(f"Evaluating OOD dataset: {name} -> {ood_name}")
            log.info(f"Wandb logger available: {self.wandb_logger is not None}")
            
            # Pass ood_name to evaluate() so it gets the correct "ood" prefix in wandb logging
            results = self.evaluate(dataset, ood_name)
            all_results[ood_name] = results
            completed_datasets.append(ood_name)
            
            # Collect analytics dataframe
            if "analytics_df" in results:
                all_analytics_dfs.append(results["analytics_df"])
            
            # Save checkpoint after each OOD evaluation
            # self._save_checkpoint(all_results, completed_datasets)
        
        # Combine all analytics dataframes and save
        if all_analytics_dfs and self.checkpoint_dir:
            combined_analytics = pd.concat(all_analytics_dfs, ignore_index=True)
            combined_path = self.checkpoint_dir / "analytics_all.csv"
            combined_analytics.to_csv(combined_path, index=False)
            log.debug(f"Saved combined analytics dataframe to {combined_path}")
            all_results["analytics_df"] = combined_analytics
        
        # Plot average wagers across all test datasets
        log.info("=== GENERATING PLOTS ===")
        log.debug(f"All results keys: {list(all_results.keys())}")
        
        log.info("Generating test datasets plot...")
        self._plot_average_wagers_across_datasets(all_results, "test")
        
        # Plot average wagers across OOD datasets if applicable
        if ood_datasets_to_eval:
            # Create filtered results dict with only OOD datasets
            ood_results = {k: v for k, v in all_results.items() if k.startswith("ood_")}
            if ood_results:
                log.info("Generating OOD datasets plot...")
                self._plot_average_wagers_across_datasets(ood_results, "ood")
        
        # Plot average wagers across both test and OOD datasets combined
        log.info("Generating combined test+OOD plot...")
        self._plot_average_wagers_across_datasets(all_results, "test_and_ood")
        log.info("=== PLOTS GENERATION COMPLETE ===")
        
        return all_results

