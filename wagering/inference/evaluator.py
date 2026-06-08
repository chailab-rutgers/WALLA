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
# Local wagering imports
from wagering.methods.base import WageringMethod
from wagering.training.analytics import WageringAnalytics
from wagering.training.trainer import WageringTrainer
from wagering.utils.wagering_metrics import compute_evaluation_metrics
from wagering.utils.wagering_plots import WageringPlotter
from wagering.aggregation.base import AggregationFunction
from wagering.utils.cache_manager import (
    collect_stacked_model_artifacts,
    compute_all_prompt_perplexities,
)
from wagering.utils.prompt_manager import (
    get_mixed_context_dataset_type,
    get_model_specific_prompts,
)
from wagering.utils.wandb_logging import (
    WandbStepTracker,
    log_eval_batch,
    log_eval_dataset_plot,
    log_eval_final,
    log_eval_multi_dataset_plot,
    resolve_initial_step,
)

log = logging.getLogger("wagering")


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
        self.method_requires_model_perplexities = bool(
            getattr(self.wagering_method, "requires_model_perplexities", False)
        )
        self._model_configs_for_sequential_perplexity = model_configs_for_sequential_perplexity
        self._perplexity_load_cache_kwargs = perplexity_load_cache_kwargs or {}

        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.wagering_method.eval_mode()

        self._wandb = WandbStepTracker(
            wandb_logger,
            initial_step=resolve_initial_step(wandb_logger, wandb_starting_step),
        )
        if self._wandb.active():
            log.info("Initialized wandb step counter to %s", self._wandb.step)

        self._plotter = WageringPlotter(
            checkpoint_dir=self.checkpoint_dir,
            metadata=self.metadata,
            dataset=None,
            models=self.models,
        )

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
        needs_hidden_states = (
            bool(getattr(self.wagering_method, "requires_hidden_states", True))
            or self.logit_calibrator is not None
        )
        
        loaded_here_by_index: Dict[int, bool] = {}

        def _unload_model_if_loaded(wb_model: WhiteboxModel, loaded_here: bool) -> None:
            if not loaded_here:
                return
            import gc
            import torch

            del wb_model.model
            del wb_model.tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def _resolve_model_on_miss(
            model_index: int, model: WhiteboxModel | str
        ) -> WhiteboxModel:
            wb_model, loaded_here = self._maybe_load_model_for_collection(model, model_index)
            loaded_here_by_index[model_index] = loaded_here
            return wb_model

        artifacts = collect_stacked_model_artifacts(
            self.models,
            dataset,
            self.option_tokens,
            collect_hidden_states=needs_hidden_states,
            resolve_model_on_cache_miss=_resolve_model_on_miss,
            release_model_after_collect=lambda idx, wb: _unload_model_if_loaded(
                wb, loaded_here_by_index.get(idx, False)
            ),
        )
        all_model_logits = artifacts.logits
        all_model_hidden_states = artifacts.combined_hidden_states()
        if needs_hidden_states and isinstance(all_model_hidden_states, np.ndarray):
            log.info("Stacked hidden states: shape %s", all_model_hidden_states.shape)
        elif needs_hidden_states and isinstance(all_model_hidden_states, list):
            hidden_dims = [hs.shape[-1] for hs in all_model_hidden_states]
            log.info(
                "Models have different hidden dimensions: %s. Keeping as list.",
                hidden_dims,
            )

        labels = artifacts.labels
        if labels is None:
            raise RuntimeError("Evaluation cache collection produced no labels")
        labels = np.asarray(labels, dtype=np.int32)

        if self.logit_calibrator is not None:
            if all_model_hidden_states is None:
                raise RuntimeError("Temperature calibration requires cached hidden states during evaluation")

            calibration_hidden_states = all_model_hidden_states

            apply_kwargs = {}
            if artifacts.context_assignments is not None:
                apply_kwargs["context_model_index_by_example"] = artifacts.context_assignments
            all_model_logits = self.logit_calibrator.apply_to_stacked_logits(
                all_model_logits,
                calibration_hidden_states,
                **apply_kwargs,
            )
            log.info("Applied frozen temperature scaling to cached evaluation logits")

        num_examples = all_model_logits.shape[1]
        model_perplexities = None
        if self.method_requires_model_perplexities:
            model_perplexities = compute_all_prompt_perplexities(
                self.models,
                dataset,
                model_configs_for_sequential=self._model_configs_for_sequential_perplexity,
                load_cache_kwargs=self._perplexity_load_cache_kwargs,
                group_identical_model_configs=True,
            )
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

            # Get questions for batch (for wagering methods that need them)
            batch_questions = dataset.x[batch_start:batch_end]  # List of question strings
            batch_questions_per_model = None
            if bool(getattr(self.wagering_method, "expects_per_model_router_prompts", False)):
                if get_mixed_context_dataset_type(dataset) is not None:
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
                total_payout_history.extend(np.asarray(batch_total_payout).tolist())
            if batch_sigmoid_wagers is not None:
                sigmoid_wagers_history.extend(np.asarray(batch_sigmoid_wagers).tolist())
            
            # Compute batch metrics using vectorized operations
            batch_correct = (batch_predictions == batch_labels)
            batch_nll = -np.log(batch_aggregated_probs[np.arange(batch_size_actual), batch_labels] + 1e-10)
            
            # Update running metrics
            running_correct += int(np.sum(batch_correct))
            running_nll_sum += np.sum(batch_nll)
            running_accuracy = running_correct / (batch_end)
            running_nll = running_nll_sum / batch_end
            
            log_eval_batch(
                self._wandb,
                dataset_name=dataset_name,
                batch_correct=batch_correct,
                batch_nll=batch_nll,
                batch_wagers=batch_wagers,
                running_accuracy=running_accuracy,
                running_nll=running_nll,
                inference_time_s=float(inference_times_s[-1]),
            )
        
        # Convert to arrays
        all_predictions = np.array(all_predictions, dtype=np.int32)
        all_aggregated_probs = np.stack(all_aggregated_probs, axis=0)
        wagers_history = np.stack(wagers_history, axis=0)  # [num_examples, num_models]
        total_payout_history = (
            np.asarray(total_payout_history, dtype=np.float32)
            if total_payout_history
            else None
        )
        sigmoid_wagers_history = (
            np.asarray(sigmoid_wagers_history, dtype=np.float32)
            if sigmoid_wagers_history
            else None
        )

        metrics = compute_evaluation_metrics(
            predictions=all_predictions,
            aggregated_probs=all_aggregated_probs,
            labels=labels,
            model_logits_stacked=all_model_logits,
            wagers_history=wagers_history,
            dataset=dataset,
            option_tokens=self.option_tokens,
            inference_times_s=inference_times_s,
            sigmoid_wagers_history=sigmoid_wagers_history,
            total_payout_history=total_payout_history,
        )

        results = {
            "dataset_name": dataset_name,
            "num_examples": num_examples,
            "predictions": all_predictions,
            "aggregated_probs": all_aggregated_probs,
            "labels": labels,
            "wagers_history": wagers_history,
            **metrics,
        }

        subset_metrics = results["subset_any_model_wrong"]
        subset_n = subset_metrics["num_examples"]
        if subset_n > 0:
            acc_s = subset_metrics.get("accuracy")
            nll_s = subset_metrics.get("nll")
            ece_s = subset_metrics.get("ece")
            brier_dr_s = subset_metrics.get("brier_d_regret")
            sub_kl = subset_metrics.get("bernoulli_kl")
            sub_tv = subset_metrics.get("bernoulli_tv")
            log.info(
                "%s - Subset(any model wrong; n=%d) Accuracy: %s, NLL: %s, KL: %s, TV: %s, ECE: %s, BrierDRegret: %s",
                dataset_name,
                subset_n,
                f"{float(acc_s):.4f}" if acc_s is not None and np.isfinite(acc_s) else "N/A",
                f"{float(nll_s):.4f}" if nll_s is not None and np.isfinite(nll_s) else "N/A",
                f"{float(sub_kl):.4f}" if sub_kl is not None and np.isfinite(sub_kl) else "N/A",
                f"{float(sub_tv):.4f}" if sub_tv is not None and np.isfinite(sub_tv) else "N/A",
                f"{float(ece_s):.4f}" if ece_s is not None and np.isfinite(ece_s) else "N/A",
                f"{float(brier_dr_s):.4f}" if brier_dr_s is not None and np.isfinite(brier_dr_s) else "N/A",
            )
        else:
            log.info("%s - Subset(any model wrong) is empty (n=0); skipped subset metrics.", dataset_name)

        def _fmt_metric(value: Any) -> str:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "N/A"
            return f"{float(value):.4f}"

        log.info(
            "%s - Accuracy: %.4f, NLL: %.4f, Brier: %s, KL: %s, TV: %s, AUC: %s, ECE: %s, "
            "BrierDRegret: %s, KendallTau: %s, BestModelMRR: %s",
            dataset_name,
            results["accuracy"],
            results["nll"],
            _fmt_metric(results["brier"]),
            _fmt_metric(results["bernoulli_kl"]),
            _fmt_metric(results["bernoulli_tv"]),
            _fmt_metric(results["auc"]),
            _fmt_metric(results["ece"]),
            _fmt_metric(results["brier_d_regret"]),
            _fmt_metric(results["kendall_tau"]),
            _fmt_metric(results["best_model_mrr"]),
        )

        avg_wagers = np.mean(wagers_history, axis=0)
        wager_info = ", ".join([f"Model {i}: {wager:.4f}" for i, wager in enumerate(avg_wagers)])
        log.info(f"{dataset_name} - Average Wagers: {wager_info}")
        wager_prob_mean_per_model = results.get("wager_prob_mean_per_model")
        wager_prob_var_per_model = results.get("wager_prob_var_per_model")
        if wager_prob_mean_per_model is not None and wager_prob_var_per_model is not None:
            wp_parts = [
                f"Model {i}: mean={wager_prob_mean_per_model[i]:.4f}, var={wager_prob_var_per_model[i]:.4f}"
                for i in range(len(wager_prob_mean_per_model))
            ]
            log.info(f"{dataset_name} - Normalized wager prob (mean, var over examples): {', '.join(wp_parts)}")
        brier_best_wager_prob_mean = results.get("brier_best_wager_prob_mean")
        brier_best_wager_prob_var = results.get("brier_best_wager_prob_var")
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
                payout_path = self.checkpoint_dir / f"total_payout_history_{dataset_name}.npy"
                np.save(payout_path, total_payout_history)
                results["total_payout_history_path"] = str(payout_path)
                log.debug("Saved total payout history to %s", payout_path)

            if sigmoid_wagers_history is not None:
                sw_path = self.checkpoint_dir / f"sigmoid_wagers_history_{dataset_name}.npy"
                np.save(sw_path, sigmoid_wagers_history)
                results["sigmoid_wagers_history_path"] = str(sw_path)
                log.debug("Saved sigmoid wagers history to %s", sw_path)

        log_eval_final(
            self._wandb,
            dataset_name=dataset_name,
            inverse_hhi=results["inverse_hhi"],
            avg_inference_time_per_batch_s=results["avg_inference_time_per_batch_s"],
            accuracy=results["accuracy"],
            nll=results["nll"],
            brier=results["brier"],
            bernoulli_kl=results["bernoulli_kl"],
            bernoulli_tv=results["bernoulli_tv"],
            auc=results["auc"],
            ece=results["ece"],
            brier_d_regret=results["brier_d_regret"],
            kendall_tau=results["kendall_tau"],
            best_model_mrr=results["best_model_mrr"],
            brier_best_wager_prob_mean=results["brier_best_wager_prob_mean"],
            brier_best_wager_prob_var=results["brier_best_wager_prob_var"],
            wagers_history=wagers_history,
            wager_prob_mean_per_model=results["wager_prob_mean_per_model"],
            wager_prob_var_per_model=results["wager_prob_var_per_model"],
        )

        self._plotter.plot_eval_wagers(
            results,
            log_dataset_plot=lambda ds, key, path: log_eval_dataset_plot(
                self._wandb, ds, key, path
            ),
        )

        return results

    def _save_checkpoint(self, all_results: Dict[str, Any], completed_datasets: List[str]):
        """Save evaluation checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint_file = self.checkpoint_dir / "eval_checkpoint.pkl"
        checkpoint_data = {
            "results": all_results,
            "completed_datasets": completed_datasets,
            "global_wandb_step": self._wandb.step,
        }
        
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)
        log.info(f"Saved evaluation checkpoint to {checkpoint_file}")

    def evaluate_multiple(
        self,
        test_datasets: List[Tuple[Dataset, str]],
        ood_datasets: Optional[List[Tuple[Dataset, str]]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate on multiple test datasets and optionally OOD datasets.
        
        Args:
            test_datasets: List of (dataset, name) tuples for test splits
            ood_datasets: Optional list of (dataset, name) tuples for OOD evaluation
            resume: If True, attempt to resume from checkpoint if available (DISABLED - always evaluates from scratch)
            
        Returns:
            Dictionary with evaluation results for all datasets, including a combined analytics_df
        """
        all_results = {}
        completed_datasets = []
        all_analytics_dfs = []
        ood_datasets_to_eval: List[Tuple[Dataset, str]] = list(ood_datasets or [])
        
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
        self._plotter.plot_average_wagers_across_datasets(
            all_results,
            "test",
            log_multi_dataset_plot=lambda et, path: log_eval_multi_dataset_plot(
                self._wandb, et, path
            ),
        )

        if ood_datasets_to_eval:
            ood_results = {k: v for k, v in all_results.items() if k.startswith("ood_")}
            if ood_results:
                log.info("Generating OOD datasets plot...")
                self._plotter.plot_average_wagers_across_datasets(
                    ood_results,
                    "ood",
                    log_multi_dataset_plot=lambda et, path: log_eval_multi_dataset_plot(
                        self._wandb, et, path
                    ),
                )

        log.info("Generating combined test+OOD plot...")
        self._plotter.plot_average_wagers_across_datasets(
            all_results,
            "test_and_ood",
            log_multi_dataset_plot=lambda et, path: log_eval_multi_dataset_plot(
                self._wandb, et, path
            ),
        )
        log.info("=== PLOTS GENERATION COMPLETE ===")
        
        return all_results

