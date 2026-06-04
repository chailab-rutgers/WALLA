"""
Analytics helper for wagering training and evaluation.

Provides utilities for creating analytics dataframes and aggregating results
across multiple runs with the same settings.
"""

import logging
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

log = logging.getLogger("wagering")


class WageringAnalytics:
    """
    Helper class for creating and aggregating analytics dataframes for wagering experiments.
    
    Supports both training and evaluation analytics, with automatic aggregation
    across multiple runs that share the same settings.
    """
    
    @staticmethod
    def create_training_analytics(
        wagering_method: Any,
        aggregation_function: Any,
        models: List[Any],
        datasets: List[Any],
        shuffle_data: bool,
        shuffle_seed: int,
        early_stopping_patience: int,
        save_every: int,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_dir: Optional[Path] = None,
        dataset_size: Optional[int] = None,
        early_stopping_criterion: str = "validation",
        use_brier_d_regret_for_early_stopping: bool = False,
        use_min_kl_for_early_stopping: bool = False,
    ) -> pd.DataFrame:
        """
        Create analytics dataframe for training results.
        
        Args:
            wagering_method: WageringMethod instance
            aggregation_function: AggregationFunction instance
            models: List of models
            datasets: List of training datasets
            shuffle_data: Whether data was shuffled
            shuffle_seed: Seed for data shuffling
            early_stopping_patience: Early stopping patience
            early_stopping_criterion: Early stopping strategy name
            use_brier_d_regret_for_early_stopping: Whether Brier dynamic regret is used
                as the monitored early stopping metric.
            use_min_kl_for_early_stopping: Whether KL(gold || pred) is used as the monitored
                early stopping metric (only valid when soft probabilistic labels exist).
            save_every: Save frequency
            results: Training results dictionary
            metadata: Optional metadata dict with model_names, dataset_names
            checkpoint_dir: Optional checkpoint directory path
            dataset_size: Optional dataset size (total number of training examples). Used to distinguish
                         different settings - runs with different sizes will not be aggregated together.
            
        Returns:
            DataFrame with settings, settings_hash, and training results
        """
        row = {}
        
        # Run identifier
        row["run_timestamp"] = datetime.now().isoformat()
        
        # Wagering method hyperparameters
        if hasattr(wagering_method, 'hidden_dim'):
            row["wagering_hidden_dim"] = wagering_method.hidden_dim
        if hasattr(wagering_method, 'common_hidden_dim'):  # For decentralized wagers
            row["wagering_hidden_dim"] = wagering_method.common_hidden_dim
        if hasattr(wagering_method, 'hidden_layers'):
            row["wagering_hidden_layers"] = str(wagering_method.hidden_layers)
        if hasattr(wagering_method, 'hidden_state_layers'):
            row["wagering_hidden_state_layers"] = str(wagering_method.hidden_state_layers)
        if hasattr(wagering_method, 'learning_rate'):
            row["wagering_learning_rate"] = wagering_method.learning_rate
        if hasattr(wagering_method, 'temperature'):
            row["wagering_temperature"] = wagering_method.temperature
        if hasattr(wagering_method, 'grad_clip_norm'):
            row["wagering_grad_clip_norm"] = wagering_method.grad_clip_norm
        if hasattr(wagering_method, 'normalize_hidden_states'):
            row["wagering_normalize_hidden_states"] = wagering_method.normalize_hidden_states
        if hasattr(wagering_method, 'device_str'):
            # Normalize device string: cuda:0 -> cuda, cuda:1 -> cuda, etc.
            # This ensures runs with different GPU IDs but same device type are aggregated
            device_str = wagering_method.device_str
            if device_str.startswith("cuda:"):
                device_str = "cuda"
            row["wagering_device"] = device_str
        row["wagering_method"] = type(wagering_method).__name__
        
        # Aggregation method
        row["aggregation_method"] = type(aggregation_function).__name__
        
        # Training hyperparameters
        row["num_models"] = len(models)
        row["num_datasets"] = len(datasets)
        row["shuffle_data"] = shuffle_data
        row["shuffle_seed"] = shuffle_seed
        row["early_stopping_patience"] = early_stopping_patience
        row["early_stopping_criterion"] = early_stopping_criterion
        row["use_brier_d_regret_for_early_stopping"] = bool(use_brier_d_regret_for_early_stopping)
        row["use_min_kl_for_early_stopping"] = bool(use_min_kl_for_early_stopping)
        row["save_every"] = save_every
        
        # Dataset size (included in settings_hash to distinguish different settings)
        if dataset_size is not None:
            row["dataset_size"] = dataset_size
        
        # Dataset information
        if metadata:
            if "dataset_names" in metadata:
                row["training_datasets"] = ",".join(sorted(metadata["dataset_names"]))
            if "model_names" in metadata:
                row["models"] = ",".join(sorted(metadata["model_names"]))
        
        # Create settings hash (excludes results, checkpoint_path, shuffle_seed, and wagering_device)
        # shuffle_seed is excluded so runs with different seeds can be aggregated together
        # wagering_device is excluded so runs with different devices can be aggregated together
        settings_dict = {k: v for k, v in row.items() if k not in ["run_timestamp", "shuffle_seed", "wagering_device"]}
        settings_str = json.dumps(settings_dict, sort_keys=True, default=str)
        row["settings_hash"] = hashlib.md5(settings_str.encode()).hexdigest()[:16]
        
        # Results (these are NOT included in settings_hash)
        row["final_accuracy"] = results.get("final_accuracy")
        row["final_nll"] = results.get("final_nll")
        row["final_ece"] = results.get("final_ece") if results.get("final_ece") is not None and not np.isnan(results.get("final_ece", np.nan)) else None
        row["final_auc"] = results.get("final_auc") if results.get("final_auc") is not None and not np.isnan(results.get("final_auc", np.nan)) else None
        row["final_d_regret"] = results.get("final_d_regret") if results.get("final_d_regret") is not None and not np.isnan(results.get("final_d_regret", np.nan)) else None
        row["final_brier_d_regret"] = results.get("final_brier_d_regret") if results.get("final_brier_d_regret") is not None and not np.isnan(results.get("final_brier_d_regret", np.nan)) else None
        row["final_meta_acc"] = results.get("final_meta_acc") if results.get("final_meta_acc") is not None and not np.isnan(results.get("final_meta_acc", np.nan)) else None
        row["final_meta_nll"] = results.get("final_meta_nll") if results.get("final_meta_nll") is not None and not np.isnan(results.get("final_meta_nll", np.nan)) else None
        row["final_meta_auc"] = results.get("final_meta_auc") if results.get("final_meta_auc") is not None and not np.isnan(results.get("final_meta_auc", np.nan)) else None
        
        # Checkpoint path (not included in settings_hash)
        if checkpoint_dir:
            row["checkpoint_path"] = str(checkpoint_dir)
        
        # Mark as training results
        row["result_type"] = "training"
        
        return pd.DataFrame([row])
    
    @staticmethod
    def create_evaluation_analytics(
        wagering_method: Any,
        aggregation_function: Any,
        models: List[Any],
        evaluation_dataset_name: str,
        training_datasets: Optional[List[str]] = None,
        results: Dict[str, Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        seed: Optional[int] = None,
        dataset_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Create analytics dataframe for evaluation results.
        
        Args:
            wagering_method: WageringMethod instance
            aggregation_function: AggregationFunction instance
            models: List of models
            evaluation_dataset_name: Name of the evaluation dataset
            training_datasets: Optional list of training dataset names
            results: Evaluation results dictionary (should contain accuracy, nll, brier, auc, ece)
            metadata: Optional metadata dict with model_names
            checkpoint_path: Optional path to the training checkpoint used
            seed: Optional random seed used for this run
            dataset_size: Optional dataset size (number of examples). Used to distinguish
                         different settings - runs with different sizes will not be aggregated together.
            
        Returns:
            DataFrame with settings, settings_hash, and evaluation results
        """
        row = {}
        
        # Run identifier
        row["run_timestamp"] = datetime.now().isoformat()
        
        # Wagering method hyperparameters
        if hasattr(wagering_method, 'hidden_dim'):
            row["wagering_hidden_dim"] = wagering_method.hidden_dim
        if hasattr(wagering_method, 'common_hidden_dim'):  # For decentralized wagers
            row["wagering_hidden_dim"] = wagering_method.common_hidden_dim
        if hasattr(wagering_method, 'hidden_layers'):
            row["wagering_hidden_layers"] = str(wagering_method.hidden_layers)
        if hasattr(wagering_method, 'hidden_state_layers'):
            row["wagering_hidden_state_layers"] = str(wagering_method.hidden_state_layers)
        if hasattr(wagering_method, 'learning_rate'):
            row["wagering_learning_rate"] = wagering_method.learning_rate
        if hasattr(wagering_method, 'temperature'):
            row["wagering_temperature"] = wagering_method.temperature
        if hasattr(wagering_method, 'grad_clip_norm'):
            row["wagering_grad_clip_norm"] = wagering_method.grad_clip_norm
        if hasattr(wagering_method, 'normalize_hidden_states'):
            row["wagering_normalize_hidden_states"] = wagering_method.normalize_hidden_states
        if hasattr(wagering_method, 'device_str'):
            # Normalize device string: cuda:0 -> cuda, cuda:1 -> cuda, etc.
            # This ensures runs with different GPU IDs but same device type are aggregated
            device_str = wagering_method.device_str
            if device_str.startswith("cuda:"):
                device_str = "cuda"
            row["wagering_device"] = device_str
        row["wagering_method"] = type(wagering_method).__name__
        
        # Aggregation method
        row["aggregation_method"] = type(aggregation_function).__name__
        
        # Model information
        row["num_models"] = len(models)
        if metadata and "model_names" in metadata:
            row["models"] = ",".join(sorted(metadata["model_names"]))
        
        # Dataset information
        row["evaluation_dataset"] = evaluation_dataset_name
        if training_datasets:
            row["training_datasets"] = ",".join(sorted(training_datasets))
        
        # Dataset size (included in settings_hash to distinguish different settings)
        if dataset_size is not None:
            row["dataset_size"] = dataset_size
        
        # Seed information
        if seed is not None:
            row["seed"] = seed
        
        # Checkpoint path (for tracking which training checkpoint was used)
        if checkpoint_path:
            row["checkpoint_path"] = str(checkpoint_path)
        
        # Create settings hash (excludes results, seed, and wagering_device)
        # seed is excluded so runs with different seeds can be aggregated together
        # wagering_device is excluded so runs with different devices can be aggregated together
        settings_dict = {k: v for k, v in row.items() if k not in ["run_timestamp", "seed", "wagering_device"]}
        settings_str = json.dumps(settings_dict, sort_keys=True, default=str)
        row["settings_hash"] = hashlib.md5(settings_str.encode()).hexdigest()[:16]
        
        # Results (these are NOT included in settings_hash)
        if results:
            row["accuracy"] = results.get("accuracy")
            row["nll"] = results.get("nll")
            row["brier"] = results.get("brier") if results.get("brier") is not None and not np.isnan(results.get("brier", np.nan)) else None
            row["bernoulli_kl"] = (
                results.get("bernoulli_kl")
                if results.get("bernoulli_kl") is not None
                and not np.isnan(results.get("bernoulli_kl", np.nan))
                else None
            )
            row["bernoulli_tv"] = (
                results.get("bernoulli_tv")
                if results.get("bernoulli_tv") is not None
                and not np.isnan(results.get("bernoulli_tv", np.nan))
                else None
            )
            row["auc"] = results.get("auc") if results.get("auc") is not None and not np.isnan(results.get("auc", np.nan)) else None
            row["ece"] = results.get("ece") if results.get("ece") is not None and not np.isnan(results.get("ece", np.nan)) else None
            row["inverse_hhi"] = results.get("inverse_hhi") if results.get("inverse_hhi") is not None and not np.isnan(results.get("inverse_hhi", np.nan)) else None
            row["avg_inference_time_per_batch_s"] = (
                results.get("avg_inference_time_per_batch_s")
                if results.get("avg_inference_time_per_batch_s") is not None
                and not np.isnan(results.get("avg_inference_time_per_batch_s", np.nan))
                else None
            )
            row["d_regret"] = results.get("d_regret") if results.get("d_regret") is not None and not np.isnan(results.get("d_regret", np.nan)) else None
            row["brier_d_regret"] = results.get("brier_d_regret") if results.get("brier_d_regret") is not None and not np.isnan(results.get("brier_d_regret", np.nan)) else None
            row["meta_acc"] = results.get("meta_acc") if results.get("meta_acc") is not None and not np.isnan(results.get("meta_acc", np.nan)) else None
            row["meta_nll"] = results.get("meta_nll") if results.get("meta_nll") is not None and not np.isnan(results.get("meta_nll", np.nan)) else None
            row["meta_auc"] = results.get("meta_auc") if results.get("meta_auc") is not None and not np.isnan(results.get("meta_auc", np.nan)) else None
            row["kendall_tau"] = results.get("kendall_tau") if results.get("kendall_tau") is not None and not np.isnan(results.get("kendall_tau", np.nan)) else None
            row["best_model_mrr"] = results.get("best_model_mrr") if results.get("best_model_mrr") is not None and not np.isnan(results.get("best_model_mrr", np.nan)) else None
            row["brier_best_wager_prob_mean"] = results.get("brier_best_wager_prob_mean")
            row["brier_best_wager_prob_var"] = results.get("brier_best_wager_prob_var")
            means = results.get("wager_prob_mean_per_model") or []
            vars_ = results.get("wager_prob_var_per_model") or []
            for i, val in enumerate(means):
                row[f"wager_prob_mean_model_{i}"] = val
            for i, val in enumerate(vars_):
                row[f"wager_prob_var_model_{i}"] = val
            row["num_examples"] = results.get("num_examples")
        
        # Mark as evaluation results
        row["result_type"] = "evaluation"
        
        return pd.DataFrame([row])
    
    @staticmethod
    def aggregate_results_by_settings(
        analytics_dfs: List[pd.DataFrame],
        result_columns: Optional[List[str]] = None,
        agg_functions: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate results from multiple runs that share the same settings.
        
        Args:
            analytics_dfs: List of analytics DataFrames
            result_columns: List of column names that contain results (default: auto-detect)
            agg_functions: Dict mapping result column names to aggregation functions
                          ('mean', 'std', 'count', 'min', 'max'). Default: mean and std for all.
        
        Returns:
            DataFrame with one row per unique settings_hash, with aggregated results.
            Settings columns are preserved, and result columns are aggregated.
        
        Example:
            # Load multiple analytics CSV files
            dfs = [pd.read_csv(f) for f in analytics_files]
            aggregated = WageringAnalytics.aggregate_results_by_settings(dfs)
            # Group by settings_hash and see mean/std of results
        """
        if not analytics_dfs:
            return pd.DataFrame()
        
        # Concatenate all DataFrames
        combined_df = pd.concat(analytics_dfs, ignore_index=True)
        
            # Auto-detect result columns if not provided
        if result_columns is None:
            # Result columns are those that start with "final_" or are numeric and not settings
            settings_columns = [
                'settings_hash', 'run_timestamp', 'checkpoint_path', 'result_type',
                'wagering_method', 'aggregation_method', 'num_models', 'num_datasets',
                'shuffle_data', 'shuffle_seed', 'early_stopping_patience', 'save_every',
                'datasets', 'models', 'training_datasets', 'evaluation_dataset',
                'wagering_hidden_dim', 'wagering_hidden_layers',
                'wagering_hidden_state_layers',
                'wagering_learning_rate', 'wagering_temperature', 'wagering_grad_clip_norm',
                'wagering_normalize_hidden_states', 'wagering_device', 'seed', 'dataset_size'
            ]
            result_columns = [
                col for col in combined_df.columns
                if (col.startswith('final_') or col in [
                    'accuracy', 'nll', 'brier', 'bernoulli_kl', 'bernoulli_tv', 'auc', 'ece', 'num_examples',
                    'inverse_hhi', 'avg_inference_time_per_batch_s',
                    'd_regret', 'brier_d_regret', 'meta_acc', 'meta_nll', 'meta_auc',
                    'kendall_tau', 'best_model_mrr',
                    'brier_best_wager_prob_mean', 'brier_best_wager_prob_var',
                ] or col.startswith('wager_prob_mean_model_') or col.startswith('wager_prob_var_model_'))
                or (col not in settings_columns and pd.api.types.is_numeric_dtype(combined_df[col]))
            ]
        
        # Default aggregation: compute both mean and std for all result columns
        if agg_functions is None:
            agg_functions = {}
        
        # Identify settings columns (all columns except results and metadata)
        settings_columns = [
            col for col in combined_df.columns
            if col not in result_columns and col not in ['run_timestamp', 'checkpoint_path']
        ]
        
        # Group by settings_hash and aggregate
        # For result columns, compute both mean and std by default (unless specified otherwise)
        agg_dict = {}
        
        for col in result_columns:
            if col in agg_functions:
                func = agg_functions[col]
                agg_dict[col] = func
            else:
                # Default: compute both mean and std for better analytics
                agg_dict[col] = ['mean', 'std']
        
        # For settings columns, take the first value (they should be the same within a group)
        # Note: settings_hash is used for grouping, so we don't need to aggregate it
        for col in settings_columns:
            if col != 'settings_hash':  # Don't aggregate settings_hash, it's the group key
                agg_dict[col] = 'first'
        
        # Group by settings_hash
        grouped = combined_df.groupby('settings_hash', as_index=False).agg(agg_dict)
        
        # Flatten multi-level column names from mean/std aggregation
        if isinstance(grouped.columns, pd.MultiIndex):
            new_columns = []
            for col in grouped.columns:
                if isinstance(col, tuple):
                    if col[1] == '':  # Empty string means it's the groupby key (settings_hash)
                        new_columns.append(col[0])  # Keep original name
                    elif col[1] == 'mean':
                        new_columns.append(col[0])  # Keep original name for mean
                    elif col[1] == 'first':
                        new_columns.append(col[0])  # Keep original name for first (settings columns)
                    else:
                        new_columns.append(f"{col[0]}_{col[1]}")  # Add suffix for std, min, max, etc.
                else:
                    new_columns.append(col)
            grouped.columns = new_columns
        
        # Add count of runs per settings
        run_counts = combined_df.groupby('settings_hash').size().reset_index(name='num_runs')
        grouped = grouped.merge(run_counts, on='settings_hash', how='left')
        
        # Remove wagering_device from aggregated output (device shouldn't matter for aggregation)
        if 'wagering_device' in grouped.columns:
            grouped = grouped.drop(columns=['wagering_device'])
        
        return grouped
    
    @staticmethod
    def get_settings_columns(analytics_df: pd.DataFrame) -> List[str]:
        """
        Get the list of settings columns (columns used for grouping runs).
        
        Args:
            analytics_df: Analytics DataFrame
            
        Returns:
            List of column names that represent settings/hyperparameters
        """
        result_columns = [
            col for col in analytics_df.columns
            if col.startswith('final_') or col in ['accuracy', 'nll', 'brier', 'auc', 'ece', 'num_examples', 'd_regret', 'meta_acc', 'meta_nll', 'meta_auc']
        ]
        settings_columns = [
            col for col in analytics_df.columns
            if col not in result_columns and col not in ['run_timestamp', 'checkpoint_path']
        ]
        return settings_columns
    
    @staticmethod
    def get_result_columns(analytics_df: pd.DataFrame) -> List[str]:
        """
        Get the list of result columns (columns that contain metrics to aggregate).
        
        Args:
            analytics_df: Analytics DataFrame
            
        Returns:
            List of column names that represent results/metrics
        """
        result_columns = [
            col for col in analytics_df.columns
            if col.startswith('final_') or col in ['accuracy', 'nll', 'brier', 'auc', 'ece', 'num_examples', 'd_regret', 'meta_acc', 'meta_nll', 'meta_auc']
            or (pd.api.types.is_numeric_dtype(analytics_df[col]) 
                and col not in ['num_models', 'num_datasets', 'shuffle_seed', 'early_stopping_patience', 
                               'save_every', 'seed', 'num_examples'])
        ]
        return result_columns
    
    @staticmethod
    def load_and_aggregate_analytics(
        analytics_paths: List[str],
        result_columns: Optional[List[str]] = None,
        agg_functions: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Load multiple analytics CSV files and aggregate results by settings.
        
        Args:
            analytics_paths: List of paths to analytics.csv files
            result_columns: List of column names that contain results (default: auto-detect)
            agg_functions: Dict mapping result column names to aggregation functions
        
        Returns:
            Aggregated DataFrame with one row per unique settings_hash
        
        Example:
            # Find all analytics.csv files in a directory
            import glob
            paths = glob.glob("workdir/**/analytics.csv", recursive=True)
            aggregated = WageringAnalytics.load_and_aggregate_analytics(paths)
        """
        analytics_dfs = []
        for path in analytics_paths:
            try:
                df = pd.read_csv(path)
                analytics_dfs.append(df)
            except Exception as e:
                log.warning(f"Could not load analytics from {path}: {e}")
        
        if not analytics_dfs:
            log.warning("No analytics DataFrames loaded")
            return pd.DataFrame()
        
        return WageringAnalytics.aggregate_results_by_settings(
            analytics_dfs, result_columns, agg_functions
        )

