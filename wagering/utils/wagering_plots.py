"""
Matplotlib plotting helpers for wagering training and validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from wagering.core.dataset import Dataset

log = logging.getLogger("wagering")


def resolve_training_dataset_name(
    metadata: Dict[str, Any],
    dataset: Dataset,
) -> str:
    if isinstance(metadata, dict):
        for key in ("training_dataset", "dataset_name", "dataset"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    ds_name = (
        getattr(dataset, "name", None)
        or getattr(dataset, "dataset_name", None)
        or getattr(dataset, "path", None)
        or getattr(dataset, "cache_dataset_name", None)
    )
    return str(ds_name) if ds_name else "dataset"


def get_model_names_for_plot(
    metadata: Dict[str, Any],
    models: List[Any],
    num_models: int,
) -> List[str]:
    model_names: List[str] = []
    if isinstance(metadata, dict):
        for key in ("models", "model_names"):
            raw_names = metadata.get(key)
            if isinstance(raw_names, (list, tuple)):
                model_names = [str(name) for name in raw_names][:num_models]
                break

    if len(model_names) != num_models and models:
        inferred_names: List[str] = []
        for i, model in enumerate(models):
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


def get_validation_context_assignment_mask(
    dataset: Dataset,
    num_examples: int,
    num_models_total: int,
    local_indices: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Per-example context assignment mask for mixed-context datasets (visualization only)."""
    if local_indices is None:
        return None, None

    local_indices_arr = np.asarray(local_indices)
    if local_indices_arr.shape[0] != num_examples:
        log.debug(
            "Skipping context-aware masking due to shape mismatch: "
            f"num_examples={num_examples}, local_indices={local_indices_arr.shape}"
        )
        return None, None

    assignment_list = getattr(dataset, "pubmedqa_context_assignment_by_example", None)
    if not isinstance(assignment_list, list) or len(assignment_list) == 0:
        return None, None

    assignments = np.asarray(assignment_list, dtype=np.int32)
    if assignments.ndim != 1:
        return None, None

    assignment_mask = np.ones((num_examples, num_models_total), dtype=bool)
    row_local_indices = local_indices_arr
    valid_local_idx_mask = (row_local_indices >= 0) & (row_local_indices < assignments.shape[0])
    if not np.any(valid_local_idx_mask):
        return None, None

    mapped_rows = np.flatnonzero(valid_local_idx_mask)
    mapped_models = assignments[row_local_indices[valid_local_idx_mask]]
    valid_model_idx_mask = (mapped_models >= 0) & (mapped_models < num_models_total)
    if not np.any(valid_model_idx_mask):
        return None, None

    mapped_rows = mapped_rows[valid_model_idx_mask]
    mapped_models = mapped_models[valid_model_idx_mask]
    assignment_mask[mapped_rows, :] = False
    assignment_mask[mapped_rows, mapped_models] = True
    return assignment_mask, "pubmedqa"


class WageringPlotter:
    """Generate and save wagering diagnostic plots."""

    def __init__(
        self,
        *,
        checkpoint_dir: Optional[Path],
        metadata: Dict[str, Any],
        dataset: Optional[Dataset],
        models: List[Any],
        log_wandb_plot: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.metadata = metadata
        self.dataset = dataset
        self.models = models
        self.log_wandb_plot = log_wandb_plot

    def plot_eval_wagers(
        self,
        results: Dict[str, Any],
        *,
        log_dataset_plot: Optional[Callable[[str, str, Path], None]] = None,
    ) -> None:
        """Plot wagers over time and average wagers per model for one evaluation dataset."""
        if self.checkpoint_dir is None or "wagers_history" not in results:
            return

        wagers_history = results["wagers_history"]
        dataset_name = results["dataset_name"]
        num_examples, num_models = wagers_history.shape
        model_names = get_model_names_for_plot(self.metadata, self.models, num_models)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        time_steps = np.arange(1, num_examples + 1)
        for i in range(num_models):
            ax.plot(time_steps, wagers_history[:, i], label=model_names[i], alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Evaluation Step", fontsize=11)
        ax.set_ylabel("Wager (Weight)", fontsize=11)
        ax.set_title(f"Wagers Over Time - {dataset_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        plt.tight_layout()

        save_path = self.checkpoint_dir / f"wagers_over_time_{dataset_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.debug(f"Saved wagers plot to {save_path}")
        if log_dataset_plot is not None:
            log_dataset_plot(dataset_name, "wagers_plot", save_path)
        plt.close()

        self.plot_average_wagers_by_model(
            wagers_history,
            dataset_name,
            model_names,
            log_dataset_plot=log_dataset_plot,
        )

    def plot_average_wagers_by_model(
        self,
        wagers_history: np.ndarray,
        dataset_name: str,
        model_names: List[str],
        *,
        log_dataset_plot: Optional[Callable[[str, str, Path], None]] = None,
    ) -> None:
        if self.checkpoint_dir is None:
            return

        num_models = wagers_history.shape[1]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        avg_wagers = np.mean(wagers_history, axis=0)
        bars = ax.bar(range(num_models), avg_wagers, alpha=0.7, color="steelblue")
        for bar, wager in zip(bars, avg_wagers):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{wager:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        ax.set_title(f"Average Wagers by Model - {dataset_name}", fontsize=12, fontweight="bold")
        ax.set_xticks(range(num_models))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.05])
        plt.tight_layout()

        save_path = self.checkpoint_dir / f"average_wagers_{dataset_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.debug(f"Saved average wagers plot to {save_path}")
        if log_dataset_plot is not None:
            log_dataset_plot(dataset_name, "average_wagers_plot", save_path)
        plt.close()

    def plot_average_wagers_across_datasets(
        self,
        all_results: Dict[str, Dict[str, Any]],
        eval_type: str = "test",
        *,
        log_multi_dataset_plot: Optional[Callable[[str, Path], None]] = None,
    ) -> None:
        """Bar chart of average wagers per model, grouped by dataset."""
        if self.checkpoint_dir is None or not all_results:
            return

        dataset_names: List[str] = []
        all_wagers_list: List[np.ndarray] = []

        for dataset_name, result in all_results.items():
            is_ood = dataset_name.startswith("ood_")
            if eval_type == "test" and is_ood:
                continue
            if eval_type == "ood" and not is_ood:
                continue

            if "wagers_history" in result:
                all_wagers_list.append(result["wagers_history"])
                display_name = dataset_name.replace("ood_", "")
                if is_ood and eval_type == "test_and_ood":
                    display_name = f"[OOD] {display_name}"
                dataset_names.append(display_name)

        if not all_wagers_list:
            return

        num_datasets = len(all_wagers_list)
        num_models = all_wagers_list[0].shape[1]
        model_names = get_model_names_for_plot(self.metadata, self.models, num_models)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        x = np.arange(num_datasets)
        width = 0.8 / num_models

        for i in range(num_models):
            avg_wagers = [
                float(np.mean(all_wagers_list[dataset_idx][:, i]))
                for dataset_idx in range(num_datasets)
            ]
            ax.bar(x + i * width, avg_wagers, width, label=model_names[i], alpha=0.8)

        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        if eval_type == "test":
            title = "Average Wagers by Dataset (Test)"
        elif eval_type == "ood":
            title = "Average Wagers by Dataset (OOD)"
        else:
            title = "Average Wagers by Dataset (Test + OOD)"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels(dataset_names, rotation=20, ha="right")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.05])
        plt.tight_layout()

        save_path = self.checkpoint_dir / f"average_wagers_by_dataset_{eval_type}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.debug(f"Saved wagers plot ({eval_type}) to {save_path}")
        if log_multi_dataset_plot is not None:
            log_multi_dataset_plot(eval_type, save_path)
        plt.close(fig)

    def plot_validation_wagers_by_dataset(
        self,
        val_wagers: np.ndarray,
        results: Dict[str, Any],
    ) -> None:
        if self.checkpoint_dir is None:
            return

        num_models = val_wagers.shape[1]
        model_names = get_model_names_for_plot(self.metadata, self.models, num_models)
        dataset_name = resolve_training_dataset_name(self.metadata, self.dataset)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        avg_wagers = [float(np.mean(val_wagers[:, i])) for i in range(num_models)]
        x = np.arange(1)
        width = 0.8 / num_models

        for i in range(num_models):
            ax.bar(x + i * width, [avg_wagers[i]], width, label=model_names[i], alpha=0.8)

        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        ax.set_title("Average Wagers by Dataset (Validation)", fontsize=12, fontweight="bold")
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels([dataset_name], rotation=20, ha="right")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.05])
        plt.tight_layout()

        avg_save_path = self.checkpoint_dir / "validation_average_wagers_by_dataset.png"
        plt.savefig(avg_save_path, dpi=150, bbox_inches="tight")
        log.debug(f"Saved validation average wagers by dataset plot to {avg_save_path}")

        if self.log_wandb_plot is not None:
            import wandb

            self.log_wandb_plot(
                {"wagers_plot/val/average_by_dataset": wandb.Image(str(avg_save_path))}
            )
        plt.close()

    def plot_wagers_over_time(
        self,
        wagers_history: np.ndarray,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
    ) -> None:
        num_examples, num_models = wagers_history.shape
        model_names = get_model_names_for_plot(self.metadata, self.models, num_models)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        time_steps = np.arange(1, num_examples + 1)
        for i in range(num_models):
            ax.plot(time_steps, wagers_history[:, i], label=model_names[i], alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Wager (Weight)", fontsize=11)
        ax.set_title("Average Wagers Over Time", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        plt.tight_layout()

        if save_path is None and self.checkpoint_dir:
            save_path = self.checkpoint_dir / "wagers_over_time.png"
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.debug(f"Saved overall wagers plot to {save_path}")
            if self.log_wandb_plot is not None:
                import wandb

                self.log_wandb_plot({"wagers_plot/overall": wandb.Image(str(save_path))})
        plt.close()

    def plot_val_wagers_vs_score_diff_for_epoch(
        self,
        val_wagers: np.ndarray,
        val_score_diffs: np.ndarray,
        model_brier_scores: Optional[np.ndarray],
        context_assignment_mask: Optional[np.ndarray],
        context_assignment_kind: Optional[str],
        epoch: int,
        batch_step: Optional[int] = None,
        plot_tag: Optional[str] = None,
    ) -> None:
        self.plot_validation_pair_scatter(
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

    def plot_val_estimated_score_diff_vs_wagers_for_epoch(
        self,
        val_wagers: np.ndarray,
        val_estimated_score_diffs: np.ndarray,
        model_brier_scores: Optional[np.ndarray],
        context_assignment_mask: Optional[np.ndarray],
        context_assignment_kind: Optional[str],
        epoch: int,
        batch_step: Optional[int] = None,
        plot_tag: Optional[str] = None,
    ) -> None:
        self.plot_validation_pair_scatter(
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

    def plot_val_own_score_vs_estimated_score_for_epoch(
        self,
        val_own_scores: np.ndarray,
        val_estimated_scores: np.ndarray,
        model_brier_scores: Optional[np.ndarray],
        context_assignment_mask: Optional[np.ndarray],
        context_assignment_kind: Optional[str],
        epoch: int,
        batch_step: Optional[int] = None,
        plot_tag: Optional[str] = None,
    ) -> None:
        self.plot_validation_pair_scatter(
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

    def plot_val_average_score_vs_estimated_average_score_for_epoch(
        self,
        val_average_scores: np.ndarray,
        val_estimated_average_scores: np.ndarray,
        model_brier_scores: Optional[np.ndarray],
        context_assignment_mask: Optional[np.ndarray],
        context_assignment_kind: Optional[str],
        epoch: int,
        batch_step: Optional[int] = None,
        plot_tag: Optional[str] = None,
    ) -> None:
        self.plot_validation_pair_scatter(
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

    def plot_validation_pair_scatter(
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
    ) -> None:
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
        model_names = get_model_names_for_plot(self.metadata, self.models, num_models)
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
            finite_brier_mask = np.isfinite(model_brier_scores)
            per_example_min_brier = np.min(
                np.where(finite_brier_mask, model_brier_scores, np.inf), axis=1
            )
            best_brier_mask = (
                np.isfinite(per_example_min_brier)[:, np.newaxis]
                & finite_brier_mask
                & np.isclose(
                    model_brier_scores,
                    per_example_min_brier[:, np.newaxis],
                    rtol=1e-6,
                    atol=1e-12,
                )
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
            elif best_brier_mask is None:
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
        fig.suptitle(plot_title, fontsize=12, fontweight="bold", y=0.985)
        fig.text(0.5, 0.942, f"Pearson r (colored): {pearson_colored:.3f}", ha="center", va="top", fontsize=11)

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
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.debug(f"Saved epoch {epoch + 1} {filename_suffix} plot to {save_path}")

        if self.log_wandb_plot is not None:
            import wandb

            if plot_tag is not None:
                wandb_key = f"wagers_plot/val/{wandb_suffix}/{plot_tag}"
            elif batch_step is not None:
                wandb_key = f"wagers_plot/val/{wandb_suffix}/batch_{batch_step}"
            else:
                wandb_key = f"wagers_plot/val/{wandb_suffix}/epoch_{epoch + 1}"
            self.log_wandb_plot({wandb_key: wandb.Image(str(save_path))})
        plt.close()
