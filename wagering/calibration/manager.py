"""Adaptive temperature scaling over cached multiple-choice logits."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.core.dataset import Dataset
from wagering.core.model import WhiteboxModel

from wagering.utils.checkpoint_utils import (
    generate_calibration_dir,
    generate_per_model_calibration_dir,
)
from wagering.utils import load_datasets_from_config, load_models_from_config
from wagering.utils.dataset_utils import calibration_dataset_configs_include_pubmedqa
from wagering.utils.multi_llm_ensemble import (
    assign_pubmedqa_context_models,
    collect_option_logits_and_hidden_states_for_model,
    extract_hidden_state_features,
    get_model_prompt_variant,
    get_cached_logits_and_hidden_states_for_model,
    set_cached_logits_and_hidden_states_for_model,
    _get_mixed_context_dataset_type,
)

log = logging.getLogger("wagering")


def calibration_enabled(args: Dict[str, Any]) -> bool:
    """Return whether cached-logit calibration is enabled for this run."""
    return bool(args.get("calibrated", False))


def get_calibration_config(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return the calibration config block."""
    if not calibration_enabled(args):
        return {}

    calibration_config = args.get("calibration")
    if not isinstance(calibration_config, dict):
        raise ValueError("calibrated=true requires a calibration config via _include_calibration or calibration")
    if "datasets" not in calibration_config or not calibration_config["datasets"]:
        raise ValueError("Calibration config must define at least one dataset")
    return calibration_config


def _coerce_apply_to_model_indices(
    calibration_config: Dict[str, Any], *, num_models: int
) -> Optional[List[int]]:
    """
    Optional setting to apply calibration to only a subset of ensemble slots.

    Config key:
      calibration.apply_to_model_indices: int | list[int]
    """
    raw = calibration_config.get("apply_to_model_indices")
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        indices = [int(raw)]
    elif isinstance(raw, (list, tuple)):
        indices = [int(x) for x in raw]
    else:
        raise ValueError(
            "calibration.apply_to_model_indices must be an int or list of ints "
            f"(got {type(raw).__name__})"
        )
    if any(i < 0 or i >= int(num_models) for i in indices):
        raise ValueError(
            f"calibration.apply_to_model_indices must be within [0, {int(num_models) - 1}], "
            f"got {indices}"
        )
    return sorted(set(indices))


class _SubsetApplyCalibrator:
    """Wrapper that applies an underlying calibrator to a subset of slots."""

    def __init__(self, calibrator: Any, apply_to_model_indices: List[int]):
        self._calibrator = calibrator
        self._apply_to = set(int(i) for i in apply_to_model_indices)

    def apply_to_stacked_logits(self, all_model_logits: np.ndarray, all_hidden_states: Any, **kwargs) -> np.ndarray:
        logits = np.asarray(all_model_logits, dtype=np.float32)
        calibrated = self._calibrator.apply_to_stacked_logits(logits, all_hidden_states, **kwargs)
        if logits.shape != calibrated.shape:
            raise ValueError("Calibrator returned logits with different shape")
        for slot_idx in range(logits.shape[0]):
            if slot_idx not in self._apply_to:
                calibrated[slot_idx] = logits[slot_idx]
        return calibrated

    def __getattr__(self, name: str) -> Any:
        return getattr(self._calibrator, name)


def resolve_calibration_artifact_dir(args: Dict[str, Any]) -> Optional[Path]:
    """Resolve the artifact directory for the temperature calibrator."""
    if not calibration_enabled(args):
        return None

    calibration_config = get_calibration_config(args)
    base_dir = Path(
        calibration_config.get(
            "checkpoint_base_dir",
            "/common/users/yl2310/MultiLLMs/calibration_checkpoints",
        )
    )
    # PubMedQA: single combined artifact keyed by full ensemble. Non-PubMedQA: per-model
    # artifacts live under base_dir/per_model/ (see generate_per_model_calibration_dir).
    if calibration_dataset_configs_include_pubmedqa(calibration_config["datasets"]):
        return generate_calibration_dir(
            base_dir=base_dir,
            models=args["models"],
            datasets=calibration_config["datasets"],
            calibration_config=calibration_config,
            create_hash=True,
        )
    return base_dir / "per_model"


class TemperatureScalingHead(nn.Module):
    """Small MLP that predicts one temperature per example from cached hidden states."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        min_temperature: float,
        max_temperature: Optional[float],
        init_temperature: float,
        dropout: float,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if init_temperature <= min_temperature:
            raise ValueError("init_temperature must be larger than min_temperature")

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = int(hidden_dim)

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(prev_dim, 1)
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature) if max_temperature is not None else None

        target_delta = max(init_temperature - self.min_temperature, 1e-3)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.fill_(math.log(math.expm1(target_delta)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        features = self.backbone(hidden_states)
        raw_temperature = self.output(features).squeeze(-1)
        temperatures = F.softplus(raw_temperature) + self.min_temperature
        if self.max_temperature is not None:
            temperatures = torch.clamp(temperatures, max=self.max_temperature)
        return temperatures


class AdaptiveTemperatureCalibrator(nn.Module):
    """Per-model adaptive temperature scaling over cached option logits."""

    artifact_name = "temperature_calibration.pt"

    def __init__(
        self,
        model_paths: Sequence[str],
        input_dims: Sequence[int],
        config: Dict[str, Any],
        *,
        slot_model_paths: Optional[Sequence[str]] = None,
        canonical_unique_heads: bool = False,
    ):
        super().__init__()
        if len(model_paths) != len(input_dims):
            raise ValueError("model_paths and input_dims must have the same length")

        self.head_model_paths = list(model_paths)
        self.input_dims = [int(dim) for dim in input_dims]
        self._canonical_unique_heads = bool(canonical_unique_heads)
        if slot_model_paths is not None:
            self.slot_model_paths = list(slot_model_paths)
        else:
            self.slot_model_paths = list(model_paths)

        if self._canonical_unique_heads:
            self._path_to_head_idx = {p: i for i, p in enumerate(self.head_model_paths)}
            for slot_idx, path in enumerate(self.slot_model_paths):
                if path not in self._path_to_head_idx:
                    raise ValueError(
                        f"Ensemble slot {slot_idx} model path {path!r} has no calibration head "
                        f"(heads: {self.head_model_paths})"
                    )
        else:
            if len(self.slot_model_paths) != len(self.head_model_paths):
                raise ValueError(
                    "Without canonical_unique_heads, slot_model_paths must match heads length"
                )

        # Back-compat alias: older code reads .model_paths
        self.model_paths = self.head_model_paths
        self.config = dict(config)
        self.device_name = str(self.config.get("device", "cpu"))
        self.device = torch.device(self.device_name)
        self.batch_size = int(self.config.get("batch_size", 64))
        self.inference_batch_size = int(self.config.get("inference_batch_size", max(self.batch_size, 256)))
        self.learning_rate = float(self.config.get("learning_rate", 1e-4))
        self.weight_decay = float(self.config.get("weight_decay", 1e-5))
        self.num_epochs = int(self.config.get("num_epochs", 20))
        self.validation_split_ratio = float(self.config.get("validation_split_ratio", 0.1))
        self.early_stopping_patience = int(self.config.get("early_stopping_patience", 5))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.shuffle_seed = int(self.config.get("shuffle_seed", 42))

        hidden_layers = self.config.get("head_hidden_layers", self.config.get("hidden_layers", []))
        dropout = float(self.config.get("dropout", 0.0))
        min_temperature = float(self.config.get("min_temperature", 0.05))
        max_temperature = self.config.get("max_temperature", 10.0)
        init_temperature = float(self.config.get("init_temperature", 1.0))

        self.heads = nn.ModuleList(
            [
                TemperatureScalingHead(
                    input_dim=input_dim,
                    hidden_layers=hidden_layers,
                    min_temperature=min_temperature,
                    max_temperature=max_temperature,
                    init_temperature=init_temperature,
                    dropout=dropout,
                )
                for input_dim in self.input_dims
            ]
        )
        self.to(self.device)
        self.eval()

    def _head_index_for_slot(self, slot_idx: int) -> int:
        if not self._canonical_unique_heads:
            return slot_idx
        path = self.slot_model_paths[slot_idx]
        return self._path_to_head_idx[path]

    @staticmethod
    def _coerce_hidden_states_by_model(hidden_states: Any) -> List[np.ndarray]:
        if isinstance(hidden_states, list):
            return [np.asarray(model_hidden, dtype=np.float32) for model_hidden in hidden_states]

        array = np.asarray(hidden_states, dtype=np.float32)
        if array.ndim == 3:
            return [array[model_idx] for model_idx in range(array.shape[0])]
        if array.ndim == 2:
            return [array]

        raise ValueError(f"Unsupported hidden state shape for calibration: {array.shape}")

    @staticmethod
    def _coerce_logits_by_model(model_logits: Any) -> List[np.ndarray]:
        if isinstance(model_logits, list):
            return [np.asarray(model_logit, dtype=np.float32) for model_logit in model_logits]

        array = np.asarray(model_logits, dtype=np.float32)
        if array.ndim == 3:
            return [array[model_idx] for model_idx in range(array.shape[0])]
        if array.ndim == 2:
            return [array]

        raise ValueError(f"Unsupported logit shape for calibration: {array.shape}")

    def freeze(self) -> None:
        """Freeze all heads for inference after calibration."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _evaluate_loss(
        self,
        head: TemperatureScalingHead,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        head.eval()
        if labels.numel() == 0:
            return float("nan")

        losses: List[float] = []
        with torch.no_grad():
            for start in range(0, labels.shape[0], self.inference_batch_size):
                end = min(start + self.inference_batch_size, labels.shape[0])
                batch_hidden = hidden_states[start:end].to(self.device)
                batch_logits = logits[start:end].to(self.device)
                batch_labels = labels[start:end].to(self.device)
                temperatures = head(batch_hidden)
                scaled_logits = batch_logits / temperatures.unsqueeze(-1)
                losses.append(float(F.cross_entropy(scaled_logits, batch_labels).detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("nan")

    def _fit_single_model(
        self,
        model_index: int,
        logits: np.ndarray,
        hidden_states: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        head = self.heads[model_index]
        trainable_params = [parameter for parameter in head.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        rng = np.random.RandomState(self.shuffle_seed + model_index)
        indices = np.arange(labels.shape[0], dtype=np.int64)
        rng.shuffle(indices)

        val_size = int(round(labels.shape[0] * self.validation_split_ratio))
        if val_size >= labels.shape[0]:
            val_size = max(labels.shape[0] - 1, 0)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        if train_indices.size == 0:
            train_indices = indices
            val_indices = np.array([], dtype=np.int64)

        train_hidden = torch.from_numpy(hidden_states[train_indices]).float()
        train_logits = torch.from_numpy(logits[train_indices]).float()
        train_labels = torch.from_numpy(labels[train_indices]).long()
        val_hidden = torch.from_numpy(hidden_states[val_indices]).float()
        val_logits = torch.from_numpy(logits[val_indices]).float()
        val_labels = torch.from_numpy(labels[val_indices]).long()

        train_dataset = TensorDataset(train_hidden, train_logits, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        best_state = copy.deepcopy(head.state_dict())
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        baseline_nll = self._evaluate_loss(
            head=head,
            hidden_states=torch.from_numpy(hidden_states).float(),
            logits=torch.from_numpy(logits).float(),
            labels=torch.from_numpy(labels).long(),
        )

        for epoch in range(self.num_epochs):
            head.train()
            epoch_losses: List[float] = []
            for batch_hidden, batch_logits, batch_labels in train_loader:
                batch_hidden = batch_hidden.to(self.device)
                batch_logits = batch_logits.to(self.device)
                batch_labels = batch_labels.to(self.device)

                temperatures = head(batch_hidden)
                scaled_logits = batch_logits / temperatures.unsqueeze(-1)
                loss = F.cross_entropy(scaled_logits, batch_labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(head.parameters(), self.max_grad_norm)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            if val_indices.size > 0:
                monitored_loss = self._evaluate_loss(head, val_hidden, val_logits, val_labels)
            else:
                monitored_loss = train_loss

            if monitored_loss + 1e-8 < best_val_loss:
                best_val_loss = monitored_loss
                best_state = copy.deepcopy(head.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    break

            log.info(
                "Calibration model %d epoch %d/%d - train_nll=%.4f monitored_nll=%.4f",
                model_index,
                epoch + 1,
                self.num_epochs,
                train_loss,
                monitored_loss,
            )

        head.load_state_dict(best_state)
        final_nll = self._evaluate_loss(
            head=head,
            hidden_states=torch.from_numpy(hidden_states).float(),
            logits=torch.from_numpy(logits).float(),
            labels=torch.from_numpy(labels).long(),
        )

        return {
            "baseline_nll": baseline_nll,
            "final_nll": final_nll,
            "best_val_nll": best_val_loss,
        }

    def fit(
        self,
        all_model_logits: Any,
        all_hidden_states: Any,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit one adaptive temperature head independently per model."""
        logits_by_model = self._coerce_logits_by_model(all_model_logits)
        hidden_states_by_model = self._coerce_hidden_states_by_model(all_hidden_states)
        if len(logits_by_model) != len(self.heads):
            raise ValueError("Mismatch between logits and number of calibration heads")
        if len(hidden_states_by_model) != len(self.heads):
            raise ValueError("Mismatch between hidden states and number of calibration heads")

        labels = np.asarray(labels, dtype=np.int64)
        metrics = {"model_metrics": []}
        for model_index, (logits, hidden_states) in enumerate(zip(logits_by_model, hidden_states_by_model)):
            if logits.shape[0] != hidden_states.shape[0] or logits.shape[0] != labels.shape[0]:
                raise ValueError("Calibration arrays must align across logits, hidden states, and labels")
            metrics["model_metrics"].append(
                self._fit_single_model(model_index, logits=logits, hidden_states=hidden_states, labels=labels)
            )

        self.freeze()
        return metrics

    def predict_temperatures(self, hidden_states: np.ndarray, model_index: int) -> np.ndarray:
        """Predict temperatures for one model over a batch of cached hidden states."""
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        head = self.heads[model_index]
        temperatures: List[np.ndarray] = []
        head.eval()
        with torch.no_grad():
            for start in range(0, hidden_states.shape[0], self.inference_batch_size):
                end = min(start + self.inference_batch_size, hidden_states.shape[0])
                batch_hidden = torch.from_numpy(hidden_states[start:end]).float().to(self.device)
                batch_temperatures = head(batch_hidden).detach().to(dtype=torch.float32).cpu().numpy()
                temperatures.append(batch_temperatures)
        if not temperatures:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(temperatures, axis=0)

    def apply_to_stacked_logits(self, all_model_logits: np.ndarray, all_hidden_states: Any) -> np.ndarray:
        """Apply frozen temperature scaling to stacked logits [num_models, num_examples, num_options]."""
        model_logits = np.asarray(all_model_logits, dtype=np.float32)
        hidden_states_by_model = self._coerce_hidden_states_by_model(all_hidden_states)

        if model_logits.ndim != 3:
            raise ValueError(f"Expected stacked logits with 3 dims, got {model_logits.shape}")
        if len(hidden_states_by_model) != model_logits.shape[0]:
            raise ValueError("Hidden states must provide one array per model")

        calibrated_logits = np.empty_like(model_logits, dtype=np.float32)
        for slot_idx in range(model_logits.shape[0]):
            head_idx = self._head_index_for_slot(slot_idx)
            temperatures = self.predict_temperatures(hidden_states_by_model[slot_idx], head_idx)
            if temperatures.shape[0] != model_logits.shape[1]:
                raise ValueError("Predicted temperatures do not align with logit count")
            calibrated_logits[slot_idx] = model_logits[slot_idx] / temperatures[:, None]
        return calibrated_logits

    def save_pretrained(self, save_dir: str | Path) -> str:
        """Save the calibrator artifact to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        artifact: Dict[str, Any] = {
            "config": self.config,
            "model_paths": self.head_model_paths,
            "input_dims": self.input_dims,
            "state_dict": self.state_dict(),
            "canonical_unique_heads": self._canonical_unique_heads,
        }
        if self._canonical_unique_heads:
            artifact["slot_model_paths"] = self.slot_model_paths
        torch.save(artifact, save_path / self.artifact_name)
        return str(save_path)

    @classmethod
    def load_pretrained(
        cls,
        save_dir: str | Path,
        device: Optional[str] = None,
        slot_model_paths: Optional[Sequence[str]] = None,
    ) -> "AdaptiveTemperatureCalibrator":
        """Load a saved calibrator artifact."""
        save_path = Path(save_dir)
        artifact = torch.load(save_path / cls.artifact_name, map_location="cpu")
        config = dict(artifact["config"])
        if device is not None:
            config["device"] = device
        canonical = bool(artifact.get("canonical_unique_heads", False))
        head_paths = list(artifact["model_paths"])
        if canonical:
            slots = (
                list(slot_model_paths)
                if slot_model_paths is not None
                else list(artifact.get("slot_model_paths", []))
            )
            if not slots:
                raise ValueError(
                    "Canonical temperature calibrator requires slot_model_paths (pass current ensemble paths)."
                )
            calibrator = cls(
                model_paths=head_paths,
                input_dims=artifact["input_dims"],
                config=config,
                slot_model_paths=slots,
                canonical_unique_heads=True,
            )
        else:
            calibrator = cls(
                model_paths=head_paths,
                input_dims=artifact["input_dims"],
                config=config,
                slot_model_paths=None,
                canonical_unique_heads=False,
            )
        calibrator.load_state_dict(artifact["state_dict"])
        calibrator.freeze()
        return calibrator

    @classmethod
    def merge_from_per_model_dirs(
        cls,
        per_model_dirs: Dict[str, Path],
        unique_paths: Sequence[str],
        ensemble_paths: Sequence[str],
        device: Optional[str] = None,
    ) -> "AdaptiveTemperatureCalibrator":
        """Stack independently saved single-head artifacts into one calibrator (non-PubMedQA)."""
        unique_paths = list(unique_paths)
        input_dims: List[int] = []
        merged_state: Dict[str, torch.Tensor] = {}
        merged_config: Dict[str, Any] = {}

        for i, path in enumerate(unique_paths):
            save_dir = Path(per_model_dirs[path])
            artifact_path = save_dir / cls.artifact_name
            artifact = torch.load(artifact_path, map_location="cpu")
            input_dims.append(int(artifact["input_dims"][0]))
            if i == 0:
                merged_config = dict(artifact["config"])
                if device is not None:
                    merged_config["device"] = device
            state = artifact["state_dict"]
            for k, v in state.items():
                if k.startswith("heads.0."):
                    suffix = k[len("heads.0.") :]
                    merged_state[f"heads.{i}.{suffix}"] = v

        calibrator = cls(
            model_paths=unique_paths,
            input_dims=input_dims,
            config=merged_config,
            slot_model_paths=list(ensemble_paths),
            canonical_unique_heads=True,
        )
        calibrator.load_state_dict(merged_state, strict=True)
        calibrator.freeze()
        return calibrator


class ContextConditionedAdaptiveTemperatureCalibrator(AdaptiveTemperatureCalibrator):
    """
    PubMedQA/RACE mixed-context variant.

    Trains and applies two heads per model:
    - head_role=0: model is the assigned "with_context" expert for this example
    - head_role=1: model is "without_context" for this example
    """

    artifact_name = "temperature_calibration_context_conditioned.pt"

    def __init__(
        self,
        model_paths: Sequence[str],
        input_dims: Sequence[int],
        config: Dict[str, Any],
        *,
        slot_model_paths: Optional[Sequence[str]] = None,
        canonical_unique_heads: bool = False,
    ):
        super().__init__(
            model_paths=model_paths,
            input_dims=input_dims,
            config=config,
            slot_model_paths=slot_model_paths,
            canonical_unique_heads=canonical_unique_heads,
        )

        # Replace heads with 2x per model.
        hidden_layers = self.config.get("head_hidden_layers", self.config.get("hidden_layers", []))
        dropout = float(self.config.get("dropout", 0.0))
        min_temperature = float(self.config.get("min_temperature", 0.05))
        max_temperature = self.config.get("max_temperature", 10.0)
        init_temperature = float(self.config.get("init_temperature", 1.0))

        conditioned_heads: List[TemperatureScalingHead] = []
        for input_dim in self.input_dims:
            conditioned_heads.append(
                TemperatureScalingHead(
                    input_dim=input_dim,
                    hidden_layers=hidden_layers,
                    min_temperature=min_temperature,
                    max_temperature=max_temperature,
                    init_temperature=init_temperature,
                    dropout=dropout,
                )
            )
            conditioned_heads.append(
                TemperatureScalingHead(
                    input_dim=input_dim,
                    hidden_layers=hidden_layers,
                    min_temperature=min_temperature,
                    max_temperature=max_temperature,
                    init_temperature=init_temperature,
                    dropout=dropout,
                )
            )
        self.heads = nn.ModuleList(conditioned_heads)
        self.to(self.device)
        # Do not freeze here; `fit()` needs trainable parameters.
        self.eval()

    def _head_index_for_slot_and_role(self, slot_idx: int, role_idx: int) -> int:
        base = 2 * self._head_index_for_slot(slot_idx)
        return base + int(role_idx)

    def fit(
        self,
        all_model_logits: Any,
        all_hidden_states: Any,
        labels: np.ndarray,
        *,
        context_model_index_by_example: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit separate heads for (with_context, without_context) per model slot."""
        logits_by_model = self._coerce_logits_by_model(all_model_logits)
        hidden_states_by_model = self._coerce_hidden_states_by_model(all_hidden_states)

        labels = np.asarray(labels, dtype=np.int64)
        context_idx = np.asarray(context_model_index_by_example, dtype=np.int64)
        if context_idx.ndim != 1 or context_idx.shape[0] != labels.shape[0]:
            raise ValueError("context_model_index_by_example must be 1D and align with labels")
        if np.any(context_idx < 0) or np.any(context_idx >= len(logits_by_model)):
            raise ValueError("context_model_index_by_example contains out-of-range model indices")

        metrics: Dict[str, Any] = {"model_metrics": []}
        for slot_idx, (logits, hidden) in enumerate(zip(logits_by_model, hidden_states_by_model)):
            if logits.shape[0] != hidden.shape[0] or logits.shape[0] != labels.shape[0]:
                raise ValueError("Calibration arrays must align across logits, hidden states, and labels")

            mask_with = context_idx == int(slot_idx)
            mask_wo = ~mask_with
            if not np.any(mask_with):
                raise RuntimeError(
                    f"No with_context examples found for model slot {slot_idx}; "
                    "did you call assign_pubmedqa_context_models for this dataset?"
                )
            if not np.any(mask_wo):
                raise RuntimeError(f"No without_context examples found for model slot {slot_idx}")

            head_with = self.heads[self._head_index_for_slot_and_role(slot_idx, 0)]
            head_wo = self.heads[self._head_index_for_slot_and_role(slot_idx, 1)]

            # Temporarily swap .heads so we can reuse _fit_single_model by index.
            original_heads = self.heads
            try:
                self.heads = nn.ModuleList([head_with, head_wo])
                with_metrics = self._fit_single_model(
                    0,
                    logits=np.asarray(logits[mask_with], dtype=np.float32),
                    hidden_states=np.asarray(hidden[mask_with], dtype=np.float32),
                    labels=np.asarray(labels[mask_with], dtype=np.int64),
                )
                wo_metrics = self._fit_single_model(
                    1,
                    logits=np.asarray(logits[mask_wo], dtype=np.float32),
                    hidden_states=np.asarray(hidden[mask_wo], dtype=np.float32),
                    labels=np.asarray(labels[mask_wo], dtype=np.int64),
                )
            finally:
                self.heads = original_heads

            metrics["model_metrics"].append(
                {
                    "with_context": with_metrics,
                    "without_context": wo_metrics,
                    "num_with_context": int(np.sum(mask_with)),
                    "num_without_context": int(np.sum(mask_wo)),
                }
            )

        self.freeze()
        return metrics

    def apply_to_stacked_logits(
        self,
        all_model_logits: np.ndarray,
        all_hidden_states: Any,
        *,
        context_model_index_by_example: np.ndarray,
    ) -> np.ndarray:
        """Apply role-conditioned temperature scaling to stacked logits."""
        model_logits = np.asarray(all_model_logits, dtype=np.float32)
        hidden_states_by_model = self._coerce_hidden_states_by_model(all_hidden_states)
        context_idx = np.asarray(context_model_index_by_example, dtype=np.int64)
        if context_idx.ndim != 1 or context_idx.shape[0] != model_logits.shape[1]:
            raise ValueError("context_model_index_by_example must be 1D and align with num_examples")
        if np.any(context_idx < 0) or np.any(context_idx >= model_logits.shape[0]):
            raise ValueError("context_model_index_by_example contains out-of-range model indices")

        calibrated_logits = np.empty_like(model_logits, dtype=np.float32)
        for slot_idx in range(model_logits.shape[0]):
            head_with = self.heads[self._head_index_for_slot_and_role(slot_idx, 0)]
            head_wo = self.heads[self._head_index_for_slot_and_role(slot_idx, 1)]
            hidden = np.asarray(hidden_states_by_model[slot_idx], dtype=np.float32)
            if hidden.shape[0] != model_logits.shape[1]:
                raise ValueError("Hidden states do not align with logit count")

            mask_with = context_idx == int(slot_idx)
            mask_wo = ~mask_with

            temps = np.empty((model_logits.shape[1],), dtype=np.float32)
            if np.any(mask_with):
                with torch.no_grad():
                    t = head_with(torch.from_numpy(hidden[mask_with]).float().to(self.device))
                    temps[mask_with] = t.detach().to(dtype=torch.float32).cpu().numpy()
            if np.any(mask_wo):
                with torch.no_grad():
                    t = head_wo(torch.from_numpy(hidden[mask_wo]).float().to(self.device))
                    temps[mask_wo] = t.detach().to(dtype=torch.float32).cpu().numpy()

            calibrated_logits[slot_idx] = model_logits[slot_idx] / temps[:, None]

        return calibrated_logits


def _prepare_models_for_datasets(
    model_cfgs: Sequence[Dict[str, Any]],
    datasets: Sequence[Dataset],
    option_tokens: Sequence[str],
    cache_path: Optional[str],
    require_hidden_states: bool,
) -> Tuple[List[WhiteboxModel | str], List[str]]:
    """Load only the models that are missing required cached artifacts."""
    cache_miss_indices: List[int] = []
    model_names: List[str] = []

    for idx, model_cfg in enumerate(model_cfgs):
        model_path = model_cfg["path"]
        model_names.append(model_path.replace("/", "_"))
        dataset_cache_ok = True
        for dataset in datasets:
            prompt_variant = get_model_prompt_variant(dataset, model_index=idx)
            cached_logits, cached_hidden_states, _ = get_cached_logits_and_hidden_states_for_model(
                model_path,
                dataset,
                list(option_tokens),
                prompt_variant=prompt_variant,
                model_index=idx,
                hidden_state_layers=[-1],
            )
            if cached_logits is None or (require_hidden_states and cached_hidden_states is None):
                dataset_cache_ok = False
                break
        if not dataset_cache_ok:
            cache_miss_indices.append(idx)

    if not cache_miss_indices:
        return [model_cfg["path"] for model_cfg in model_cfgs], model_names

    missing_cfgs = [model_cfgs[idx] for idx in cache_miss_indices]
    missing_models, missing_model_names = load_models_from_config(
        missing_cfgs,
        cache_kwargs={"cache_dir": cache_path} if cache_path else {},
    )
    missing_name_map = {idx: name for idx, name in zip(cache_miss_indices, missing_model_names)}
    missing_iter = iter(missing_models)

    prepared_models: List[WhiteboxModel | str] = []
    for idx, model_cfg in enumerate(model_cfgs):
        if idx in cache_miss_indices:
            prepared_models.append(next(missing_iter))
            model_names[idx] = missing_name_map.get(idx, model_names[idx])
        else:
            prepared_models.append(model_cfg["path"])

    return prepared_models, model_names


def _collect_calibration_arrays(
    models: Sequence[WhiteboxModel | str],
    datasets: Sequence[Dataset],
    option_tokens: Sequence[str],
    balance_datasets: bool,
    shuffle_seed: int,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Load or collect cached logits and hidden states for calibration datasets."""
    per_dataset_logits: List[np.ndarray] = []
    per_dataset_hidden_states: List[List[np.ndarray]] = []
    per_dataset_labels: List[np.ndarray] = []
    per_dataset_context_assignments: List[np.ndarray] = []
    saw_mixed_context = False

    for dataset in datasets:
        dataset_type = _get_mixed_context_dataset_type(dataset)
        if dataset_type is not None:
            saw_mixed_context = True
            raw = getattr(dataset, f"{dataset_type}_context_assignment_by_example", None)
            if not isinstance(raw, list) or len(raw) != len(dataset.x):
                raise RuntimeError(
                    "Mixed-context calibration dataset missing per-example assignments. "
                    "Call assign_pubmedqa_context_models before calibration cache checks/collection."
                )
            per_dataset_context_assignments.append(np.asarray(raw, dtype=np.int64))
        else:
            per_dataset_context_assignments.append(np.full((len(dataset.x),), -1, dtype=np.int64))

        dataset_model_logits: List[np.ndarray] = []
        dataset_model_hidden_states: List[np.ndarray] = []
        dataset_labels: Optional[np.ndarray] = None

        for model_idx, model in enumerate(models):
            model_path = model if isinstance(model, str) else model.model_path
            prompt_variant = get_model_prompt_variant(dataset, model_index=model_idx)
            cached_logits, cached_hidden_states, cached_labels = get_cached_logits_and_hidden_states_for_model(
                model_path,
                dataset,
                list(option_tokens),
                prompt_variant=prompt_variant,
                model_index=model_idx,
                hidden_state_layers=[-1],
            )

            if cached_logits is None or cached_hidden_states is None:
                if isinstance(model, str):
                    raise RuntimeError(
                        f"Cache miss for model path {model}. A loaded model instance is required to collect calibration caches."
                    )
                model_logits, model_hidden_states_all_layers, model_labels = collect_option_logits_and_hidden_states_for_model(
                    model,
                    dataset,
                    list(option_tokens),
                    model_identifier=str(model_path),
                    model_index=model_idx,
                    hidden_state_layers=[-1],
                )
                set_cached_logits_and_hidden_states_for_model(
                    model,
                    dataset,
                    list(option_tokens),
                    model_logits,
                    model_hidden_states_all_layers,
                    model_labels,
                    prompt_variant=prompt_variant,
                    model_index=model_idx,
                    hidden_state_layers=[-1],
                )
                model_hidden_states = extract_hidden_state_features(model_hidden_states_all_layers, [-1])
                if model_hidden_states is None:
                    raise RuntimeError("Failed to extract hidden states for calibration")
            else:
                model_logits = cached_logits
                model_hidden_states = cached_hidden_states
                model_labels = cached_labels

            model_logits = np.asarray(model_logits, dtype=np.float32)
            model_hidden_states = np.asarray(model_hidden_states, dtype=np.float32)
            model_labels = np.asarray(model_labels, dtype=np.int64)
            if dataset_labels is None:
                dataset_labels = model_labels
            elif not np.array_equal(dataset_labels, model_labels):
                raise RuntimeError("Calibration labels must match across models for the same dataset")

            dataset_model_logits.append(model_logits)
            dataset_model_hidden_states.append(model_hidden_states)

        if dataset_labels is None:
            raise RuntimeError("Calibration dataset produced no labels")

        per_dataset_logits.append(np.stack(dataset_model_logits, axis=0))
        per_dataset_hidden_states.append(dataset_model_hidden_states)
        per_dataset_labels.append(dataset_labels)

    if balance_datasets and per_dataset_labels:
        rng = np.random.RandomState(shuffle_seed)
        min_size = min(label_array.shape[0] for label_array in per_dataset_labels)
        for dataset_idx, label_array in enumerate(per_dataset_labels):
            if label_array.shape[0] == min_size:
                indices = np.arange(min_size, dtype=np.int64)
            else:
                indices = np.sort(rng.choice(label_array.shape[0], size=min_size, replace=False))
            per_dataset_logits[dataset_idx] = per_dataset_logits[dataset_idx][:, indices, :]
            per_dataset_hidden_states[dataset_idx] = [hidden_state[indices] for hidden_state in per_dataset_hidden_states[dataset_idx]]
            per_dataset_labels[dataset_idx] = label_array[indices]

    stacked_logits = np.concatenate(per_dataset_logits, axis=1)
    hidden_states_by_model = [
        np.concatenate([per_dataset_hidden_states[dataset_idx][model_idx] for dataset_idx in range(len(per_dataset_hidden_states))], axis=0)
        for model_idx in range(len(models))
    ]
    labels = np.concatenate(per_dataset_labels, axis=0)
    context_assignments = (
        np.concatenate(per_dataset_context_assignments, axis=0) if saw_mixed_context else None
    )
    return stacked_logits, hidden_states_by_model, labels, context_assignments


def _fit_or_load_per_model_calibrators(
    args: Dict[str, Any],
    calibration_config: Dict[str, Any],
    base_root: Path,
    force_refit: bool,
) -> Tuple[AdaptiveTemperatureCalibrator, str, bool]:
    """Fit or load one temperature head per unique HF path; merge for the ensemble (non-PubMedQA)."""
    ensemble_paths = [m["path"] for m in args["models"] if "path" in m]
    unique_paths = sorted(set(ensemble_paths))
    if not unique_paths:
        raise ValueError("No model paths in args['models']")

    reuse_existing = bool(calibration_config.get("reuse_existing", True))
    option_tokens = args.get("option_tokens", ["A", "B", "C", "D"])
    dataset_split_seed = int(args.get("dataset_split_seed", 42))
    device_str = str(calibration_config.get("device", "cpu"))

    per_model_dirs: Dict[str, Path] = {
        p: generate_per_model_calibration_dir(
            base_dir=base_root,
            model_path=p,
            datasets=calibration_config["datasets"],
            calibration_config=calibration_config,
        )
        for p in unique_paths
    }

    if reuse_existing and not force_refit:
        all_exist = all(
            (per_model_dirs[p] / AdaptiveTemperatureCalibrator.artifact_name).exists()
            for p in unique_paths
        )
        if all_exist:
            calibrator = AdaptiveTemperatureCalibrator.merge_from_per_model_dirs(
                per_model_dirs=per_model_dirs,
                unique_paths=unique_paths,
                ensemble_paths=ensemble_paths,
                device=device_str,
            )
            log.info(
                "Loaded existing per-model temperature calibrators under %s (merged %d heads)",
                base_root / "per_model",
                len(unique_paths),
            )
            return calibrator, str(base_root / "per_model"), False

    datasets, dataset_names = load_datasets_from_config(
        calibration_config["datasets"],
        split="train",
        random_seed=dataset_split_seed,
    )
    log.info("Loaded %d calibration datasets: %s", len(datasets), dataset_names)

    any_fitted = False
    total_models = len(unique_paths)
    for model_idx, p in enumerate(unique_paths, start=1):
        artifact_file = per_model_dirs[p] / AdaptiveTemperatureCalibrator.artifact_name
        if artifact_file.exists() and reuse_existing and not force_refit:
            log.info(
                "Calibration artifact already exists for model %d/%d: %s (skipping refit)",
                model_idx,
                total_models,
                p,
            )
            continue
        any_fitted = True
        log.info(
            "Calibration cache miss for model %d/%d: %s. Loading model to collect caches...",
            model_idx,
            total_models,
            p,
        )
        models, _ = _prepare_models_for_datasets(
            model_cfgs=[{"path": p}],
            datasets=datasets,
            option_tokens=option_tokens,
            cache_path=args.get("cache_path"),
            require_hidden_states=True,
        )
        all_logits, all_hidden, labels, _ = _collect_calibration_arrays(
            models=models,
            datasets=datasets,
            option_tokens=option_tokens,
            balance_datasets=bool(calibration_config.get("balance_datasets", True)),
            shuffle_seed=int(calibration_config.get("shuffle_seed", 42)),
        )
        dim = all_hidden[0].shape[1]
        one_calibrator = AdaptiveTemperatureCalibrator(
            model_paths=[p],
            input_dims=[dim],
            config=calibration_config,
            slot_model_paths=[p],
            canonical_unique_heads=False,
        )
        metrics = one_calibrator.fit(all_logits, all_hidden, labels)
        one_calibrator.save_pretrained(per_model_dirs[p])
        log.info(
            "Saved per-model temperature calibrator for %s to %s metrics=%s",
            p,
            per_model_dirs[p],
            metrics,
        )

    merged = AdaptiveTemperatureCalibrator.merge_from_per_model_dirs(
        per_model_dirs=per_model_dirs,
        unique_paths=unique_paths,
        ensemble_paths=ensemble_paths,
        device=device_str,
    )
    return merged, str(base_root / "per_model"), any_fitted


def fit_or_load_logit_calibrator(
    args: Dict[str, Any],
    calibration_path: Optional[str] = None,
    force_refit: bool = False,
) -> Tuple[Optional[AdaptiveTemperatureCalibrator], Optional[str], bool]:
    """Fit or load the cached-logit temperature calibrator for a run."""
    if not calibration_enabled(args):
        return None, None, False

    calibration_config = get_calibration_config(args)
    apply_to_model_indices = _coerce_apply_to_model_indices(
        calibration_config, num_models=len(args.get("models") or [])
    )
    include_pubmedqa = calibration_dataset_configs_include_pubmedqa(calibration_config["datasets"])
    base_root = Path(calibration_path) if calibration_path is not None else Path(
        calibration_config.get(
            "checkpoint_base_dir",
            "/common/users/yl2310/MultiLLMs/calibration_checkpoints",
        )
    )

    if not include_pubmedqa:
        calibrator, artifact_dir, fitted = _fit_or_load_per_model_calibrators(
            args, calibration_config, base_root, force_refit
        )
        if apply_to_model_indices is not None:
            calibrator = _SubsetApplyCalibrator(calibrator, apply_to_model_indices)
        return calibrator, artifact_dir, fitted

    artifact_dir = Path(calibration_path) if calibration_path is not None else generate_calibration_dir(
        base_dir=base_root,
        models=args["models"],
        datasets=calibration_config["datasets"],
        calibration_config=calibration_config,
        create_hash=True,
    )
    reuse_existing = bool(calibration_config.get("reuse_existing", True))
    ensemble_paths = [model_cfg["path"] for model_cfg in args["models"]]
    all_slots_share_model = len(set(ensemble_paths)) == 1
    conditioned_requested = bool(calibration_config.get("condition_on_context_assignment", False))
    artifact_file = artifact_dir / AdaptiveTemperatureCalibrator.artifact_name
    conditioned_file = artifact_dir / ContextConditionedAdaptiveTemperatureCalibrator.artifact_name
    if (artifact_file.exists() or conditioned_file.exists()) and reuse_existing and not force_refit:
        # Backwards compatible: prefer conditioned artifact if requested & present.
        if conditioned_requested and conditioned_file.exists():
            calibrator = ContextConditionedAdaptiveTemperatureCalibrator.load_pretrained(
                artifact_dir,
                device=str(calibration_config.get("device", "cpu")),
                slot_model_paths=ensemble_paths,
            )
        else:
            calibrator = AdaptiveTemperatureCalibrator.load_pretrained(
                artifact_dir,
                device=str(calibration_config.get("device", "cpu")),
                slot_model_paths=ensemble_paths,
            )
        if (
            all_slots_share_model
            and not conditioned_requested
            and not getattr(calibrator, "_canonical_unique_heads", False)
        ):
            log.warning(
                "Existing calibrator at %s uses per-slot heads even though all ensemble slots share model path %s. "
                "Refitting canonical shared-head calibrator.",
                artifact_dir,
                ensemble_paths[0],
            )
        else:
            log.info("Loaded existing temperature calibrator from %s", artifact_dir)
            if apply_to_model_indices is not None:
                calibrator = _SubsetApplyCalibrator(calibrator, apply_to_model_indices)
            return calibrator, str(artifact_dir), False

    dataset_split_seed = int(args.get("dataset_split_seed", 42))
    datasets, dataset_names = load_datasets_from_config(
        calibration_config["datasets"],
        split="train",
        random_seed=dataset_split_seed,
    )
    log.info("Loaded %d calibration datasets: %s", len(datasets), dataset_names)

    pubmedqa_context_seed = dataset_split_seed
    pubmedqa_assignments = assign_pubmedqa_context_models(
        datasets,
        [model_cfg["path"] for model_cfg in args["models"]],
        random_seed=pubmedqa_context_seed,
    )
    for dataset_idx, assignment_info in pubmedqa_assignments.items():
        dataset_name = dataset_names[dataset_idx] if dataset_idx < len(dataset_names) else f"dataset_{dataset_idx}"
        assignment_hash = assignment_info.get("assignment_hash", "unknown")
        num_examples = assignment_info.get("num_examples", len(datasets[dataset_idx].x))
        routing_seed = assignment_info.get("routing_seed", pubmedqa_context_seed)
        model_context_counts = assignment_info.get("model_context_counts", [])
        log.info(
            "PubMedQA balanced mixed-context assignment for calibration dataset %s: assignment_hash=%s, num_examples=%s, routing_seed=%s",
            dataset_name,
            assignment_hash,
            num_examples,
            routing_seed,
        )
        if isinstance(model_context_counts, list):
            for model_idx, context_count in enumerate(model_context_counts):
                model_path = args["models"][model_idx]["path"] if model_idx < len(args["models"]) else f"model_{model_idx}"
                log.info(
                    "PubMedQA context count for calibration dataset %s: model_index=%d, model=%s, context_examples=%d",
                    dataset_name,
                    model_idx,
                    model_path,
                    int(context_count),
                )

    models, _ = _prepare_models_for_datasets(
        model_cfgs=args["models"],
        datasets=datasets,
        option_tokens=args.get("option_tokens", ["A", "B", "C", "D"]),
        cache_path=args.get("cache_path"),
        require_hidden_states=True,
    )

    all_model_logits, all_hidden_states, labels, context_assignments = _collect_calibration_arrays(
        models=models,
        datasets=datasets,
        option_tokens=args.get("option_tokens", ["A", "B", "C", "D"]),
        balance_datasets=bool(calibration_config.get("balance_datasets", True)),
        shuffle_seed=int(calibration_config.get("shuffle_seed", 42)),
    )

    if conditioned_requested and context_assignments is None:
        raise RuntimeError(
            "condition_on_context_assignment=true requires a mixed-context calibration dataset "
            "with per-example context assignments (e.g. PubMedQA mixed-context)."
        )

    # NOTE: For context-conditioned calibration on PubMedQA-style runs, we need
    # per-slot heads because `context_assignments` is expressed in slot indices.
    # Collapsing to a single canonical slot would make those indices out-of-range.
    if all_slots_share_model and not conditioned_requested:
        # When all ensemble entries point to the same HF model path, calibrating
        # separate per-slot heads introduces artificial slot-specific differences.
        # Train one shared canonical head and map every slot to it.
        shared_path = ensemble_paths[0]
        shared_logits = np.concatenate(
            [all_model_logits[slot_idx] for slot_idx in range(all_model_logits.shape[0])],
            axis=0,
        )
        shared_hidden = np.concatenate(
            [all_hidden_states[slot_idx] for slot_idx in range(len(all_hidden_states))],
            axis=0,
        )
        shared_labels = np.concatenate(
            [labels for _ in range(all_model_logits.shape[0])],
            axis=0,
        )

        calibrator = AdaptiveTemperatureCalibrator(
            model_paths=[shared_path],
            input_dims=[shared_hidden.shape[1]],
            config=calibration_config,
            slot_model_paths=ensemble_paths,
            canonical_unique_heads=True,
        )
        metrics = calibrator.fit(
            all_model_logits=shared_logits[np.newaxis, :, :],
            all_hidden_states=[shared_hidden],
            labels=shared_labels,
        )
    else:
        if conditioned_requested:
            calibrator = ContextConditionedAdaptiveTemperatureCalibrator(
                model_paths=ensemble_paths,
                input_dims=[hidden_state.shape[1] for hidden_state in all_hidden_states],
                config=calibration_config,
                slot_model_paths=None,
                canonical_unique_heads=False,
            )
            metrics = calibrator.fit(
                all_model_logits=all_model_logits,
                all_hidden_states=all_hidden_states,
                labels=labels,
                context_model_index_by_example=context_assignments,
            )
        else:
            calibrator = AdaptiveTemperatureCalibrator(
                model_paths=ensemble_paths,
                input_dims=[hidden_state.shape[1] for hidden_state in all_hidden_states],
                config=calibration_config,
                slot_model_paths=None,
                canonical_unique_heads=False,
            )
            metrics = calibrator.fit(
                all_model_logits=all_model_logits,
                all_hidden_states=all_hidden_states,
                labels=labels,
            )
    calibrator.save_pretrained(artifact_dir)
    log.info("Saved temperature calibrator to %s", artifact_dir)
    log.info("Calibration metrics: %s", metrics)
    if apply_to_model_indices is not None:
        calibrator = _SubsetApplyCalibrator(calibrator, apply_to_model_indices)
    return calibrator, str(artifact_dir), True