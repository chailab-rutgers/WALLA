"""
Shared base for per-model hidden-state WALLA routers (WallaV1 / WallaV2).
"""

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import WageringMethod
from wagering.utils.tensor_helpers import (
    build_mlp,
    ensure_batch_logits,
    l2_normalize_rows,
    sigmoid_row_normalize,
)
from wagering.utils.walla_mechanism import extract_mechanism


class WallaHiddenRouterBase(WageringMethod):
    """Per-model projection + router with sigmoid-normalized wagers."""

    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(num_models, config)
        config = config or {}

        self.common_hidden_dim = int(config.get("common_hidden_dim", 4096))
        self.hidden_layers = list(config.get("hidden_layers", [512, 256]))
        self.learning_rate = float(config.get("learning_rate", 1e-5))
        self.temperature = float(config.get("temperature", 2.0))
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
        self.normalize_hidden_states = bool(config.get("normalize_hidden_states", True))
        self.device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)
        self.lr_decay_factor = float(config.get("lr_decay_factor", 1.0))
        self.lr_decay_steps = int(config.get("lr_decay_steps", 1))

        self.routers = nn.ModuleList()
        self.model_projections = nn.ModuleDict()
        self.optimizers: List[Tuple[int, torch.optim.Optimizer]] = []
        self.schedulers: List[Tuple[int, torch.optim.lr_scheduler._LRScheduler]] = []

        for _ in range(num_models):
            self.routers.append(self._build_router().to(self.device))

        self._training = True
        self._cached_wagers: Optional[torch.Tensor] = None
        self._cached_projected_states: Optional[torch.Tensor] = None
        self._cached_hidden_states_list: Optional[List[torch.Tensor]] = None

    def _build_router(self) -> nn.Module:
        return build_mlp(self.common_hidden_dim, self.hidden_layers, 1)

    @abstractmethod
    def _average_scores_fn(self) -> Callable[..., torch.Tensor]:
        """Return average_scores_v1 or average_scores_v2 from walla_mechanism."""

    def extract_wagers_brs_and_nash_gap(
        self,
        sigmoid_wagers: torch.Tensor,
        model_logits_tensor: torch.Tensor,
        gold_label_tensor: torch.Tensor,
        gold_label_distribution_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return extract_mechanism(
            sigmoid_wagers,
            model_logits_tensor,
            gold_label_tensor,
            gold_label_distribution_tensor,
            self._average_scores_fn(),
        )

    def _forward_wagers_from_hidden(
        self,
        hidden_states_list: List[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        projected_batch_list = []

        for i in range(self.num_models):
            model_hs_batch = hidden_states_list[i]
            model_hidden_dim = model_hs_batch.shape[-1]
            proj_key = f"proj_{i}"

            if proj_key not in self.model_projections:
                projection = nn.Linear(model_hidden_dim, self.common_hidden_dim).to(self.device)
                self.model_projections[proj_key] = projection
                if self._training:
                    params = list(self.routers[i].parameters()) + list(projection.parameters())
                    optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=self.lr_decay_steps,
                        gamma=self.lr_decay_factor,
                    )
                    self.optimizers.append((i, optimizer))
                    self.schedulers.append((i, scheduler))

            model_hs_tensor = torch.as_tensor(
                model_hs_batch, dtype=torch.float32, device=self.device
            )
            if self.normalize_hidden_states:
                model_hs_tensor = l2_normalize_rows(model_hs_tensor)

            with torch.set_grad_enabled(self._training):
                projected_batch = self.model_projections[proj_key](model_hs_tensor)
            projected_batch_list.append(projected_batch)

        raw_wagers_list = [self.routers[i](projected_batch_list[i]) for i in range(self.num_models)]
        raw_wagers_tensor = torch.cat(raw_wagers_list, dim=1)
        wagers, sigmoid_wagers = sigmoid_row_normalize(raw_wagers_tensor, self.temperature)
        return wagers, sigmoid_wagers, projected_batch_list

    def compute_wagers(
        self,
        hidden_states_list: List[np.ndarray],
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if len(hidden_states_list) != self.num_models:
            raise ValueError(f"Expected {self.num_models} models, got {len(hidden_states_list)}")

        wagers, sigmoid_wagers, projected_batch_list = self._forward_wagers_from_hidden(
            hidden_states_list
        )

        if self._training:
            self._cached_wagers = wagers
            self._cached_projected_states = torch.cat(projected_batch_list, dim=1)
            self._cached_hidden_states_list = [
                torch.as_tensor(hidden_states_list[i], dtype=torch.float32, device=self.device).requires_grad_(True)
                for i in range(self.num_models)
            ]

        model_logits, gold_label = ensure_batch_logits(model_logits, gold_label)
        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32, device=self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long, device=self.device)

        gold_label_distribution = kwargs.get("gold_label_distribution")
        gold_label_distribution_tensor = None
        if gold_label_distribution is not None:
            gold_label_distribution_tensor = torch.as_tensor(
                gold_label_distribution, dtype=torch.float32, device=self.device
            )

        brs, nash_gap, score_diff, total_payout = self.extract_wagers_brs_and_nash_gap(
            sigmoid_wagers,
            model_logits_tensor,
            gold_label_tensor,
            gold_label_distribution_tensor=gold_label_distribution_tensor,
        )

        wagers_np = wagers.detach().cpu().numpy()
        if np.any(np.isnan(wagers_np)) or np.any(np.isinf(wagers_np)):
            raise ValueError("Invalid wagers detected (NaN or inf).")

        return {
            "wagers": wagers_np,
            "sigmoid_wagers": sigmoid_wagers.detach().cpu().numpy(),
            "nash_gap": nash_gap.detach().cpu().numpy(),
            "score_diff": score_diff.detach().cpu().numpy(),
            "total_payout": total_payout.detach().cpu().numpy(),
        }

    def update(
        self,
        aggregated_probs: np.ndarray,
        aggregated_pred: np.ndarray,
        gold_label: np.ndarray,
        model_probs: np.ndarray,
        model_logits: np.ndarray,
        question: Optional[str] = None,
        hidden_states: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        del aggregated_probs, aggregated_pred, model_probs, question

        if not self._training:
            return {}
        if hidden_states is None:
            raise ValueError("hidden_states must be provided to update()")
        if not isinstance(hidden_states, (list, tuple)):
            hidden_states = [hidden_states]
        if len(hidden_states) != self.num_models:
            raise ValueError(f"Expected {self.num_models} models in hidden_states, got {len(hidden_states)}")

        self._cached_wagers = None
        self._cached_projected_states = None
        self._cached_hidden_states_list = None

        model_logits, gold_label = ensure_batch_logits(model_logits, gold_label)
        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32, device=self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long, device=self.device)

        gold_label_distribution = kwargs.get("gold_label_distribution")
        gold_label_distribution_tensor = None
        if gold_label_distribution is not None:
            gold_label_distribution_tensor = torch.as_tensor(
                gold_label_distribution, dtype=torch.float32, device=self.device
            )

        hidden_states_tensors = []
        for i in range(self.num_models):
            hs = hidden_states[i]
            if isinstance(hs, np.ndarray):
                hs_tensor = torch.as_tensor(hs, dtype=torch.float32, device=self.device)
            else:
                hs_tensor = hs.to(self.device)
            hs_tensor.requires_grad_(True)
            hidden_states_tensors.append(hs_tensor)

        with torch.enable_grad():
            raw_wagers_list = []
            for j in range(self.num_models):
                proj_key = f"proj_{j}"
                if proj_key in self.model_projections:
                    model_projected_j = self.model_projections[proj_key](hidden_states_tensors[j])
                else:
                    model_projected_j = hidden_states_tensors[j]

                if self.normalize_hidden_states:
                    norms = torch.norm(model_projected_j, dim=-1, keepdim=True)
                    model_projected_j = model_projected_j / (norms + 1e-8)

                raw_wagers_list.append(self.routers[j](model_projected_j))

            raw_wagers_tensor = torch.cat(raw_wagers_list, dim=1)
            _, sigmoid_wagers = sigmoid_row_normalize(raw_wagers_tensor, self.temperature)

            brs, _, _, _ = self.extract_wagers_brs_and_nash_gap(
                sigmoid_wagers,
                model_logits_tensor,
                gold_label_tensor,
                gold_label_distribution_tensor=gold_label_distribution_tensor,
            )
            mseloss = F.mse_loss(sigmoid_wagers, brs, reduction="none")
            all_losses = [mseloss[:, i].mean() for i in range(self.num_models)]

            total_loss = 0.0
            num_updated_models = 0
            for i in range(self.num_models):
                optimizer_i = next((opt for idx, opt in self.optimizers if idx == i), None)
                scheduler_i = next((sch for idx, sch in self.schedulers if idx == i), None)
                if optimizer_i is None:
                    raise RuntimeError(f"No optimizer found for model {i}")

                optimizer_i.zero_grad()
                params_i = list(self.routers[i].parameters())
                proj_key_i = f"proj_{i}"
                if proj_key_i in self.model_projections:
                    params_i += list(self.model_projections[proj_key_i].parameters())

                retain = i < self.num_models - 1
                grads = torch.autograd.grad(
                    all_losses[i], params_i, retain_graph=retain, allow_unused=True
                )
                for param, grad in zip(params_i, grads):
                    if grad is not None:
                        param.grad = grad

                torch.nn.utils.clip_grad_norm_(self.routers[i].parameters(), self.grad_clip_norm)
                if proj_key_i in self.model_projections:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_projections[proj_key_i].parameters(), self.grad_clip_norm
                    )

                optimizer_i.step()
                if scheduler_i is not None:
                    scheduler_i.step()

                total_loss += float(all_losses[i].detach().cpu().numpy())
                num_updated_models += 1

        return {"loss": total_loss / num_updated_models}

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        params = list(self.routers.parameters())
        for proj in self.model_projections.values():
            params.extend(proj.parameters())
        return params

    def train_mode(self):
        for router in self.routers:
            router.train()
        self._training = True
        self._cached_wagers = None
        self._cached_projected_states = None
        self._cached_hidden_states_list = None

    def eval_mode(self):
        for router in self.routers:
            router.eval()
        self._training = False
        self._cached_wagers = None
        self._cached_projected_states = None
        self._cached_hidden_states_list = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "routers_state_dict": {
                f"router_{i}": router.state_dict() for i, router in enumerate(self.routers)
            },
            "model_projections_state_dict": {
                k: v.state_dict() for k, v in self.model_projections.items()
            },
            "optimizers_state_dict": {
                f"optimizer_{model_idx}": opt.state_dict() for model_idx, opt in self.optimizers
            },
            "config": {
                "common_hidden_dim": self.common_hidden_dim,
                "hidden_layers": self.hidden_layers,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "normalize_hidden_states": self.normalize_hidden_states,
                "device": self.device_str,
                "lr_decay_factor": self.lr_decay_factor,
                "lr_decay_steps": self.lr_decay_steps,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if "routers_state_dict" in state_dict:
            for i, router in enumerate(self.routers):
                key = f"router_{i}"
                if key in state_dict["routers_state_dict"]:
                    router.load_state_dict(state_dict["routers_state_dict"][key])

        if "model_projections_state_dict" in state_dict:
            for key, proj_state in state_dict["model_projections_state_dict"].items():
                if key not in self.model_projections:
                    in_features = proj_state["weight"].shape[1]
                    out_features = proj_state["weight"].shape[0]
                    projection = nn.Linear(in_features, out_features).to(self.device)
                    self.model_projections[key] = projection
                    if self._training:
                        model_idx = int(key.split("_")[1])
                        params = list(self.routers[model_idx].parameters()) + list(projection.parameters())
                        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                        self.optimizers.append((model_idx, optimizer))
                self.model_projections[key].load_state_dict(proj_state)

        if "optimizers_state_dict" in state_dict:
            for model_idx, optimizer in self.optimizers:
                key = f"optimizer_{model_idx}"
                if key in state_dict["optimizers_state_dict"]:
                    optimizer.load_state_dict(state_dict["optimizers_state_dict"][key])
