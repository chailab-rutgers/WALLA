"""
Shared augmented MSE-BR wagering implementation.

This module provides a reusable base class for V2/V3 augmented variants with:
- Primary wagering head (existing behavior)
- Optional ablation heads for own-score and average-score estimation
- Independent gradient updates for the three heads
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Ensure local project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from .base import WageringMethod


class MSEBrWagersAugmentedBase(WageringMethod):
    """Reusable augmented base for MSE-BR wagering variants."""

    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(num_models, config)
        config = config or {}

        self.common_hidden_dim = int(config.get("common_hidden_dim", 4096))
        self.hidden_layers = list(config.get("hidden_layers", [512, 256]))
        self.learning_rate = float(config.get("learning_rate", 1e-5))
        self.temperature = float(config.get("temperature", 2.0))
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
        self.normalize_hidden_states = bool(config.get("normalize_hidden_states", True))
        self.score_function_name = str(config.get("score_function", "normalized_linear"))
        self.device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)
        self.lr_decay_factor = float(config.get("lr_decay_factor", 1.0))
        self.lr_decay_steps = int(config.get("lr_decay_steps", 1))
        self.ablation_study = bool(config.get("ablation_study", False))

        hidden_state_layers_cfg = config.get("hidden_state_layers")
        if hidden_state_layers_cfg is None:
            # Default to first, second-last, and last transformer layers.
            self.hidden_state_layers = [0, -2, -1]
        else:
            if not isinstance(hidden_state_layers_cfg, (list, tuple)):
                raise ValueError(
                    "hidden_state_layers must be a list/tuple of ints, "
                    f"got {type(hidden_state_layers_cfg).__name__}"
                )
            self.hidden_state_layers = [int(x) for x in hidden_state_layers_cfg]

        self.wager_routers = nn.ModuleList([self._build_head() for _ in range(num_models)])
        self.wager_projections = nn.ModuleDict()

        self.estimate_score_routers = nn.ModuleList([self._build_head() for _ in range(num_models)])
        self.estimate_score_projections = nn.ModuleDict()

        self.estimate_average_score_routers = nn.ModuleList([self._build_head() for _ in range(num_models)])
        self.estimate_average_score_projections = nn.ModuleDict()

        self.optimizers: List[Tuple[int, str, torch.optim.Optimizer]] = []
        self.schedulers: List[Tuple[int, str, torch.optim.lr_scheduler._LRScheduler]] = []
        self._training = True

        # Backward-compatible aliases used by some callers.
        self.routers = self.wager_routers
        self.model_projections = self.wager_projections

    def _build_head(self) -> nn.Module:
        layers = []
        prev_dim = self.common_hidden_dim
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers).to(self.device)

    def _get_projection_store(self, head_name: str) -> nn.ModuleDict:
        if head_name == "wager":
            return self.wager_projections
        if head_name == "score":
            return self.estimate_score_projections
        if head_name == "avg":
            return self.estimate_average_score_projections
        raise ValueError(f"Unknown head_name={head_name}")

    def _get_router_store(self, head_name: str) -> nn.ModuleList:
        if head_name == "wager":
            return self.wager_routers
        if head_name == "score":
            return self.estimate_score_routers
        if head_name == "avg":
            return self.estimate_average_score_routers
        raise ValueError(f"Unknown head_name={head_name}")

    def _ensure_head_modules(self, model_idx: int, hidden_dim: int, head_name: str):
        projection_store = self._get_projection_store(head_name)
        router_store = self._get_router_store(head_name)
        proj_key = f"{head_name}_proj_{model_idx}"

        if proj_key in projection_store:
            return

        projection = nn.Linear(hidden_dim, self.common_hidden_dim).to(self.device)
        projection_store[proj_key] = projection

        if self._training:
            params = list(router_store[model_idx].parameters()) + list(projection.parameters())
            optimizer = torch.optim.Adam(params, lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_decay_steps,
                gamma=self.lr_decay_factor,
            )
            self.optimizers.append((model_idx, head_name, optimizer))
            self.schedulers.append((model_idx, head_name, scheduler))

    def _project_and_route(
        self,
        hidden_states_list: List[np.ndarray],
        head_name: str,
    ) -> torch.Tensor:
        routed = []
        projection_store = self._get_projection_store(head_name)
        router_store = self._get_router_store(head_name)

        for i in range(self.num_models):
            hs_batch = hidden_states_list[i]
            hidden_dim = hs_batch.shape[-1]
            self._ensure_head_modules(i, hidden_dim, head_name)

            hs_tensor = torch.as_tensor(hs_batch, dtype=torch.float32, device=self.device)
            if self.normalize_hidden_states:
                hs_norm = torch.norm(hs_tensor, dim=1, keepdim=True)
                hs_tensor = hs_tensor / (hs_norm + 1e-8)

            proj_key = f"{head_name}_proj_{i}"
            projected = projection_store[proj_key](hs_tensor)
            routed.append(router_store[i](projected))

        return torch.cat(routed, dim=1)

    def _normalize_wagers(self, raw_wagers_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigmoid_wagers = torch.sigmoid(raw_wagers_tensor / self.temperature)
        sigmoid_wagers = torch.clamp(sigmoid_wagers, min=1e-16, max=1.0 - 1e-16)
        sigmoid_sum = torch.sum(sigmoid_wagers, dim=1, keepdim=True)
        if torch.any(sigmoid_sum < 1e-16):
            raise RuntimeError("Near-zero sigmoid sum detected during compute_wagers().")
        wagers = sigmoid_wagers / sigmoid_sum
        return wagers, sigmoid_wagers

    def _extract_components(
        self,
        sigmoid_wagers: torch.Tensor,
        model_logits_tensor: torch.Tensor,
        gold_label_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        probs = F.softmax(model_logits_tensor, dim=-1)
        _, num_models, num_options = probs.shape
        gt_onehot = F.one_hot(gold_label_tensor, num_classes=num_options).float()
        gt_onehot_expanded = gt_onehot.unsqueeze(1).expand(-1, num_models, num_options)

        squared_errors = (probs - gt_onehot_expanded) ** 2
        brier_scores = squared_errors.sum(dim=-1)
        scores = 0.5 * (2 - brier_scores)

        average_scores = self._compute_average_scores(
            probs=probs,
            gt_onehot_expanded=gt_onehot_expanded,
            scores=scores,
            sigmoid_wagers=sigmoid_wagers,
        )
        score_diff = scores - average_scores
        brs = torch.clamp(score_diff, min=1e-16, max=1.0 - 1e-16)
        total_payout = sigmoid_wagers * (score_diff - 0.5 * sigmoid_wagers)
        nash_gap = brs * (score_diff - 0.5 * brs) - total_payout

        return {
            "scores": scores,
            "average_scores": average_scores,
            "score_diff": score_diff,
            "brs": brs,
            "nash_gap": nash_gap,
            "total_payout": total_payout,
        }

    def _compute_average_scores(
        self,
        probs: torch.Tensor,
        gt_onehot_expanded: torch.Tensor,
        scores: torch.Tensor,
        sigmoid_wagers: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_optimizer_and_scheduler(self, model_idx: int, head_name: str):
        optimizer_i = None
        scheduler_i = None
        for idx, name, opt in self.optimizers:
            if idx == model_idx and name == head_name:
                optimizer_i = opt
                break
        for idx, name, sch in self.schedulers:
            if idx == model_idx and name == head_name:
                scheduler_i = sch
                break
        if optimizer_i is None:
            raise RuntimeError(f"No optimizer found for model {model_idx}, head {head_name}")
        return optimizer_i, scheduler_i

    def _get_head_params(self, model_idx: int, head_name: str) -> List[torch.nn.Parameter]:
        router_store = self._get_router_store(head_name)
        projection_store = self._get_projection_store(head_name)
        params = list(router_store[model_idx].parameters())
        proj_key = f"{head_name}_proj_{model_idx}"
        if proj_key in projection_store:
            params += list(projection_store[proj_key].parameters())
        return params

    def compute_wagers(
        self,
        hidden_states_list: List[np.ndarray],
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if len(hidden_states_list) != self.num_models:
            raise ValueError(f"Expected {self.num_models} models, got {len(hidden_states_list)}")

        raw_wagers_tensor = self._project_and_route(hidden_states_list, head_name="wager")
        wagers, sigmoid_wagers = self._normalize_wagers(raw_wagers_tensor)

        estimated_scores = None
        estimated_average_scores = None
        estimated_score_diff = None

        if self.ablation_study:
            score_logits = self._project_and_route(hidden_states_list, head_name="score")
            avg_logits = self._project_and_route(hidden_states_list, head_name="avg")
            estimated_scores = torch.sigmoid(score_logits)
            estimated_average_scores = torch.sigmoid(avg_logits)
            estimated_score_diff = torch.relu(estimated_scores - estimated_average_scores)

        is_batch = model_logits.ndim == 3
        if not is_batch:
            model_logits = model_logits[np.newaxis, :, :]
            gold_label = np.array([gold_label])

        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32, device=self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long, device=self.device)
        components = self._extract_components(
            sigmoid_wagers=sigmoid_wagers,
            model_logits_tensor=model_logits_tensor,
            gold_label_tensor=gold_label_tensor,
        )

        wagers_np = wagers.detach().cpu().numpy()
        if np.any(np.isnan(wagers_np)) or np.any(np.isinf(wagers_np)):
            raise ValueError("Invalid wagers detected (NaN or inf).")

        result = {
            "wagers": wagers_np,
            "sigmoid_wagers": sigmoid_wagers.detach().cpu().numpy(),
            "nash_gap": components["nash_gap"].detach().cpu().numpy(),
            "score_diff": components["score_diff"].detach().cpu().numpy(),
            "total_payout": components["total_payout"].detach().cpu().numpy(),
            "scores": components["scores"].detach().cpu().numpy(),
            "average_scores": components["average_scores"].detach().cpu().numpy(),
        }

        if self.ablation_study:
            result["estimated_score"] = estimated_scores.detach().cpu().numpy()
            result["estimated_average_scores"] = estimated_average_scores.detach().cpu().numpy()
            result["estimated_score_diff"] = estimated_score_diff.detach().cpu().numpy()

        return result

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
        if not self._training:
            return {}
        if hidden_states is None:
            raise ValueError("hidden_states must be provided to update()")
        if not isinstance(hidden_states, (list, tuple)):
            hidden_states = [hidden_states]
        if len(hidden_states) != self.num_models:
            raise ValueError(f"Expected {self.num_models} models in hidden_states, got {len(hidden_states)}")

        is_batch = model_logits.ndim == 3
        if not is_batch:
            model_logits = model_logits[np.newaxis, :, :]
            gold_label = np.array([gold_label])

        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32, device=self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long, device=self.device)

        with torch.enable_grad():
            raw_wagers_tensor = self._project_and_route(hidden_states, head_name="wager")
            _, sigmoid_wagers = self._normalize_wagers(raw_wagers_tensor)
            components = self._extract_components(
                sigmoid_wagers=sigmoid_wagers,
                model_logits_tensor=model_logits_tensor,
                gold_label_tensor=gold_label_tensor,
            )
            brs = components["brs"]

            wager_losses = F.mse_loss(sigmoid_wagers, brs, reduction="none")

            score_losses = None
            avg_losses = None
            if self.ablation_study:
                estimated_scores = torch.sigmoid(self._project_and_route(hidden_states, head_name="score"))
                estimated_average_scores = torch.sigmoid(self._project_and_route(hidden_states, head_name="avg"))
                score_losses = F.mse_loss(estimated_scores, components["scores"], reduction="none")
                avg_losses = F.mse_loss(
                    estimated_average_scores,
                    components["average_scores"],
                    reduction="none",
                )

            total_loss = 0.0
            for model_idx in range(self.num_models):
                losses_by_head = [("wager", wager_losses[:, model_idx].mean())]
                if self.ablation_study:
                    losses_by_head.append(("score", score_losses[:, model_idx].mean()))
                    losses_by_head.append(("avg", avg_losses[:, model_idx].mean()))

                for loss_pos, (head_name, loss_tensor) in enumerate(losses_by_head):
                    optimizer_i, scheduler_i = self._get_optimizer_and_scheduler(model_idx, head_name)
                    optimizer_i.zero_grad()
                    params_i = self._get_head_params(model_idx, head_name)

                    retain_graph = not (
                        model_idx == self.num_models - 1 and loss_pos == len(losses_by_head) - 1
                    )
                    grads = torch.autograd.grad(
                        loss_tensor,
                        params_i,
                        retain_graph=retain_graph,
                        allow_unused=True,
                    )

                    for param, grad in zip(params_i, grads):
                        if grad is not None:
                            param.grad = grad

                    torch.nn.utils.clip_grad_norm_(params_i, self.grad_clip_norm)
                    optimizer_i.step()
                    if scheduler_i is not None:
                        scheduler_i.step()

                    total_loss += float(loss_tensor.detach().cpu().numpy())

        num_losses_per_model = 3 if self.ablation_study else 1
        return {"loss": total_loss / (self.num_models * num_losses_per_model)}

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        params = []
        params.extend(self.wager_routers.parameters())
        for proj in self.wager_projections.values():
            params.extend(proj.parameters())
        if self.ablation_study:
            params.extend(self.estimate_score_routers.parameters())
            params.extend(self.estimate_average_score_routers.parameters())
            for proj in self.estimate_score_projections.values():
                params.extend(proj.parameters())
            for proj in self.estimate_average_score_projections.values():
                params.extend(proj.parameters())
        return params

    def train_mode(self):
        for router in self.wager_routers:
            router.train()
        for router in self.estimate_score_routers:
            router.train()
        for router in self.estimate_average_score_routers:
            router.train()
        self._training = True

    def eval_mode(self):
        for router in self.wager_routers:
            router.eval()
        for router in self.estimate_score_routers:
            router.eval()
        for router in self.estimate_average_score_routers:
            router.eval()
        self._training = False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "wager_routers_state_dict": {
                f"router_{i}": router.state_dict() for i, router in enumerate(self.wager_routers)
            },
            "wager_projections_state_dict": {
                k: v.state_dict() for k, v in self.wager_projections.items()
            },
            "estimate_score_routers_state_dict": {
                f"router_{i}": router.state_dict() for i, router in enumerate(self.estimate_score_routers)
            },
            "estimate_score_projections_state_dict": {
                k: v.state_dict() for k, v in self.estimate_score_projections.items()
            },
            "estimate_average_score_routers_state_dict": {
                f"router_{i}": router.state_dict() for i, router in enumerate(self.estimate_average_score_routers)
            },
            "estimate_average_score_projections_state_dict": {
                k: v.state_dict() for k, v in self.estimate_average_score_projections.items()
            },
            "optimizers_state_dict": {
                f"optimizer_{model_idx}_{head_name}": opt.state_dict()
                for model_idx, head_name, opt in self.optimizers
            },
            "config": {
                "common_hidden_dim": self.common_hidden_dim,
                "hidden_layers": self.hidden_layers,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "normalize_hidden_states": self.normalize_hidden_states,
                "hidden_state_layers": self.hidden_state_layers,
                "score_function": self.score_function_name,
                "device": self.device_str,
                "lr_decay_factor": self.lr_decay_factor,
                "lr_decay_steps": self.lr_decay_steps,
                "ablation_study": self.ablation_study,
            },
        }

    def _load_projection_state_dict(
        self,
        projection_store: nn.ModuleDict,
        projection_state_dicts: Dict[str, Dict[str, torch.Tensor]],
    ):
        for key, proj_state in projection_state_dicts.items():
            if key not in projection_store:
                in_features = proj_state["weight"].shape[1]
                out_features = proj_state["weight"].shape[0]
                projection_store[key] = nn.Linear(in_features, out_features).to(self.device)
            projection_store[key].load_state_dict(proj_state)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for i, router in enumerate(self.wager_routers):
            key = f"router_{i}"
            if key in state_dict.get("wager_routers_state_dict", {}):
                router.load_state_dict(state_dict["wager_routers_state_dict"][key])

        for i, router in enumerate(self.estimate_score_routers):
            key = f"router_{i}"
            if key in state_dict.get("estimate_score_routers_state_dict", {}):
                router.load_state_dict(state_dict["estimate_score_routers_state_dict"][key])

        for i, router in enumerate(self.estimate_average_score_routers):
            key = f"router_{i}"
            if key in state_dict.get("estimate_average_score_routers_state_dict", {}):
                router.load_state_dict(state_dict["estimate_average_score_routers_state_dict"][key])

        self._load_projection_state_dict(
            self.wager_projections,
            state_dict.get("wager_projections_state_dict", {}),
        )
        self._load_projection_state_dict(
            self.estimate_score_projections,
            state_dict.get("estimate_score_projections_state_dict", {}),
        )
        self._load_projection_state_dict(
            self.estimate_average_score_projections,
            state_dict.get("estimate_average_score_projections_state_dict", {}),
        )

        self.optimizers = []
        self.schedulers = []
        for model_idx in range(self.num_models):
            head_names = ["wager"]
            if self.ablation_study:
                head_names.extend(["score", "avg"])
            for head_name in head_names:
                # Ensure optimizer/scheduler exists for each loaded head.
                if head_name == "wager":
                    proj_key = f"wager_proj_{model_idx}"
                    if proj_key not in self.wager_projections:
                        continue
                    params = list(self.wager_routers[model_idx].parameters()) + list(
                        self.wager_projections[proj_key].parameters()
                    )
                elif head_name == "score":
                    proj_key = f"score_proj_{model_idx}"
                    if proj_key not in self.estimate_score_projections:
                        continue
                    params = list(self.estimate_score_routers[model_idx].parameters()) + list(
                        self.estimate_score_projections[proj_key].parameters()
                    )
                else:
                    proj_key = f"avg_proj_{model_idx}"
                    if proj_key not in self.estimate_average_score_projections:
                        continue
                    params = list(self.estimate_average_score_routers[model_idx].parameters()) + list(
                        self.estimate_average_score_projections[proj_key].parameters()
                    )

                optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.lr_decay_steps,
                    gamma=self.lr_decay_factor,
                )
                opt_key = f"optimizer_{model_idx}_{head_name}"
                if opt_key in state_dict.get("optimizers_state_dict", {}):
                    try:
                        optimizer.load_state_dict(state_dict["optimizers_state_dict"][opt_key])
                    except Exception:
                        pass
                self.optimizers.append((model_idx, head_name, optimizer))
                self.schedulers.append((model_idx, head_name, scheduler))
