"""
Weighted score wagers implementation: each model has its own router that outputs a single scalar wager.
Wagers are normalized via softmax to sum to 1.
"""

import numpy as np
from pytest import raises
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure local project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from .base import WageringMethod
from wagering.core.model import WhiteboxModel
from wagering.aggregation.linear_pooling import LinearPooling

class MSEBrWagers(WageringMethod):
    """
    Wagering method where each model has its own router that outputs a single scalar wager.
    
    Architecture:
    - Each model's hidden state is projected to a common dimension
    - Each model has its own router (MLP) that outputs a single scalar (raw wager logit)
    - Raw wagers are collected and normalized via softmax to sum to 1
    
    Router Architecture:
    - Input: Single model's projected hidden state [common_hidden_dim]
    - Output: Single scalar (raw wager logit)
    - Structure: MLP(hidden_dim → hidden_layers → 1)
    """
    
    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the weighted score wagers method.
        
        Args:
            num_models: Number of LLMs in the ensemble
            config: Configuration dictionary with:
                - common_hidden_dim: Common dimension for projected hidden states (default: 4096)
                - hidden_layers: List of hidden layer sizes (default: [512, 256])
                - learning_rate: Learning rate for optimizer (default: 1e-5)
                - device: Device to run on (default: 'cuda' if available, else 'cpu')
                -score_function: Scoring function name ('linear', 'log', 'brier', 'normalized_linear') (default: 'linear')
        """
        super().__init__(num_models, config)
        
        # Get configuration (ensure proper types)
        self.common_hidden_dim = int(config.get("common_hidden_dim", 4096))
        self.hidden_layers = list(config.get("hidden_layers", [512, 256]))
        self.learning_rate = float(config.get("learning_rate", 1e-5))  # Lower default LR: 1e-5 instead of 1e-4
        self.temperature = float(config.get("temperature", 2.0))  # Temperature for softmax (higher = softer)
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))  # Gradient clipping norm
        self.normalize_hidden_states = config.get("normalize_hidden_states", True)  # L2 normalize hidden states
        self.score_function_name = str(config.get("score_function", "normalized_linear"))  # Scoring function: 'linear', 'log', or 'brier'
        self.device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)
        self.frozen_model_indices = {
            int(idx) for idx in config.get("frozen_model_indices", [])
        }
        self.inactive_model_indices = {
            int(idx) for idx in config.get("inactive_model_indices", [])
        }
        for idx in self.frozen_model_indices.union(self.inactive_model_indices):
            if idx < 0 or idx >= num_models:
                raise ValueError(
                    f"Model index {idx} is out of range for num_models={num_models}"
                )
        
        # Initialize scoring function
        
        # Build per-model routers and projections
        # Each router takes a single model's projected hidden state and outputs a scalar
        self.routers = nn.ModuleList()
        self.model_projections = nn.ModuleDict()
        self.optimizers = []
        
        for i in range(num_models):
            router = self._build_router().to(self.device)
            self.routers.append(router)
        
        # Projections will be created dynamically and added to their corresponding optimizers
        
        # Training mode flag
        self._training = True
        
        # Cache for computed values during training (to avoid recomputation in update())
        # These are set by compute_wagers() and cleared by update()
        self._cached_wagers: Optional[torch.Tensor] = None
        self._cached_projected_states: Optional[List[torch.Tensor]] = None
        self._cached_hidden_states_list: Optional[List[torch.Tensor]] = None

    def _is_model_trainable(self, model_idx: int) -> bool:
        return (
            model_idx not in self.frozen_model_indices
            and model_idx not in self.inactive_model_indices
        )
    
    def _build_router(self) -> nn.Module:
        """
        Build a single router MLP.
        
        Returns:
            nn.Sequential: Router network that maps [common_hidden_dim] -> [1]
        """
        layers = []
        prev_dim = self.common_hidden_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: outputs single scalar
        output_layer = nn.Linear(prev_dim, 1)
        # Initialize bias to 0.0 so sigmoid(0) = 0.5, preventing all-negative wagers initially
        # nn.init.constant_(output_layer.bias, 0.0)
        layers.append(output_layer)
        
        return nn.Sequential(*layers)

    def compute_wagers(
        self,
        hidden_states_list: List[np.ndarray],
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute wagers for heterogeneous batch (variable hidden dimensions per model).
        
        Args:
            hidden_states_list: List of [num_models] where each is [batch_size, hidden_dim_i]
            model_logits: Optional, not used
            gold_label: Optional, gold labels for computing BRs and Nash gap
            **kwargs: Additional keyword arguments
            
        Returns:
            wagers: [batch_size, num_models]
        """
        if len(hidden_states_list) != self.num_models:
            raise ValueError(f"Expected {self.num_models} models, got {len(hidden_states_list)}")
        
        batch_size = hidden_states_list[0].shape[0]
        projected_batch_list = []
        
        # Project each model's batch
        for i in range(self.num_models):
            model_hs_batch = hidden_states_list[i]
            model_hidden_dim = model_hs_batch.shape[-1]
            
            proj_key = f"proj_{i}"
            if proj_key not in self.model_projections:
                projection = nn.Linear(model_hidden_dim, self.common_hidden_dim).to(self.device)
                self.model_projections[proj_key] = projection
                if self._training:
                    # Create optimizer for this model (router i + projection i)
                    params = list(self.routers[i].parameters()) + list(projection.parameters())
                    optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                    self.optimizers.append((i, optimizer))
            
            projection = self.model_projections[proj_key]
            # Convert to tensor (no requires_grad - hidden states are inputs, not learnable)
            model_hs_tensor = torch.as_tensor(model_hs_batch, dtype=torch.float32, device=self.device)
            
            if self.normalize_hidden_states:
                norms = torch.norm(model_hs_tensor, dim=1, keepdim=True)
                model_hs_tensor = model_hs_tensor / (norms + 1e-8)
            
            with torch.set_grad_enabled(self._training):
                projected_batch = projection(model_hs_tensor)
            
            projected_batch_list.append(projected_batch)
        
        # Route: compute raw wagers for each router
        raw_wagers_list = []
        for i in range(self.num_models):
            model_projected = projected_batch_list[i]
            router_i = self.routers[i]
            raw_wager_i = router_i(model_projected)
            raw_wagers_list.append(raw_wager_i)
        
        # Normalize
        raw_wagers_tensor = torch.cat(raw_wagers_list, dim=1)
        sigmoid_wagers = torch.sigmoid(raw_wagers_tensor/self.temperature)
        sigmoid_wagers = torch.clamp(sigmoid_wagers, min=1e-10, max=1.0-1e-10)  # Prevent zero wagers which can cause issues with the loss
        if len(self.inactive_model_indices) > 0:
            inactive_list = sorted(self.inactive_model_indices)
            sigmoid_wagers[:, inactive_list] = 0.0
        # Compute sum for normalization
        sigmoid_sum = torch.sum(sigmoid_wagers, dim=1, keepdim=True)
        
        # Check for near-zero sums and fallback to uniform if needed
        if torch.any(sigmoid_sum < 1e-16):
            raise RuntimeError("Near-zero sigmoid sum detected during compute_wagers(). This should not happen due to clipping.")
        else:
            wagers = sigmoid_wagers / sigmoid_sum
        
        if self._training:
            self._cached_wagers = wagers
            projected_concatenated = torch.cat(projected_batch_list, dim=1)
            self._cached_projected_states = projected_concatenated
            # Cache hidden states with requires_grad=True so gradients can flow through them
            # (they won't be updated, but they need to be in the computation graph)
            self._cached_hidden_states_list = [
                torch.as_tensor(hidden_states_list[i], dtype=torch.float32, device=self.device)
                    .requires_grad_(True)
                for i in range(self.num_models)
            ]
        is_batch = model_logits.ndim == 3
        if not is_batch:
            model_logits = model_logits[np.newaxis, :, :]
            gold_label = np.array([gold_label])
        
        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32).to(self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long).to(self.device)
        brs, nash_gap, score_diff, total_payout = self.extract_wagers_brs_and_nash_gap(sigmoid_wagers, model_logits_tensor, gold_label_tensor)  # This will populate the cache with BRs and Nash gaps for the current wagers
        # Convert to numpy and validate
        wagers_np = wagers.detach().cpu().numpy()
        nash_gap = nash_gap.detach().cpu().numpy()
        score_diff = score_diff.detach().cpu().numpy()
        total_payout = total_payout.detach().cpu().numpy()
        # Check for NaN or inf
        if np.any(np.isnan(wagers_np)) or np.any(np.isinf(wagers_np)):
            import sys
            print(f"WARNING: Invalid wagers detected (NaN or inf). Resetting to uniform.", file=sys.stderr)
            # Fallback to uniform wagers
            raise ValueError("Invalid wagers detected (NaN or inf).")
        
        return {"wagers": wagers_np, "nash_gap": nash_gap, "score_diff": score_diff, "total_payout": total_payout}
    
    def extract_wagers_brs_and_nash_gap(self, sigmoid_wagers, model_logits_tensor, gold_label_tensor):
        
        # def score_function(ground_truth: torch.Tensor, predictions: torch.Tensor, wagers: torch.Tensor = None) -> torch.Tensor:
        probs = F.softmax(model_logits_tensor, dim=-1)  # [batch_size, num_models, num_options]

        batch_size, num_models, num_options = probs.shape
        
        # Create one-hot encoding of ground truth
        # ground_truth: [batch_size] -> [batch_size, num_options]
        gt_onehot = F.one_hot(gold_label_tensor, num_classes=num_options).float()  # [batch_size, num_options]
        
        # Expand to match predictions: [batch_size, 1, num_options] -> [batch_size, num_models, num_options]
        gt_onehot_expanded = gt_onehot.unsqueeze(1).expand(batch_size, num_models, num_options)
        
        # Compute squared error: [batch_size, num_models, num_options]
        squared_errors = (probs - gt_onehot_expanded) ** 2
        
        # Sum over options (Brier score): [batch_size, num_models]
        brier_scores = squared_errors.sum(dim=-1)
        
        # Return negative (since lower Brier score is better, but we want higher scores to be better)
        scores = 0.5 * (2-brier_scores - sigmoid_wagers)  # Custom modification
            
        average_scores = ((scores * sigmoid_wagers).sum(dim=1, keepdim=True).expand_as(scores * sigmoid_wagers)
                            - (scores * sigmoid_wagers)) / (sigmoid_wagers.sum(dim=1, keepdim=True).expand_as(sigmoid_wagers) - sigmoid_wagers)
        # average_scores = brier_scores.sum(dim=1, keepdim=True)/(brier_scores.shape[1] - 1) - brier_scores
        brs = 0.5 * (2 - brier_scores) - average_scores
        # brs = average_scores
        brs = torch.clamp(brs, max=1.0-1e-10, min=1e-10)  # Prevent zero or negative BRs which can cause issues with the loss

        br_scores = 0.5 * (2-brier_scores - brs)
        total_payout = sigmoid_wagers * (scores - average_scores - 0.5 * sigmoid_wagers)
        nash_gap = brs * (br_scores - average_scores) - sigmoid_wagers * (scores - average_scores)
        score_diff = scores - average_scores
        return brs, nash_gap, score_diff, total_payout

    def update(
        self,
        aggregated_probs: np.ndarray,
        aggregated_pred: np.ndarray,
        gold_label: np.ndarray,
        model_probs: np.ndarray,
        model_logits: np.ndarray,
        question: Optional[str] = None,
        hidden_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update the routers by maximizing per-model scores using the configured scoring function.
        
        Recomputes forward pass with gradient tracking for independent router updates.
        Supports single samples ([num_models, hidden_dim]) and batches ([batch_size, num_models, ...]).
        
        Args:
            aggregated_probs: [batch_size, num_options] or [num_options]
            aggregated_pred: [batch_size] or scalar
            gold_label: [batch_size] or scalar
            model_probs: [batch_size, num_models, num_options] or [num_models, num_options]
            model_logits: [batch_size, num_models, num_options] or [num_models, num_options]
            question: Optional
            hidden_states: List of [num_models] where each is [batch_size, hidden_dim_i] or list of numpy arrays
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with loss
        """
        if not self._training:
            return {}
        
        # Validate hidden_states are provided
        if hidden_states is None:
            raise ValueError("hidden_states must be provided to update()")
        
        if not isinstance(hidden_states, (list, tuple)):
            hidden_states = [hidden_states]
        
        if len(hidden_states) != self.num_models:
            raise ValueError(f"Expected {self.num_models} models in hidden_states, got {len(hidden_states)}")
        
        # Clear cache
        self._cached_wagers = None
        self._cached_projected_states = None
        self._cached_hidden_states_list = None
        
        # Normalize dimensions
        is_batch = model_logits.ndim == 3
        if not is_batch:
            model_logits = model_logits[np.newaxis, :, :]
            gold_label = np.array([gold_label])
        
        batch_size = model_logits.shape[0]
        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32).to(self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long).to(self.device)
        
        # Convert hidden states to tensors with requires_grad=True for gradient flow
        hidden_states_tensors = []
        for i in range(self.num_models):
            hs = hidden_states[i]
            if isinstance(hs, np.ndarray):
                hs_tensor = torch.as_tensor(hs, dtype=torch.float32, device=self.device)
            else:
                hs_tensor = hs.to(self.device)
            hs_tensor.requires_grad_(True)
            hidden_states_tensors.append(hs_tensor)
        
        # PHASE 1: Compute forward pass ONCE from original state (with all intermediate tensors)
        # This ensures all N losses are computed from the same parameter state
        with torch.enable_grad():
            raw_wagers_list = []
            for j in range(self.num_models):
                model_hs_j = hidden_states_tensors[j]
                
                # Apply projection if exists
                proj_key = f"proj_{j}"
                if proj_key in self.model_projections:
                    projection_j = self.model_projections[proj_key]
                    model_projected_j = projection_j(model_hs_j)
                else:
                    model_projected_j = model_hs_j
                
                # Normalize if needed
                if self.normalize_hidden_states:
                    norms = torch.norm(model_projected_j, dim=-1, keepdim=True)
                    model_projected_j = model_projected_j / (norms + 1e-8)
                
                # Apply router
                raw_wager_j = self.routers[j](model_projected_j)
                raw_wagers_list.append(raw_wager_j)
            
            # Normalize wagers (single computation for all)
            raw_wagers_tensor = torch.cat(raw_wagers_list, dim=1)
            sigmoid_wagers = torch.sigmoid(raw_wagers_tensor / self.temperature) # [batch_size, num_models]
            sigmoid_wagers = torch.clamp(sigmoid_wagers, min=1e-10, max=1.0-1e-10)  # Prevent zero wagers which can cause issues with the loss
            if len(self.inactive_model_indices) > 0:
                inactive_list = sorted(self.inactive_model_indices)
                sigmoid_wagers[:, inactive_list] = 0.0

            # scores = ((scores - #.detach()

            # net_payout = sigmoid_wagers * scores
            # 0/0
            # batch_size = model_logits.shape[0]
            # model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32).to(self.device)
            # gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long).to(self.device)
            
            # # Aggregate and compute loss (batch processing)
            # aggregated_probs_torch = LinearPooling.aggregate_torch(
            #     model_logits_tensor, wagers_all
            # )  # [batch_size, num_options]
            brs, nash_gap, score_diff, _ = self.extract_wagers_brs_and_nash_gap(sigmoid_wagers, model_logits_tensor, gold_label_tensor)
            # losses = -torch.log(aggregated_probs_torch[torch.arange(batch_size), gold_label_tensor] + 1e-10) # [batch_size]
            mseloss = F.mse_loss(sigmoid_wagers, brs, reduction='none')#.mean(dim=1)
            # Compute all N losses from the same forward pass tensors
            all_losses = []
            for i in range(self.num_models):
                # Loss for router i: maximize its score (computed from same wagers_all)
                loss_i = mseloss[:, i].mean() # losses.mean()  #
                all_losses.append(loss_i)
            
            # PHASE 2: Update each router independently using manual gradient computation
            # This avoids in-place operation conflicts with retain_graph=True
            total_loss = 0.0
            num_updated_models = 0
            for i in range(self.num_models):
                if not self._is_model_trainable(i):
                    continue

                # Find optimizer for model i
                optimizer_i = None
                for model_idx, opt in self.optimizers:
                    if model_idx == i:
                        optimizer_i = opt
                        break
                
                if optimizer_i is None:
                    raise RuntimeError(f"No optimizer found for model {i}")
                
                # Zero optimizer for model i
                optimizer_i.zero_grad()
                
                # Collect parameters for model i (router + projection)
                params_i = list(self.routers[i].parameters())
                proj_key_i = f"proj_{i}"
                if proj_key_i in self.model_projections:
                    params_i += list(self.model_projections[proj_key_i].parameters())
                
                # Manual gradient computation: retain graph for all but the last
                retain = (i < self.num_models - 1)
                grads = torch.autograd.grad(
                    all_losses[i],
                    params_i,
                    retain_graph=retain,
                    allow_unused=True
                )
                
                # Manually assign gradients and verify they exist
                grads_assigned = 0
                for param, grad in zip(params_i, grads):
                    if grad is not None:
                        param.grad = grad
                        grads_assigned += 1
                
                # Sanity check: ensure we got gradients for all parameters
                if grads_assigned == 0:
                    import sys
                    print(f"WARNING: Model {i} received no gradients! This may indicate a gradient flow issue.", 
                          file=sys.stderr)
                
                # Clip gradients for router i and projection i
                torch.nn.utils.clip_grad_norm_(self.routers[i].parameters(), self.grad_clip_norm)
                if proj_key_i in self.model_projections:
                    torch.nn.utils.clip_grad_norm_(self.model_projections[proj_key_i].parameters(), self.grad_clip_norm)
                
                # Step: update only model i (router i + projection i)
                optimizer_i.step()
                
                total_loss += float(all_losses[i].detach().cpu().numpy())
                num_updated_models += 1

            if num_updated_models == 0:
                raise RuntimeError(
                    "No trainable models are active. Check frozen_model_indices/inactive_model_indices configuration."
                )

        # Return average loss across all routers
        return {"loss": total_loss / num_updated_models}

    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters (all routers + projections)."""
        params = list(self.routers.parameters())
        for proj in self.model_projections.values():
            params.extend(proj.parameters())
        return params
    
    def train_mode(self):
        """Set the method to training mode."""
        for router in self.routers:
            router.train()
        self._training = True
        # Clear cache when switching modes
        self._cached_wagers = None
        self._cached_projected_states = None
        self._cached_hidden_states_list = None
    
    def eval_mode(self):
        """Set the method to evaluation mode."""
        for router in self.routers:
            router.eval()
        self._training = False
        # Clear cache when switching modes
        self._cached_wagers = None
        self._cached_projected_states = None
        self._cached_hidden_states_list = None
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        routers_state_dict = {f"router_{i}": router.state_dict() for i, router in enumerate(self.routers)}
        projections_state_dict = {k: v.state_dict() for k, v in self.model_projections.items()}
        optimizers_state_dict = {f"optimizer_{model_idx}": opt.state_dict() for model_idx, opt in self.optimizers}
        return {
            "routers_state_dict": routers_state_dict,
            "model_projections_state_dict": projections_state_dict,
            "optimizers_state_dict": optimizers_state_dict,
            "config": {
                "common_hidden_dim": self.common_hidden_dim,
                "hidden_layers": self.hidden_layers,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "normalize_hidden_states": self.normalize_hidden_states,
                "score_function": self.score_function_name,
                "device": self.device_str,
                "frozen_model_indices": sorted(self.frozen_model_indices),
                "inactive_model_indices": sorted(self.inactive_model_indices),
            },
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary from checkpoint."""
        if "routers_state_dict" in state_dict:
            routers_state_dict = state_dict["routers_state_dict"]
            for i, router in enumerate(self.routers):
                key = f"router_{i}"
                if key in routers_state_dict:
                    router.load_state_dict(routers_state_dict[key])
        if "model_projections_state_dict" in state_dict:
            for key, proj_state in state_dict["model_projections_state_dict"].items():
                if key not in self.model_projections:
                    in_features = proj_state["weight"].shape[1]
                    out_features = proj_state["weight"].shape[0]
                    projection = nn.Linear(in_features, out_features).to(self.device)
                    self.model_projections[key] = projection
                    if self._training:
                        try:
                            model_idx = int(key.split("_")[1])
                            params = list(self.routers[model_idx].parameters()) + list(projection.parameters())
                            optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                            self.optimizers.append((model_idx, optimizer))
                        except Exception:
                            pass
                self.model_projections[key].load_state_dict(proj_state)
        if "optimizers_state_dict" in state_dict:
            optimizers_state_dict = state_dict["optimizers_state_dict"]
            for model_idx, optimizer in self.optimizers:
                key = f"optimizer_{model_idx}"
                if key in optimizers_state_dict:
                    try:
                        optimizer.load_state_dict(optimizers_state_dict[key])
                    except (ValueError, KeyError) as e:
                        import logging
                        log = logging.getLogger("wagering")
                        log.warning(
                            f"Could not load optimizer {model_idx} state dict (parameter mismatch): {e}. "
                            "Continuing with fresh optimizer state."
                        )

