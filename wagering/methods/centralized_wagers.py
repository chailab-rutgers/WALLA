"""
Centralized wagers implementation: uses an MLP router to generate wagers from hidden states.
"""

import numpy as np
import torch
import torch.nn as nn
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


class CentralizedWagers(WageringMethod):
    """
    Wagering method that uses an MLP router to generate wagers from hidden states.
    
    The router takes the last hidden states from all LLMs as input and outputs
    a probability distribution over models via softmax.
    """
    
    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the centralized wagers method.
        
        Args:
            num_models: Number of LLMs in the ensemble
            config: Configuration dictionary with:
                - hidden_dim: Dimension of hidden states (default: 4096)
                - hidden_layers: List of hidden layer sizes (default: [512, 256])
                - learning_rate: Learning rate for optimizer (default: 1e-4)
                - device: Device to run on (default: 'cuda' if available, else 'cpu')
        """
        super().__init__(num_models, config)
        
        # Get configuration (ensure proper types)
        self.hidden_dim = int(config.get("hidden_dim", 4096))
        self.hidden_layers = list(config.get("hidden_layers", [512, 256]))
        self.learning_rate = float(config.get("learning_rate", 1e-5))  # Lower default LR: 1e-5 instead of 1e-4
        self.temperature = float(config.get("temperature", 2.0))  # Temperature for softmax (higher = softer)
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))  # Gradient clipping norm
        self.normalize_hidden_states = config.get("normalize_hidden_states", True)  # L2 normalize hidden states
        self.device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)
        self.hidden_state_layers = config.get("hidden_state_layers", [-1])
        self.hidden_state_layers_per_model = config.get("hidden_state_layers_per_model")
        
        # Build per-model projection layers to handle variable hidden dimensions
        # These will be created dynamically when we first see the hidden states
        self.model_projections = nn.ModuleDict()
        self._model_hidden_dims = {}  # Track each model's hidden dim
        
        # Build MLP router
        # Input: concatenated hidden states from all models (after projection)
        # Output: logits for each model (before softmax)
        input_dim = num_models * self.hidden_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: outputs logits for num_models
        layers.append(nn.Linear(prev_dim, num_models))
        
        self.router = nn.Sequential(*layers).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.router.parameters(), lr=self.learning_rate) #betas=(0.0, 0.9),
        
        # Training mode flag
        self._training = True
        
        # Cache for computed values during training (to avoid recomputation in update())
        # These are set by compute_wagers() and cleared by update()
        self._cached_wagers: Optional[torch.Tensor] = None
        self._cached_hidden_states_flat: Optional[torch.Tensor] = None
    
    def compute_wagers(
        self,
        hidden_states_list: List[np.ndarray],
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute wagers for a batch of samples with heterogeneous (variable) hidden dimensions per model.
        
        For centralized wagers, this projects each model's states then concatenates them before routing.
        
        Args:
            hidden_states_list: List of [num_models] where each element is np.ndarray of shape [batch_size, hidden_dim_i]
            model_logits: Optional, not used
            gold_label: Optional, ground-truth labels (used by methods that compute label-aware diagnostics)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with:
                "wagers": np.ndarray of shape [batch_size, num_models] with probabilities (sum to 1 for each row)
                "nash_gap": Optional float or np.ndarray of shape [batch_size] (if computed)
        """
        if len(hidden_states_list) != self.num_models:
            raise ValueError(
                f"Expected {self.num_models} models, got {len(hidden_states_list)}"
            )
        
        batch_size = hidden_states_list[0].shape[0]
        
        # Step 1: Project each model's hidden states in batch
        projected_batch_list = []
        
        for i in range(self.num_models):
            model_hs_batch = hidden_states_list[i]  # [batch_size, hidden_dim_i]
            model_hidden_dim = model_hs_batch.shape[-1]
            
            # Create projection layer if needed
            proj_key = f"proj_{i}"
            if proj_key not in self.model_projections:
                projection = nn.Linear(model_hidden_dim, self.hidden_dim).to(self.device)
                self.model_projections[proj_key] = projection
                if self._training:
                    self.optimizer.add_param_group({'params': projection.parameters()})
            
            projection = self.model_projections[proj_key]
            
            # Convert to tensor and project
            model_hs_tensor = torch.as_tensor(model_hs_batch, dtype=torch.float32).to(self.device)
            
            if self.normalize_hidden_states:
                norms = torch.norm(model_hs_tensor, dim=1, keepdim=True)
                model_hs_tensor = model_hs_tensor / (norms + 1e-8)
            
            with torch.set_grad_enabled(self._training):
                projected_batch = projection(model_hs_tensor)
            
            projected_batch_list.append(projected_batch)
        
        # Step 2: Concatenate all projected states
        batch_hidden_states_cat = torch.cat(projected_batch_list, dim=1)  # [batch_size, num_models * common_hidden_dim]
        
        # Step 3: Route through centralized router
        self.router.eval() if not self._training else self.router.train()
        with torch.set_grad_enabled(self._training):
            logits = self.router(batch_hidden_states_cat)  # [batch_size, num_models]
            wagers = torch.softmax(logits / self.temperature, dim=1)  # [batch_size, num_models]
        
        if self._training:
            self._cached_wagers = wagers
            self._cached_hidden_states_flat = batch_hidden_states_cat
        
        wagers_np = wagers.detach().cpu().numpy()
        return {"wagers": wagers_np}

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
        """Update for a batch of samples."""
        batch_size = model_logits.shape[0]
        
        # Reuse cached values from compute_wagers() if available
        if (self._cached_wagers is not None and 
            self._cached_hidden_states_flat is not None):
            wagers = self._cached_wagers  # [batch_size, num_models]
            hidden_states_flat = self._cached_hidden_states_flat
        else:
            raise ValueError(
                "CentralizedWagers.update() requires cached wagers from compute_wagers(). "
                "Please ensure compute_wagers() is called before update()."
            )
        
        # Clear cache after use
        self._cached_wagers = None
        self._cached_hidden_states_flat = None
        
        # Convert model_logits to tensor: [batch_size, num_models, num_options]
        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32).to(self.device)
        
        # Compute aggregated probabilities for entire batch
        from wagering.aggregation.linear_pooling import LinearPooling
        # LinearPooling.aggregate_torch handles batch: [batch_size, num_models, num_options] + [batch_size, num_models]
        batch_aggregated_probs = LinearPooling.aggregate_torch(
            model_logits_tensor, wagers
        )  # [batch_size, num_options]
        
        gold_label_distribution = kwargs.get("gold_label_distribution", None)
        if gold_label_distribution is not None:
            # Expected cross-entropy under a soft-label distribution q:
            #   E_{y~q}[-log p(y)] = -sum_k q_k log p_k
            gold_label_distribution_tensor = torch.as_tensor(
                gold_label_distribution, dtype=torch.float32, device=self.device
            )
            if (
                gold_label_distribution_tensor.ndim != 2
                or gold_label_distribution_tensor.shape[0] != batch_size
                or gold_label_distribution_tensor.shape[1] != batch_aggregated_probs.shape[1]
            ):
                raise ValueError(
                    "gold_label_distribution must be shape [batch_size, num_options], "
                    f"got {tuple(gold_label_distribution_tensor.shape)}"
                )
            log_probs = torch.log(batch_aggregated_probs + 1e-10)
            loss = -torch.mean(torch.sum(gold_label_distribution_tensor * log_probs, dim=-1))
        else:
            # Standard hard-label cross-entropy.
            gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long).to(self.device)
            batch_indices = torch.arange(batch_size, device=self.device)
            probs_at_gold = batch_aggregated_probs[batch_indices, gold_label_tensor]
            loss = -torch.mean(torch.log(probs_at_gold + 1e-10))
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.router.parameters(), self.grad_clip_norm)
        if len(self.model_projections) > 0:
            proj_params = [p for proj in self.model_projections.values() for p in proj.parameters()]
            torch.nn.utils.clip_grad_norm_(proj_params, self.grad_clip_norm)
        self.optimizer.step()
        
        # Compute batch metrics using computed aggregated probs (not the passed-in parameter)
        batch_aggregated_probs_np = batch_aggregated_probs.detach().cpu().numpy()
        batch_correct = (np.argmax(batch_aggregated_probs_np, axis=1) == gold_label)
        batch_accuracy = float(np.mean(batch_correct))
        avg_prob_correct = float(np.mean(batch_aggregated_probs_np[np.arange(batch_size), gold_label]))
        
        return {
            "loss": float(loss.item()),
            "batch_accuracy": batch_accuracy,
            "avg_prob_correct": avg_prob_correct,
            "batch_size": batch_size,
        }
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters (router + projections)."""
        params = list(self.router.parameters())
        for proj in self.model_projections.values():
            params.extend(proj.parameters())
        return params
    
    def train_mode(self):
        """Set the method to training mode."""
        self.router.train()
        self._training = True
        # Clear cache when switching modes
        self._cached_wagers = None
        self._cached_hidden_states_flat = None
    
    def eval_mode(self):
        """Set the method to evaluation mode."""
        self.router.eval()
        self._training = False
        # Clear cache when switching modes
        self._cached_wagers = None
        self._cached_hidden_states_flat = None
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        return {
            "router_state_dict": self.router.state_dict(),
            "model_projections_state_dict": {k: v.state_dict() for k, v in self.model_projections.items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "hidden_dim": self.hidden_dim,
                "hidden_layers": self.hidden_layers,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "normalize_hidden_states": self.normalize_hidden_states,
                "device": self.device_str,
            },
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary from checkpoint."""
        if "router_state_dict" in state_dict:
            self.router.load_state_dict(state_dict["router_state_dict"])
        if "model_projections_state_dict" in state_dict:
            for key, proj_state in state_dict["model_projections_state_dict"].items():
                if key not in self.model_projections:
                    # Create projection layer from checkpoint shape
                    in_features = proj_state["weight"].shape[1]
                    out_features = proj_state["weight"].shape[0]
                    projection = nn.Linear(in_features, out_features).to(self.device)
                    self.model_projections[key] = projection
                    if self._training:
                        self.optimizer.add_param_group({'params': projection.parameters()})
                self.model_projections[key].load_state_dict(proj_state)
        if "optimizer_state_dict" in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                # Optimizer state dict may not match if projection layers were added/removed
                # This is acceptable - we'll continue with a fresh optimizer
                import logging
                log = logging.getLogger("wagering")
                log.warning(
                    f"Could not load optimizer state dict (parameter mismatch): {e}. "
                    "Continuing with fresh optimizer state."
                )

