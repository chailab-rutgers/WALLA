"""
RouteLLM-style BERT router for multi-LLM wagering.

Implements the BERT encoder + linear routing head described in RouteLLM
(Ong et al., arXiv:2406.18665): encode the prompt with BERT, map [CLS] (or pooled)
representation to logits over experts, then softmax with temperature.

When human preference pairs are unavailable, training uses the same pooled
cross-entropy as other trainable routers (see update()).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from transformers import AutoModel, AutoTokenizer

from .base import WageringMethod
from wagering.aggregation.linear_pooling import LinearPooling
from wagering.utils.checkpoint_loading import (
    load_module_state,
    load_optimizer_state,
    load_scheduler_state,
)
from wagering.utils.encoder_batching import (
    encode_questions_per_model_batch,
    encode_transformer_batch,
)


class RouteLLMBertWagers(WageringMethod):
    """
    BERT prompt encoder + linear router logits over models (RouteLLM BERT variant).

    Unlike StackedGeneralization, routing uses only the task prompt text encoded by
    BERT, not LLM forward hidden states.
    """

    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(num_models, config or {})
        cfg = self.config

        self.bert_model_name = str(cfg.get("bert_model_name", "bert-base-uncased"))
        self.max_seq_length = int(cfg.get("max_seq_length", 512))
        self.learning_rate = float(cfg.get("learning_rate", 5e-5))
        self.temperature = float(cfg.get("temperature", 2.0))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
        self.weight_decay = float(cfg.get("weight_decay", 0.01))
        self.freeze_bert = bool(cfg.get("freeze_bert", False))
        # Default to keeping context (caller can disable it explicitly).
        self.pubmedqa_strip_context = bool(cfg.get("pubmedqa_strip_context", False))
        self.debug_router_prompts = bool(cfg.get("debug_router_prompts", False))
        self.router_dropout_p = float(cfg.get("router_dropout", 0.1))
        self.lr_decay_factor = float(cfg.get("lr_decay_factor", 1.0))
        self.lr_decay_steps = int(cfg.get("lr_decay_steps", 100))

        self.device_str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)

        self.requires_hidden_states = False
        self.concat_prompt_embeddings = not self.pubmedqa_strip_context
        self.expects_per_model_router_prompts = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        # Ensure router encoder stays fp32 (trainer uses plain AdamW, no AMP GradScaler).
        self.bert = AutoModel.from_pretrained(
            self.bert_model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        hidden_size = int(self.bert.config.hidden_size)
        router_in_dim = hidden_size * num_models if self.concat_prompt_embeddings else hidden_size
        self.dropout = nn.Dropout(self.router_dropout_p)
        self.router_head = nn.Linear(router_in_dim, num_models).to(self.device)

        if self.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
            self.bert.eval()

        trainable: List[torch.nn.Parameter] = list(self.router_head.parameters())
        if not self.freeze_bert:
            trainable.extend(list(self.bert.parameters()))

        self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=max(1, self.lr_decay_steps),
            gamma=self.lr_decay_factor,
        )

        self._training = True
        self._cached_wagers: Optional[torch.Tensor] = None
        self._debug_logged_once: bool = False

    def _encode_questions_batch(self, questions: List[str]) -> torch.Tensor:
        if self.debug_router_prompts and (not self._debug_logged_once) and len(questions) > 0:
            self._debug_logged_once = True
            q0 = str(questions[0])
            print(
                "[route_llm_bert debug] pubmedqa_strip_context="
                f"{self.pubmedqa_strip_context} concat_prompt_embeddings={self.concat_prompt_embeddings} "
                f"expects_per_model_router_prompts={getattr(self, 'expects_per_model_router_prompts', None)}"
            )
            print(
                "[route_llm_bert debug] raw_q0_has_Context="
                f"{('Context:' in q0)}"
            )
            print("[route_llm_bert debug] raw_q0_head:", q0[:220].replace("\n", "\\n"))
        return encode_transformer_batch(
            questions,
            tokenizer=self.tokenizer,
            model=self.bert,
            device=self.device,
            max_seq_length=self.max_seq_length,
            training=self._training,
            freeze_backbone=self.freeze_bert,
            strip_context=False,
        )

    def _encode_questions_per_model_batch(self, questions_per_model: List[List[str]]) -> torch.Tensor:
        return encode_questions_per_model_batch(
            questions_per_model,
            num_models=self.num_models,
            device=self.device,
            concat_prompt_embeddings=self.concat_prompt_embeddings,
            encode_batch=self._encode_questions_batch,
            method_name="route_llm_bert",
        )

    def compute_wagers(
        self,
        questions: Optional[List[str]] = None,
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        hidden_states_list: Optional[List[np.ndarray]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if questions is None:
            questions = kwargs.get("questions")
        if questions is None:
            raise ValueError(
                "RouteLLMBertWagers.compute_wagers() requires `questions` (batch of prompt strings)."
            )

        questions_per_model = kwargs.get("questions_per_model", None)
        if (
            self.debug_router_prompts
            and (not self._debug_logged_once)
            and questions_per_model is not None
            and isinstance(questions_per_model, list)
            and len(questions_per_model) > 0
            and len(questions_per_model[0]) > 0
        ):
            # Leave the "once" flip to _encode_questions_batch() so we still get the processed text log too.
            print(
                "[route_llm_bert debug] questions_per_model provided: "
                f"M={len(questions_per_model)} B={len(questions_per_model[0])}"
            )
            max_models_to_print = min(len(questions_per_model), 16)
            for mi in range(max_models_to_print):
                p = str(questions_per_model[mi][0])
                print(
                    f"[route_llm_bert debug] model{mi}_q0_has_Context={('Context:' in p)} "
                    f"head={p[:220].replace(chr(10), r'\\n')}"
                )
            if len(questions_per_model) > max_models_to_print:
                print(
                    "[route_llm_bert debug] (skipping remaining models in questions_per_model; "
                    f"printed first {max_models_to_print})"
                )
        if questions_per_model is not None:
            pooled = self._encode_questions_per_model_batch(questions_per_model)
        elif self.concat_prompt_embeddings:
            replicated = [list(questions) for _ in range(self.num_models)]
            pooled = self._encode_questions_per_model_batch(replicated)
        else:
            pooled = self._encode_questions_batch(questions)
        self.router_head.train() if self._training else self.router_head.eval()

        with torch.set_grad_enabled(self._training):
            h = self.dropout(pooled) if self._training else pooled
            logits = self.router_head(h)
            wagers = torch.softmax(logits / self.temperature, dim=1)

        if self._training:
            self._cached_wagers = wagers

        return {"wagers": wagers.detach().cpu().numpy()}

    def update(
        self,
        aggregated_probs: np.ndarray,
        aggregated_pred: np.ndarray,
        gold_label: np.ndarray,
        model_probs: np.ndarray,
        model_logits: np.ndarray,
        question: Optional[str] = None,
        hidden_states: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        batch_size = model_logits.shape[0]
        if self._cached_wagers is None:
            raise ValueError(
                "RouteLLMBertWagers.update() requires cached wagers from compute_wagers() "
                "called in training mode beforehand."
            )

        wagers = self._cached_wagers
        self._cached_wagers = None

        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32, device=self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long, device=self.device)

        batch_aggregated_probs = LinearPooling.aggregate_torch(model_logits_tensor, wagers)
        gold_label_distribution = kwargs.get("gold_label_distribution", None)
        if gold_label_distribution is not None:
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
            ce_loss = -torch.mean(torch.sum(gold_label_distribution_tensor * log_probs, dim=-1))
        else:
            batch_indices = torch.arange(batch_size, device=self.device)
            probs_at_gold = batch_aggregated_probs[batch_indices, gold_label_tensor]
            ce_loss = -torch.mean(torch.log(probs_at_gold + 1e-10))

        loss = ce_loss

        self.optimizer.zero_grad()
        loss.backward()

        trainable_params = list(self.router_head.parameters())
        if not self.freeze_bert:
            trainable_params.extend(list(self.bert.parameters()))
        torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        batch_aggregated_probs_np = batch_aggregated_probs.detach().cpu().numpy()
        batch_accuracy = float(np.mean(np.argmax(batch_aggregated_probs_np, axis=1) == gold_label))
        avg_prob_correct = float(
            np.mean(batch_aggregated_probs_np[np.arange(batch_size), gold_label])
        )

        return {
            "loss": float(loss.item()),
            "batch_accuracy": batch_accuracy,
            "avg_prob_correct": avg_prob_correct,
            "batch_size": batch_size,
        }

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        params = list(self.router_head.parameters())
        if not self.freeze_bert:
            params.extend(list(self.bert.parameters()))
        return params

    def train_mode(self) -> None:
        self.router_head.train()
        if not self.freeze_bert:
            self.bert.train()
        self.dropout.train()
        self._training = True
        self._cached_wagers = None

    def eval_mode(self) -> None:
        self.router_head.eval()
        self.bert.eval()
        self.dropout.eval()
        self._training = False
        self._cached_wagers = None

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "bert_state_dict": self.bert.state_dict(),
            "router_head_state_dict": self.router_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {
                "bert_model_name": self.bert_model_name,
                "max_seq_length": self.max_seq_length,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "weight_decay": self.weight_decay,
                "freeze_bert": self.freeze_bert,
                "pubmedqa_strip_context": self.pubmedqa_strip_context,
                "router_dropout": self.router_dropout_p,
                "lr_decay_factor": self.lr_decay_factor,
                "lr_decay_steps": self.lr_decay_steps,
                "device": self.device_str,
            },
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        load_module_state(self.bert, state_dict.get("bert_state_dict"))
        load_module_state(self.router_head, state_dict.get("router_head_state_dict"))
        load_optimizer_state(self.optimizer, state_dict.get("optimizer_state_dict"))
        load_scheduler_state(self.scheduler, state_dict.get("scheduler_state_dict"))
