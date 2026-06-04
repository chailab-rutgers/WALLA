"""
RouterDC-style query router for multi-LLM wagering (Chen et al., NeurIPS 2024; arXiv:2409.19886).

Query encoder + trainable expert embeddings; routing scores are similarity (cosine or dot)
between the query embedding and each expert embedding, then softmax with temperature.

Training uses sample–LLM contrastive loss: positives/negatives are derived from each
expert's probability on the gold label (from cached `model_logits`), matching the
reference implementation's use of per-expert scores without requiring task/cluster IDs.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from transformers import AutoModel, AutoTokenizer

from .base import WageringMethod
from .utils import preprocess_pubmedqa_prompts_for_embedding
from wagering.aggregation.linear_pooling import LinearPooling

logger = logging.getLogger("wagering")


class RouterDCWagers(WageringMethod):
    """
    Encoder + expert embeddings + similarity routing (RouterDC-style).

    Unlike `CentralizedWagers`, routing uses only the task prompt text, not LLM hidden states.
    Unlike `RouteLLMBertWagers`, experts are represented by trainable embedding vectors and
    the training objective is sample–LLM contrastive (multi-positive) rather than a linear
    head + mixture cross-entropy (optional CE can be added later).
    """

    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(num_models, config or {})
        cfg = self.config

        # RouterDC routes using the question text via its own encoder and does not
        # require LLM hidden states. Opt out so the trainer does not load/cache
        # hidden states for every ensemble model.
        self.requires_hidden_states = False

        self.encoder_model_name = str(
            cfg.get("encoder_model_name", cfg.get("bert_model_name", "microsoft/mdeberta-v3-base"))
        )
        self.max_seq_length = int(cfg.get("max_seq_length", 512))
        self.learning_rate = float(cfg.get("learning_rate", 5e-5))
        self.temperature = float(cfg.get("temperature", 1.0))
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and > 0 for router_dc")
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
        self.weight_decay = float(cfg.get("weight_decay", 0.01))
        self.freeze_encoder = bool(cfg.get("freeze_encoder", False))
        # Default to keeping context (caller can disable it explicitly).
        self.pubmedqa_strip_context = bool(cfg.get("pubmedqa_strip_context", False))
        self.similarity_function = str(cfg.get("similarity_function", "cos")).lower()
        if self.similarity_function not in ("cos", "dot"):
            raise ValueError("similarity_function must be 'cos' or 'dot'")

        self.top_k = int(cfg.get("top_k", 3))
        self.last_k = int(cfg.get("last_k", 3))
        self.min_pos_p = float(cfg.get("min_pos_p", 0.01))
        self.neg_mask_threshold = float(cfg.get("neg_mask_threshold", 0.5))
        self.inactive_model_indices = {int(i) for i in cfg.get("inactive_model_indices", [])}
        if any(i < 0 or i >= num_models for i in self.inactive_model_indices):
            raise ValueError(
                f"inactive_model_indices must be within [0, {num_models - 1}], got {sorted(self.inactive_model_indices)}"
            )
        if len(self.inactive_model_indices) >= num_models:
            raise ValueError("router_dc requires at least one active model")

        self.lr_decay_factor = float(cfg.get("lr_decay_factor", 1.0))
        self.lr_decay_steps = int(cfg.get("lr_decay_steps", 100))
        # Optional stability knobs.
        self.encoder_lr_mult = float(cfg.get("encoder_lr_mult", 1.0))
        if not np.isfinite(self.encoder_lr_mult) or self.encoder_lr_mult <= 0.0:
            raise ValueError("encoder_lr_mult must be finite and > 0 for router_dc")
        self.expert_weight_decay = float(cfg.get("expert_weight_decay", self.weight_decay))
        self.max_grad_norm_before_skip = float(cfg.get("max_grad_norm_before_skip", 1.0e6))
        if not np.isfinite(self.max_grad_norm_before_skip) or self.max_grad_norm_before_skip <= 0.0:
            raise ValueError("max_grad_norm_before_skip must be finite and > 0 for router_dc")

        self.device_str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)

        self.hidden_state_layers = cfg.get("hidden_state_layers", [-1])
        # When pubmedqa_strip_context is False, we build one encoder embedding per model prompt
        # (context/no-context variants) and concatenate them in model order.
        self.concat_prompt_embeddings = not self.pubmedqa_strip_context
        # Used by trainer/evaluator to decide whether to pass per-model prompt variants.
        self.expects_per_model_router_prompts = True
        # Micro-batch size for DeBERTa encoding (caps peak VRAM usage).
        self.micro_batch_size = int(cfg.get("micro_batch_size", 0) or 0)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.encoder_model_name, truncation_side="left", padding=True
        )
        # Root-cause fix: RouterDC fine-tunes the encoder with plain AdamW (no AMP GradScaler /
        # fp32 master weights). Therefore, we must keep router encoder parameters in fp32.
        #
        # If some upstream code or environment forces fp16/bf16, `from_pretrained()` may produce
        # half-precision weights; stepping those with AdamW can yield NaNs immediately.
        self.encoder = AutoModel.from_pretrained(
            self.encoder_model_name,
            torch_dtype=torch.float32,
        ).to(device=self.device)
        hidden_size = int(self.encoder.config.hidden_size)

        expert_dim = hidden_size * num_models if self.concat_prompt_embeddings else hidden_size
        std_dev = float(cfg.get("expert_embedding_std", 0.78))
        self.expert_embeddings = torch.nn.Embedding(num_models, expert_dim).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            torch.nn.init.normal_(self.expert_embeddings.weight, mean=0.0, std=std_dev)

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # Use param groups so the encoder LR can be tuned separately.
        optimizer_param_groups: List[Dict[str, Any]] = [
            {
                "params": list(self.expert_embeddings.parameters()),
                "lr": self.learning_rate,
                "weight_decay": self.expert_weight_decay,
            }
        ]
        if not self.freeze_encoder:
            optimizer_param_groups.append(
                {
                    "params": list(self.encoder.parameters()),
                    "lr": self.learning_rate * self.encoder_lr_mult,
                    "weight_decay": self.weight_decay,
                }
            )

        self.optimizer = torch.optim.AdamW(optimizer_param_groups)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=max(1, self.lr_decay_steps),
            gamma=self.lr_decay_factor,
        )

        self._training = True
        self._cached_wagers: Optional[torch.Tensor] = None
        self._cached_router_logits: Optional[torch.Tensor] = None

    def _active_model_mask(self) -> Optional[torch.Tensor]:
        if not self.inactive_model_indices:
            return None
        mask = torch.ones((1, self.num_models), dtype=torch.float32, device=self.device)
        inactive = sorted(self.inactive_model_indices)
        mask[:, inactive] = 0.0
        return mask

    def _normalize_wagers_safe(self, wagers: torch.Tensor) -> torch.Tensor:
        """Ensure finite, non-negative, row-normalized wagers (no fallback)."""
        active_mask = self._active_model_mask()
        if active_mask is not None:
            wagers = wagers * active_mask.to(dtype=wagers.dtype)

        if torch.any(~torch.isfinite(wagers)):
            raise ValueError("router_dc produced non-finite wagers before normalization")
        if torch.any(wagers < 0):
            raise ValueError("router_dc produced negative wagers before normalization")

        row_sums = wagers.sum(dim=1, keepdim=True)
        if torch.any(~torch.isfinite(row_sums)) or torch.any(row_sums <= 1e-12):
            raise ValueError("router_dc produced invalid wager row sums during normalization")
        normalized = wagers / row_sums
        return normalized

    def _encode_questions_batch(self, questions: List[str]) -> torch.Tensor:
        if not hasattr(self, "_logged_first_encode_batch"):
            setattr(self, "_logged_first_encode_batch", False)
        processed = preprocess_pubmedqa_prompts_for_embedding(
            questions,
            # Do not regex-strip content here. When pubmedqa_strip_context is enabled for this method,
            # the trainer/evaluator is responsible for passing the dataset's prompt_without_context
            # variant verbatim (even if it contains "Context:" text).
            strip_context=False,
        )
        if not getattr(self, "_logged_first_encode_batch", False):
            try:
                if torch.cuda.is_available() and self.device.type == "cuda":
                    dev = torch.cuda.current_device()
                    alloc = float(torch.cuda.memory_allocated(dev)) / (1024**3)
                    resv = float(torch.cuda.memory_reserved(dev)) / (1024**3)
                    logger.info(
                        "[router_dc] encode_batch: n_prompts=%d max_seq_length=%d device=%s cuda_dev=%s alloc=%.2fGiB reserved=%.2fGiB",
                        len(processed),
                        int(self.max_seq_length),
                        str(self.device),
                        str(dev),
                        alloc,
                        resv,
                    )
                else:
                    logger.info(
                        "[router_dc] encode_batch: n_prompts=%d max_seq_length=%d device=%s",
                        len(processed),
                        int(self.max_seq_length),
                        str(self.device),
                    )
            except Exception:
                pass
        mbs = int(self.micro_batch_size) if int(self.micro_batch_size) > 0 else len(processed)
        chunks: List[torch.Tensor] = []
        for start in range(0, len(processed), mbs):
            end = min(start + mbs, len(processed))
            inputs = self.tokenizer(
                processed[start:end],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
                padding=True,
            ).to(self.device)
            if not getattr(self, "_logged_first_encode_batch", False):
                try:
                    if torch.cuda.is_available() and self.device.type == "cuda":
                        dev = torch.cuda.current_device()
                        alloc = float(torch.cuda.memory_allocated(dev)) / (1024**3)
                        resv = float(torch.cuda.memory_reserved(dev)) / (1024**3)
                        input_shape = tuple(getattr(inputs, "input_ids", torch.empty(0)).shape)
                        logger.info(
                            "[router_dc] after_tokenize_to_device: input_ids_shape=%s alloc=%.2fGiB reserved=%.2fGiB micro_batch_size=%d",
                            str(input_shape),
                            alloc,
                            resv,
                            int(mbs),
                        )
                except Exception:
                    pass
                setattr(self, "_logged_first_encode_batch", True)

            grad_enc = self._training and not self.freeze_encoder
            with torch.set_grad_enabled(grad_enc):
                outputs = self.encoder(**inputs)
            chunks.append(outputs.last_hidden_state[:, 0, :])

        if not chunks:
            return torch.empty((0, int(self.encoder.config.hidden_size)), device=self.device, dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def _encode_questions_per_model_batch(self, questions_per_model: List[List[str]]) -> torch.Tensor:
        """
        Encode per-model prompt variants, then concatenate embeddings in model order.

        questions_per_model: list length M, each element is a list of B strings.
        returns: [B, M*H] (or [B, H] if concat_prompt_embeddings is False).
        """
        if not isinstance(questions_per_model, list) or len(questions_per_model) != self.num_models:
            raise ValueError(
                "router_dc expects questions_per_model as a list of length num_models "
                f"(expected {self.num_models})."
            )
        batch_size = len(questions_per_model[0]) if self.num_models > 0 else 0
        for mi, prompts in enumerate(questions_per_model):
            if len(prompts) != batch_size:
                raise ValueError(
                    "router_dc questions_per_model batch mismatch: "
                    f"model_index={mi}, len={len(prompts)}, expected={batch_size}"
                )

        if not self.concat_prompt_embeddings:
            # Use model 0 prompts as the single router prompt stream.
            return self._encode_questions_batch(list(questions_per_model[0]))

        # Deduplicate by prompt text to save encoder calls.
        flat_prompts: List[str] = []
        for mi in range(self.num_models):
            flat_prompts.extend([str(p) for p in questions_per_model[mi]])
        unique_prompts: List[str] = []
        index_by_prompt: Dict[str, int] = {}
        for p in flat_prompts:
            if p in index_by_prompt:
                continue
            index_by_prompt[p] = len(unique_prompts)
            unique_prompts.append(p)

        unique_emb = self._encode_questions_batch(unique_prompts)  # [U, H]
        # Reconstruct per-model embeddings and concatenate in model order.
        per_model_emb: List[torch.Tensor] = []
        for mi in range(self.num_models):
            idx = torch.as_tensor(
                [index_by_prompt[str(p)] for p in questions_per_model[mi]],
                device=self.device,
                dtype=torch.long,
            )
            per_model_emb.append(unique_emb.index_select(0, idx))  # [B, H]
        return torch.cat(per_model_emb, dim=1)  # [B, M*H]

    def _compute_similarity(self, query_emb: torch.Tensor) -> torch.Tensor:
        """query_emb: [B, H], returns logits [B, M] before temperature scaling."""
        expert_w = self.expert_embeddings.weight  # [M, H]
        if query_emb.dtype != expert_w.dtype:
            query_emb = query_emb.to(dtype=expert_w.dtype)
        if self.similarity_function == "cos":
            q = F.normalize(query_emb, dim=-1)
            e = F.normalize(expert_w, dim=-1)
            return q @ e.T
        return query_emb @ expert_w.T

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
                "RouterDCWagers.compute_wagers() requires `questions` (batch of prompt strings)."
            )

        questions_per_model = kwargs.get("questions_per_model", None)
        if questions_per_model is not None:
            query_emb = self._encode_questions_per_model_batch(questions_per_model)
        elif self.concat_prompt_embeddings:
            # Fallback: replicate the same prompt stream across models.
            replicated = [list(questions) for _ in range(self.num_models)]
            query_emb = self._encode_questions_per_model_batch(replicated)
        else:
            query_emb = self._encode_questions_batch(questions)
        self.expert_embeddings.train() if self._training else self.expert_embeddings.eval()

        with torch.set_grad_enabled(self._training):
            logits = self._compute_similarity(query_emb)
            logits = logits / self.temperature
            wagers = torch.softmax(logits, dim=1)
            wagers = self._normalize_wagers_safe(wagers)

        if self._training:
            self._cached_wagers = wagers
            self._cached_router_logits = logits

        return {"wagers": wagers.detach().cpu().numpy()}

    def _sample_llm_contrastive_loss(
        self,
        router_logits: torch.Tensor,
        p_gold: torch.Tensor,
    ) -> torch.Tensor:
        """
        router_logits: [B, M] (already scaled by temperature)
        p_gold: [B, M] probability each expert assigns to gold label
        """
        B, M = router_logits.shape
        device = router_logits.device
        router_logits = torch.nan_to_num(router_logits, nan=0.0, posinf=50.0, neginf=-50.0)
        p_gold = torch.nan_to_num(p_gold, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)

        if self.inactive_model_indices:
            inactive = sorted(self.inactive_model_indices)
            p_gold[:, inactive] = -1.0
            router_logits[:, inactive] = float("-inf")

        k_pos = min(self.top_k, M)
        k_neg = min(self.last_k, M)

        _, top_idx = torch.topk(p_gold, k=k_pos, dim=1)
        _, bot_idx = torch.topk(p_gold, k=k_neg, dim=1, largest=False)

        total = torch.zeros((), device=device)
        n_terms = 0

        for i in range(k_pos):
            pos_idx = top_idx[:, i]
            pos_logit = torch.gather(router_logits, 1, pos_idx.unsqueeze(1)).squeeze(1)
            pos_p = torch.gather(p_gold, 1, pos_idx.unsqueeze(1)).squeeze(1)
            mask = pos_p > self.min_pos_p

            neg_logits = torch.gather(router_logits, 1, bot_idx)
            neg_p = torch.gather(p_gold, 1, bot_idx)
            neg_logits = torch.where(
                neg_p > self.neg_mask_threshold,
                torch.full_like(neg_logits, float("-inf")),
                neg_logits,
            )
            neg_logits = torch.where(
                bot_idx == pos_idx.unsqueeze(1),
                torch.full_like(neg_logits, float("-inf")),
                neg_logits,
            )

            stacked = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
            log_probs = F.log_softmax(stacked, dim=1)
            term = -log_probs[:, 0]
            # Match the reference RouterDC implementation: average over the full batch,
            # so rare masked positives do not get upweighted (which can cause unstable steps).
            total = total + (term * mask.float()).mean()
            n_terms += 1

        return total / max(n_terms, 1)

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
        if self._cached_wagers is None or self._cached_router_logits is None:
            raise ValueError(
                "RouterDCWagers.update() requires cached wagers from compute_wagers() "
                "called in training mode beforehand."
            )

        wagers = self._cached_wagers
        router_logits = self._cached_router_logits
        self._cached_wagers = None
        self._cached_router_logits = None

        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32, device=self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long, device=self.device)

        probs = F.softmax(model_logits_tensor, dim=-1)
        gold_label_distribution = kwargs.get("gold_label_distribution", None)
        if gold_label_distribution is not None:
            gold_label_distribution_tensor = torch.as_tensor(
                gold_label_distribution, dtype=torch.float32, device=self.device
            )
            if (
                gold_label_distribution_tensor.ndim != 2
                or gold_label_distribution_tensor.shape[0] != batch_size
                or gold_label_distribution_tensor.shape[1] != probs.shape[2]
            ):
                raise ValueError(
                    "gold_label_distribution must be shape [batch_size, num_options], "
                    f"got {tuple(gold_label_distribution_tensor.shape)}"
                )
            # Expected probability mass each expert assigns to the (soft) label distribution:
            #   p_gold[b,m] = sum_k q[b,k] * p[b,m,k]
            q_expanded = gold_label_distribution_tensor.to(dtype=probs.dtype).unsqueeze(1).expand(
                -1, self.num_models, -1
            )
            p_gold = torch.sum(probs * q_expanded, dim=-1)
        else:
            idx = gold_label_tensor.view(-1, 1, 1).expand(-1, self.num_models, 1)
            p_gold = torch.gather(probs, dim=2, index=idx).squeeze(2)

        loss = self._sample_llm_contrastive_loss(router_logits, p_gold)
        if not torch.isfinite(loss):
            logger.warning("[router_dc] Non-finite loss detected; skipping optimizer step")
            batch_aggregated_probs = LinearPooling.aggregate_torch(model_logits_tensor, wagers)
            batch_aggregated_probs_np = batch_aggregated_probs.detach().cpu().numpy()
            batch_accuracy = float(np.mean(np.argmax(batch_aggregated_probs_np, axis=1) == gold_label))
            avg_prob_correct = float(
                np.mean(batch_aggregated_probs_np[np.arange(batch_size), gold_label])
            )
            return {
                "loss": float("nan"),
                "batch_accuracy": batch_accuracy,
                "avg_prob_correct": avg_prob_correct,
                "batch_size": batch_size,
                "skipped_update_nonfinite_loss": True,
            }

        self.optimizer.zero_grad()
        loss.backward()

        # Keep named parameters for better debugging if an optimizer step overflows.
        named_trainable_params: List[tuple[str, torch.nn.Parameter]] = [
            (f"expert_embeddings.{n}", p) for n, p in self.expert_embeddings.named_parameters()
        ]
        if not self.freeze_encoder:
            named_trainable_params.extend(
                [(f"encoder.{n}", p) for n, p in self.encoder.named_parameters()]
            )
        trainable_params = [p for _, p in named_trainable_params]

        grads_finite = True
        for _, p in named_trainable_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grads_finite = False
                break
        if not grads_finite:
            logger.warning("[router_dc] Non-finite gradients detected; skipping optimizer step")
            self.optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip_norm)
            if not torch.isfinite(grad_norm) or float(grad_norm) > self.max_grad_norm_before_skip:
                logger.warning(
                    "[router_dc] Abnormal grad norm (%.4e) detected; skipping optimizer step",
                    float(grad_norm) if torch.isfinite(grad_norm) else float("nan"),
                )
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
                self.scheduler.step()

                # Fail fast if optimizer produced invalid parameters.
                for name, p in named_trainable_params:
                    if not torch.isfinite(p).all():
                        raise RuntimeError(
                            "router_dc parameters became non-finite after optimizer step "
                            f"(first bad tensor: {name}, dtype={p.dtype}). "
                            "Root cause is usually fp16/bf16 weights being stepped without AMP. "
                            "Ensure router encoder params are fp32 or use proper AMP training."
                        )

        batch_aggregated_probs = LinearPooling.aggregate_torch(model_logits_tensor, wagers)
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
        params = list(self.expert_embeddings.parameters())
        if not self.freeze_encoder:
            params.extend(list(self.encoder.parameters()))
        return params

    def train_mode(self) -> None:
        self.expert_embeddings.train()
        if not self.freeze_encoder:
            self.encoder.train()
        self._training = True
        self._cached_wagers = None
        self._cached_router_logits = None

    def eval_mode(self) -> None:
        self.expert_embeddings.eval()
        self.encoder.eval()
        self._training = False
        self._cached_wagers = None
        self._cached_router_logits = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder_state_dict": self.encoder.state_dict(),
            "expert_embeddings_state_dict": self.expert_embeddings.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {
                "encoder_model_name": self.encoder_model_name,
                "max_seq_length": self.max_seq_length,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "weight_decay": self.weight_decay,
                "freeze_encoder": self.freeze_encoder,
                "pubmedqa_strip_context": self.pubmedqa_strip_context,
                "similarity_function": self.similarity_function,
                "top_k": self.top_k,
                "last_k": self.last_k,
                "min_pos_p": self.min_pos_p,
                "neg_mask_threshold": self.neg_mask_threshold,
                "inactive_model_indices": sorted(self.inactive_model_indices),
                "lr_decay_factor": self.lr_decay_factor,
                "lr_decay_steps": self.lr_decay_steps,
                "hidden_state_layers": self.hidden_state_layers,
                "device": self.device_str,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "encoder_state_dict" in state_dict:
            self.encoder.load_state_dict(state_dict["encoder_state_dict"])
        if "expert_embeddings_state_dict" in state_dict:
            self.expert_embeddings.load_state_dict(state_dict["expert_embeddings_state_dict"])
        if "optimizer_state_dict" in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                import logging

                logging.getLogger("wagering").warning(
                    "Could not load optimizer state dict: %s. Using fresh optimizer.", e
                )
        if "scheduler_state_dict" in state_dict:
            try:
                self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            except (ValueError, KeyError) as e:
                import logging

                logging.getLogger("wagering").warning(
                    "Could not load scheduler state dict: %s. Using fresh scheduler.", e
                )
