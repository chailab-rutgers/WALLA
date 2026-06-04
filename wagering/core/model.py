import logging
import os
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    LogitsProcessorList,
)

from wagering.core.generation_parameters import (
    GenerationParameters,
    GenerationParametersFactory,
)

log = logging.getLogger("wagering")


class WhiteboxModel:
    """Minimal white-box model wrapper used by wagering pipelines."""

    def __init__(
        self,
        model,
        tokenizer,
        model_path: Optional[str] = None,
        model_type: str = "CausalLM",
        generation_parameters: GenerationParameters = GenerationParameters(),
        instruct: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.model_type = model_type
        self.generation_parameters = generation_parameters
        self.instruct = instruct

    def _validate_args(self, args: Dict) -> Dict:
        args_copy = args.copy()

        if "presence_penalty" in args_copy and args_copy["presence_penalty"] != 0.0:
            log.warning(
                "Skipping unsupported argument presence_penalty=%s",
                args_copy["presence_penalty"],
            )

        for key in ["presence_penalty", "allow_newlines"]:
            args_copy.pop(key, None)

        # Remove max_length if max_new_tokens is set to avoid transformers warning
        if args_copy.get("max_new_tokens") is not None and "max_length" in args_copy:
            args_copy.pop("max_length", None)

        return args_copy

    class _ScoresProcessor:
        """Stores original token log-probabilities during generation."""

        def __init__(self):
            self.scores = []

        def __call__(self, input_ids=None, scores=None):
            self.scores.append(scores.log_softmax(-1))
            return scores

    @staticmethod
    def _generation_arg_names() -> set:
        names = set(GenerationParameters.__dataclass_fields__.keys())
        names.update(
            {
                "max_length",
                "min_length",
                "output_hidden_states",
                "return_dict_in_generate",
            }
        )
        return names

    def generate(self, **args):
        """Generate with stored per-step log-probabilities."""
        default_params = asdict(self.generation_parameters)

        processor = self._ScoresProcessor()
        if "logits_processor" in args:
            logits_processor = LogitsProcessorList([processor, args["logits_processor"]])
        else:
            logits_processor = LogitsProcessorList([processor])
        args["logits_processor"] = logits_processor

        default_params.update(args)
        merged_args = self._validate_args(default_params)

        generation_config = merged_args.get("generation_config")
        if generation_config is not None:
            # Transformers deprecates mixing generation_config with explicit generation
            # kwargs. Consolidate all generation controls into generation_config.
            generation_config = deepcopy(generation_config)
            for key in self._generation_arg_names():
                if key == "generation_config" or key not in merged_args:
                    continue
                value = merged_args.pop(key)
                if value is not None and hasattr(generation_config, key):
                    setattr(generation_config, key, value)

            if getattr(generation_config, "max_new_tokens", None) is not None:
                generation_config.max_length = None

            merged_args["generation_config"] = generation_config
        elif merged_args.get("max_new_tokens") is not None:
            # Keep explicit-kwargs mode consistent and warning-free.
            merged_args["max_length"] = None

        if "stop_strings" in merged_args:
            merged_args["tokenizer"] = self.tokenizer

        generation = self.model.generate(**merged_args)

        generation.generation_scores = getattr(generation, "scores", None)
        generation.scores = processor.scores
        return generation

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """Generate decoded texts for a batch of input prompts."""
        default_params = asdict(self.generation_parameters)
        default_params.update(args)
        merged_args = self._validate_args(default_params)

        merged_args["return_dict_in_generate"] = True
        batch = self.tokenize(input_texts)
        batch = {k: v.to(self.device()) for k, v in batch.items()}
        generation_output = self.generate(**batch, **merged_args)
        sequences = generation_output.sequences.cpu()

        input_len = batch["input_ids"].shape[1]
        decode_args = {}
        if getattr(self.tokenizer, "chat_template", None) is not None:
            decode_args["skip_special_tokens"] = True

        texts = []
        for seq in sequences:
            if self.model_type == "CausalLM":
                texts.append(self.tokenizer.decode(seq[input_len:], **decode_args))
            else:
                texts.append(self.tokenizer.decode(seq[1:], **decode_args))

        return texts

    def __call__(self, **args):
        return self.model(**args)

    def device(self):
        return self.model.device

    @staticmethod
    def _resolve_common_hf_kwargs(kwargs: Dict) -> Dict:
        common = {}
        for key in ["token", "cache_dir", "revision", "local_files_only"]:
            if key in kwargs:
                common[key] = kwargs[key]
        return common

    @staticmethod
    def from_pretrained(
        model_path: str,
        generation_params: Optional[Dict] = None,
        add_bos_token: bool = True,
        instruct: bool = False,
        **kwargs,
    ) -> "WhiteboxModel":
        """Load a HuggingFace model and tokenizer into a WhiteboxModel."""
        generation_params = generation_params or {}

        # Prevent Transformers from spawning background conversion PR threads.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

        common_hf_kwargs = WhiteboxModel._resolve_common_hf_kwargs(kwargs)

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            **common_hf_kwargs,
        )

        model_kwargs = dict(kwargs)
        model_kwargs.setdefault("trust_remote_code", True)
        model_kwargs.pop("use_fast", None)
        model_kwargs.pop("add_bos_token", None)

        architectures = getattr(config, "architectures", None) or []
        model_type = "CausalLM"

        if any("BartModel" in architecture for architecture in architectures):
            model_type = "Seq2SeqLM"
            model = BartForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        elif any(
            ("Seq2SeqLM" in architecture) or ("ConditionalGeneration" in architecture)
            for architecture in architectures
        ):
            model_type = "Seq2SeqLM"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
        elif any(
            ("CausalLM" in architecture) or ("JAISLMHeadModel" in architecture)
            for architecture in architectures
        ):
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            # Fallback order for unknown architecture metadata.
            try:
                model_type = "CausalLM"
                model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            except Exception:
                model_type = "Seq2SeqLM"
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)

        tokenizer_kwargs = dict(common_hf_kwargs)
        tokenizer_kwargs["padding_side"] = "left"
        tokenizer_kwargs["add_bos_token"] = add_bos_token
        if "use_fast" in kwargs:
            tokenizer_kwargs["use_fast"] = kwargs["use_fast"]

        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)

        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.pad_token_id is not None:
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
            if hasattr(model, "config"):
                model.config.pad_token_id = tokenizer.pad_token_id

        if hasattr(model, "generation_config") and model.generation_config is not None:
            native_cfg = model.generation_config.to_dict()
        else:
            native_cfg = dict(getattr(model, "config", {}).__dict__)

        generation_parameters = GenerationParametersFactory.from_params(
            yaml_config=generation_params,
            native_config=native_cfg,
        )

        return WhiteboxModel(
            model=model,
            tokenizer=tokenizer,
            model_path=model_path,
            model_type=model_type,
            generation_parameters=generation_parameters,
            instruct=instruct,
        )

    def tokenize(
        self,
        texts: Union[List[str], List[List[Dict[str, str]]]],
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text prompts for generation."""
        add_start_symbol = True

        if self.instruct:
            chat_template = getattr(self.tokenizer, "chat_template", None)
            if chat_template is not None:
                formatted_texts: List[str] = []
                for chat in texts:
                    if isinstance(chat, str):
                        chat = [{"role": "user", "content": chat}]
                    try:
                        formatted = self.tokenizer.apply_chat_template(
                            chat,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    except (ValueError, TypeError):
                        if isinstance(chat, list) and chat and isinstance(chat[0], dict):
                            formatted = str(chat[0].get("content", ""))
                        else:
                            formatted = str(chat)
                    formatted_texts.append(formatted)

                texts = formatted_texts
                add_start_symbol = False

        return self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=add_start_symbol,
        )
