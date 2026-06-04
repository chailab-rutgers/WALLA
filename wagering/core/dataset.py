import os
import ast
import json
import inspect
import importlib
import pandas as pd
import numpy as np
import logging
import requests
import io

from sklearn.model_selection import train_test_split


def _configure_default_hf_cache_env_early() -> None:
    """Configure default HF cache env vars before importing datasets/huggingface_hub."""
    if (
        os.environ.get("HF_HOME")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_DATASETS_CACHE")
    ):
        return

    user = os.environ.get("USER", "").strip()
    if not user:
        return

    shared_cache_root = f"/common/users/{user}/.cache"
    if not os.path.isdir(shared_cache_root):
        return

    shared_hf_home = os.path.join(shared_cache_root, "huggingface")
    if not os.path.isdir(shared_hf_home):
        return

    os.environ["HF_HOME"] = shared_hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(shared_hf_home, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(shared_hf_home, "datasets")


_configure_default_hf_cache_env_early()

from datasets import DownloadConfig, load_dataset, Dataset as hf_dataset

from typing import Iterable, Tuple, List, Union, Optional
from PIL import Image

log = logging.getLogger("wagering")


def _configure_hf_cache_home_if_needed() -> None:
    """Point HF cache env vars to a shared cache when the default home cache is missing."""
    if (
        os.environ.get("HF_HOME")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_DATASETS_CACHE")
    ):
        return

    default_hf_home = os.path.expanduser("~/.cache/huggingface")
    if os.path.isdir(default_hf_home):
        return

    user = os.environ.get("USER", "").strip()
    if not user:
        return

    shared_hf_home = f"/common/users/{user}/.cache/huggingface"
    if not os.path.isdir(shared_hf_home):
        return

    os.environ["HF_HOME"] = shared_hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(shared_hf_home, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(shared_hf_home, "datasets")
    log.info("Using shared Hugging Face cache directory: %s", shared_hf_home)


def _normalize_pubmedqa_label(label: object) -> Optional[str]:
    """Normalize PubMedQA labels to YES/NO and drop unsupported labels like maybe."""
    if label is None:
        return None

    normalized = str(label).strip().lower()
    if normalized == "yes":
        return "YES"
    if normalized == "no":
        return "NO"
    return None


def _stringify_pubmedqa_field(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_pubmedqa_context(context_field: object) -> str:
    """Format PubMedQA context payload into a readable block."""
    if isinstance(context_field, dict):
        raw_contexts = context_field.get("contexts", [])
        raw_labels = context_field.get("labels", [])
        contexts = raw_contexts if isinstance(raw_contexts, list) else [raw_contexts]
        labels = raw_labels if isinstance(raw_labels, list) else [raw_labels]

        formatted_sections = []
        for idx, ctx in enumerate(contexts):
            ctx_text = _stringify_pubmedqa_field(ctx)
            if not ctx_text:
                continue

            label_text = ""
            if idx < len(labels):
                label_text = _stringify_pubmedqa_field(labels[idx])
            if label_text:
                pretty_label = label_text.replace("_", " ").title()
                formatted_sections.append(f"({pretty_label}) {ctx_text}")
            else:
                formatted_sections.append(ctx_text)
        return "\n".join(formatted_sections)

    if isinstance(context_field, list):
        cleaned = [_stringify_pubmedqa_field(item) for item in context_field]
        return "\n".join([item for item in cleaned if item])

    return _stringify_pubmedqa_field(context_field)


def _build_pubmedqa_prompt(
    question: str,
    long_answer: str,
    context_text: str,
    include_context: bool,
) -> str:
    if include_context:
        return (
            f"Question:\n{question}\n"
            f"Context:\n{context_text}\n"
            f"Long Answer:\n{long_answer}\n"
            "Is the long answer provided correct or incorrect? "
            "Answer with YES or NO. Answer:"
        )

    return (
        f"Question:\n{question}\n"
        f"Long Answer:\n{long_answer}\n"
        "Is the long answer provided correct or incorrect? "
        "Answer with YES or NO. Answer:"
    )


def _build_pubmedqa_context_only_prompt(
    question: str,
    context_text: str,
) -> str:
    return (
        f"Question:\n{question}\n"
        f"Context:\n{context_text}\n"
        "Based only on the context, is the answer to the question YES or NO? "
        "Answer with YES or NO. Answer:"
    )


def _normalize_multiple_choice_answer(label: object) -> Optional[str]:
    """Map mixed answer representations to A/B/C/D labels."""
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    if label is None:
        return None

    if isinstance(label, int):
        return option_map.get(int(label))

    normalized = str(label).strip().upper()
    if normalized in {"A", "B", "C", "D"}:
        return normalized

    if normalized in {"0", "1", "2", "3"}:
        return option_map.get(int(normalized))

    return None


def _format_choices_block(option_a: str, option_b: str, option_c: str, option_d: str) -> str:
    """Format non-empty multiple-choice options in A/B/C/D order."""
    choices_list = []
    for label, text in [("A", option_a), ("B", option_b), ("C", option_c), ("D", option_d)]:
        cleaned = str(text).strip() if text is not None else ""
        if cleaned:
            choices_list.append(f"{label}) {cleaned}")
    return "\n".join(choices_list)


class Dataset:
    """
    Seq2seq or vision-language dataset for calculating quality of uncertainty estimation method.
    """

    def __init__(
        self, x: List[str], y: List[str], batch_size: int, images: Optional[str] = None
    ):
        """
        Parameters:
            x (List[str]): a list of input texts.
            y (List[str]): a list of output (target) texts. Must have the same length as `x`.
            batch_size (int): the size of the texts batch.
        """
        self.x = x
        self.y = y
        self.images = images
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[List[str], List[str], Optional[List]]]:
        """
        Returns:
            Iterable[Tuple[List[str], List[str]]]: iterates over batches in dataset,
                returns list of input texts and list of corresponding output texts.
        """
        for i in range(0, len(self.x), self.batch_size):
            batch_x = self.x[i : i + self.batch_size]
            batch_y = self.y[i : i + self.batch_size]
            batch_images = (
                self.images[i : i + self.batch_size]
                if self.images is not None
                else None
            )
            yield (batch_x, batch_y, batch_images)

    def __len__(self) -> int:
        """
        Returns:
            int: number of batches in the dataset.
        """
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def select(self, indices: List[int]):
        """
        Shrinks the dataset down to only texts with the specified index.

        Parameters:
            indices (List[int]): indices to left in the dataset.Must have the same length as input texts.
        """
        self.x = [self.x[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        if self.images is not None:
            self.images = [self.images[i] for i in indices]
        if hasattr(self, "probabilistic_labels") and isinstance(self.probabilistic_labels, list):
            self.probabilistic_labels = [self.probabilistic_labels[i] for i in indices]
        if hasattr(self, "probability_labels") and isinstance(self.probability_labels, list):
            self.probability_labels = [self.probability_labels[i] for i in indices]
        return self

    def train_test_split(self, test_size: int, seed: int, split: str = "train"):
        """
        Samples dataset into train and test parts.

        Parameters:
            test_size (int): size of test dataset,
            seed (int): seed to perform random splitting with,
            split (str): either 'train' or 'test'. If 'train', lefts only train data in the current dataset object.
                If 'test', left only test data. Default: 'train'.

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: train input and target texts list,
                test input and target texts list.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(self.x),
            np.array(self.y),
            test_size=test_size,
            random_state=seed,
        )
        if self.images is not None:
            images_train, images_test = train_test_split(
                np.array(self.images), test_size=test_size, random_state=seed
            )
        else:
            images_train = images_test = None

        if split == "train":
            self.x, self.y, self.images = (
                X_train.tolist(),
                y_train.tolist(),
                images_train.tolist() if images_train is not None else None,
            )
        else:
            self.x, self.y, self.images = (
                X_test.tolist(),
                y_test.tolist(),
                images_test.tolist() if images_test is not None else None,
            )

        return (
            X_train.tolist(),
            X_test.tolist(),
            y_train.tolist(),
            y_test.tolist(),
        )

    def subsample(self, size: int, seed: int):
        """
        Subsamples the dataset to the provided size.

        Parameters:
            size (int): size of the resulting dataset,
            seed (int): seed to perform random subsampling with.
        """
        np.random.seed(seed)
        if len(self.x) < size:
            indices = list(range(len(self.x)))
        else:
            if size < 1:
                size = int(size * len(self.x))
            indices = np.random.choice(len(self.x), size, replace=False)
        self.select(indices)

    @staticmethod
    def from_csv(
        csv_path: str,
        x_column: str,
        y_column: str,
        batch_size: int,
        prompt: str = "",
        **kwargs,
    ):
        """
        Creates the dataset from .CSV table.

        Parameters:
            csv_path (str): path to .csv table,
            x_column (str): name of column to take input texts from,
            y_column (str): name of column to take target texts from,
            batch_size (int): the size of the texts batch.
        """
        csv = pd.read_csv(csv_path)

        if x_column in csv.columns:
            raw_x = csv[x_column].tolist()
        elif len(prompt):
            # Prompt templates can be rendered from arbitrary CSV columns,
            # so the text column can be omitted for compact variable-only datasets.
            raw_x = [""] * len(csv)
        else:
            raise ValueError(
                f"x_column='{x_column}' not found in CSV columns and no prompt template was provided"
            )

        x = raw_x
        y = csv[y_column].tolist()
        records = csv.to_dict(orient="records")

        prompt_without_context = str(kwargs.get("prompt_without_context", "") or "")
        mixed_context_routing = str(kwargs.get("mixed_context_routing", "") or "").strip().lower()
        prompt_helper_spec = str(kwargs.get("prompt_helper", "") or "").strip()
        prompt_without_context_helper_spec = str(
            kwargs.get("prompt_without_context_helper", "") or ""
        ).strip()

        with_context_prompts = None
        without_context_prompts = None

        def _resolve_helper(helper_spec: str, key_name: str):
            if not helper_spec:
                return None

            module_path, sep, function_name = helper_spec.partition(":")
            if not sep or not module_path or not function_name:
                raise ValueError(
                    f"{key_name} must use 'module.path:function_name' format, got: {helper_spec}"
                )

            try:
                module = importlib.import_module(module_path)
            except Exception as exc:
                raise ValueError(
                    f"Failed to import module '{module_path}' for {key_name}: {exc}"
                ) from exc

            helper = getattr(module, function_name, None)
            if helper is None or not callable(helper):
                raise ValueError(
                    f"{key_name} references missing or non-callable '{function_name}' in module '{module_path}'"
                )
            return helper

        def _render_with_helper(helper, row: dict, key_name: str) -> str:
            try:
                signature = inspect.signature(helper)
            except (TypeError, ValueError):
                signature = None

            try:
                if signature is None:
                    return str(helper(**row))

                call_kwargs = {}
                has_var_keyword = False
                for param in signature.parameters.values():
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        has_var_keyword = True
                        break
                    if param.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.VAR_POSITIONAL,
                    ):
                        raise ValueError(
                            f"{key_name} helper '{helper.__name__}' must use keyword-compatible params"
                        )
                    if param.name in row:
                        call_kwargs[param.name] = row[param.name]
                    elif param.default is inspect.Parameter.empty:
                        raise ValueError(
                            f"{key_name} helper '{helper.__name__}' requires CSV column '{param.name}'"
                        )

                if has_var_keyword:
                    call_kwargs = dict(row)

                return str(helper(**call_kwargs))
            except Exception as exc:
                raise ValueError(
                    f"Failed to render {key_name} using helper '{helper.__module__}:{helper.__name__}' "
                    f"for CSV row with id={row.get('id', '<missing>')}: {exc}"
                ) from exc

        prompt_helper = _resolve_helper(prompt_helper_spec, "prompt_helper")
        prompt_without_context_helper = _resolve_helper(
            prompt_without_context_helper_spec,
            "prompt_without_context_helper",
        )

        if prompt_helper is not None:
            with_context_prompts = [
                _render_with_helper(prompt_helper, row, "prompt_helper") for row in records
            ]
            x = with_context_prompts
        elif len(prompt):
            formatted_x = []
            for text_value, row in zip(raw_x, records):
                format_kwargs = dict(row)
                format_kwargs.setdefault("text", text_value)
                try:
                    formatted_x.append(prompt.format(**format_kwargs))
                except KeyError as exc:
                    missing_key = str(exc).strip("'")
                    raise ValueError(
                        f"Prompt references missing CSV column '{{{missing_key}}}' in {csv_path}"
                    ) from exc
            with_context_prompts = formatted_x
            x = formatted_x

        if prompt_without_context_helper is not None:
            without_context_prompts = [
                _render_with_helper(
                    prompt_without_context_helper,
                    row,
                    "prompt_without_context_helper",
                )
                for row in records
            ]
        elif len(prompt_without_context):
            formatted_without_context = []
            for text_value, row in zip(raw_x, records):
                format_kwargs = dict(row)
                format_kwargs.setdefault("text", text_value)
                try:
                    formatted_without_context.append(prompt_without_context.format(**format_kwargs))
                except KeyError as exc:
                    missing_key = str(exc).strip("'")
                    raise ValueError(
                        f"prompt_without_context references missing CSV column '{{{missing_key}}}' in {csv_path}"
                    ) from exc
            without_context_prompts = formatted_without_context

        if mixed_context_routing == "pubmedqa":
            if with_context_prompts is None:
                raise ValueError(
                    "mixed_context_routing='pubmedqa' requires either 'prompt' or 'prompt_helper'"
                )
            if without_context_prompts is None:
                raise ValueError(
                    "mixed_context_routing='pubmedqa' requires either 'prompt_without_context' or 'prompt_without_context_helper'"
                )
            # PubMedQA-style mixed-context routing uses reduced prompts by default and
            # swaps in evidence-bearing prompts only for the assigned model per example.
            x = without_context_prompts

        formatted_dataset = Dataset(x, y, batch_size)

        if mixed_context_routing == "pubmedqa":
            formatted_dataset.pubmedqa_with_context_x = list(with_context_prompts)
            formatted_dataset.pubmedqa_without_context_x = list(without_context_prompts)
            formatted_dataset.pubmedqa_prompt_strategy = "mixed_context"

        probability_label_column = kwargs.get("probability_label_column", None)
        if probability_label_column is not None:
            if probability_label_column not in csv.columns:
                raise ValueError(
                    f"probability_label_column='{probability_label_column}' not found in CSV columns"
                )
            prob_values = pd.to_numeric(csv[probability_label_column], errors="coerce")
            if prob_values.isna().any():
                raise ValueError(
                    f"Column '{probability_label_column}' contains non-numeric probabilistic labels"
                )
            prob_array = prob_values.to_numpy(dtype=np.float64)
            if np.any(prob_array < 0.0) or np.any(prob_array > 1.0):
                raise ValueError(
                    f"Column '{probability_label_column}' must contain values in [0, 1]"
                )
            prob_list = prob_array.astype(np.float32).tolist()
            # Backwards-compatible name + clearer alias aligned with config key
            # (`probability_label_column`).
            formatted_dataset.probabilistic_labels = prob_list
            formatted_dataset.probability_labels = prob_list

        positive_label = kwargs.get("positive_label", None)
        if positive_label is not None:
            formatted_dataset.positive_label = str(positive_label)

        return formatted_dataset

    @staticmethod
    def load_hf_dataset(
        path: Union[str, List[str]],
        split: str,
        **kwargs,
    ):
        _configure_hf_cache_home_if_needed()

        log.info(
            "HF dataset loader env: HF_HOME=%s HF_HUB_CACHE=%s HF_DATASETS_CACHE=%s",
            os.environ.get("HF_HOME", "<unset>"),
            os.environ.get("HF_HUB_CACHE", "<unset>"),
            os.environ.get("HF_DATASETS_CACHE", "<unset>"),
        )

        # Always try local cache first; only fallback to remote if local lookup fails.
        requested_local_files_only = kwargs.pop("local_files_only", True)
        allow_remote_fallback = bool(kwargs.pop("allow_remote_fallback", True))
        if requested_local_files_only is not True:
            log.warning(
                "Ignoring local_files_only=%r and enforcing local-first dataset loading.",
                requested_local_files_only,
            )

        base_download_config = kwargs.pop("download_config", None)

        def _set_offline_env(enabled: bool):
            """Temporarily control HF offline flags for deterministic local-vs-remote behavior."""
            keys = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
            previous = {k: os.environ.get(k) for k in keys}
            for key in keys:
                if enabled:
                    os.environ[key] = "1"
                else:
                    # Keep explicit user-configured values if present; otherwise unset.
                    if previous[key] is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous[key]
            return previous

        def _restore_env(previous: dict):
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        def _load_with_local_setting(local_only: bool):
            previous_offline = _set_offline_env(enabled=local_only)
            if base_download_config is None:
                effective_download_config = DownloadConfig(local_files_only=local_only)
            else:
                # Preserve caller-provided config while controlling local/remote behavior.
                base_download_config.local_files_only = local_only
                effective_download_config = base_download_config
            try:
                if isinstance(path, str):
                    return path, load_dataset(path, split=split, download_config=effective_download_config, **kwargs)
                return path[0], load_dataset(*path, split=split, download_config=effective_download_config, **kwargs)
            finally:
                _restore_env(previous_offline)

        load_from_disk = kwargs.pop("load_from_disk", False)
        if load_from_disk:
            dataset_name = path
            dataset = hf_dataset.load_from_disk(path)
            log.info("HF dataset source: load_from_disk path=%s split=%s", path, split)
        else:
            try:
                dataset_name, dataset = _load_with_local_setting(local_only=True)
                log.info("HF dataset source: local_cache path=%s split=%s", path, split)
            except Exception as local_error:
                if not allow_remote_fallback:
                    raise
                log.warning(
                    "Local-only dataset load failed for path=%s split=%s (%s). Falling back to remote download.",
                    path,
                    split,
                    local_error,
                )
                dataset_name, dataset = _load_with_local_setting(local_only=False)
                log.info("HF dataset source: remote_fallback path=%s split=%s", path, split)

        return dataset_name, dataset

    @staticmethod
    def from_datasets(
        dataset_path: Union[str, List[str]],
        x_column: str,
        y_column: str,
        batch_size: int,
        im_column: Optional[str] = None,
        prompt: str = "",
        description: str = "",
        mmlu_max_subject_size: int = 100,
        n_shot: int = 0,
        few_shot_split: str = "train",
        few_shot_prompt: Optional[str] = None,
        instruct: bool = False,
        split: str = "test",
        size: int = None,
        **kwargs,
    ):
        """
        Creates the dataset from Huggingface datasets.

        Parameters:
            dataset_path (str): HF path to dataset,
            x_column (str): name of column to take input texts from,
            y_column (str): name of column to take target texts from,
            batch_size (int): the size of the texts batch,
            prompt (str): prompt template to use for input texts (default: ''),
            split (str): dataset split to take data from (default: 'text'),
            size (Optional[int]): size to subsample dataset to. If None, the full dataset split will be taken.
                Default: None.
        """
        # Internal formatting-only keys should never be forwarded to Hugging Face loader kwargs.
        prompt_without_context = kwargs.pop("prompt_without_context", "")
        pubmedqa_context_model_path = kwargs.pop("pubmedqa_context_model_path", None)
        dataset_format = str(kwargs.pop("dataset_format", "")).strip().lower()

        dataset_name, dataset = Dataset.load_hf_dataset(dataset_path, split, **kwargs)
        log.debug(f"Loaded HF dataset '{dataset_name}' split '{split}': {len(dataset)} samples")

        pubmedqa_with_context_prompts = None
        pubmedqa_without_context_prompts = None
        pubmedqa_context_only_prompts = None
        pubmedqa_questions = None
        pubmedqa_long_answers = None
        pubmedqa_context_texts = None
        race_with_context_prompts = None
        race_without_context_prompts = None

        if size is not None:
            log.info(f"Size parameter provided: {size}, dataset length: {len(dataset)}")
            if size < len(dataset):
                log.info(f"Selecting first {size} samples from dataset")
                dataset = dataset.select(range(size))
                log.info(f"After selection: {len(dataset)} samples")
            else:
                log.info(f"Size {size} >= dataset length {len(dataset)}, using full dataset")
        else:
            log.info("No size parameter provided, using full dataset")

        if "allenai/c4" in dataset_name.lower():
            x, y = [], []
            for inst in dataset:
                if len(inst[x_column]) <= 1024:
                    x.append(inst[x_column])
                    y.append(inst[y_column])
        elif (
            ("medqa" in dataset_name.lower() and "pubmedqa" not in dataset_name.lower())
            or (
                isinstance(dataset_path, str)
                and "medqa" in dataset_path.lower()
                and "pubmedqa" not in dataset_path.lower()
            )
        ):
            # Special handling for MedQA-USMLE-4-options format
            # Format: question + options as input, answer_idx as target
            log.debug("Detected MedQA dataset format, formatting question and options")
            x, y = [], []
            for inst in dataset:
                # Format the question and options
                question = inst.get("question", "")
                options = inst.get("options", {})
                option_a = options.get("A", "")
                option_b = options.get("B", "")
                option_c = options.get("C", "")
                option_d = options.get("D", "")
                
                # Format the prompt
                if prompt:
                    formatted_input = prompt.format(
                        question=question.strip(),
                        option_a=option_a,
                        option_b=option_b,
                        option_c=option_c,
                        option_d=option_d,
                        text=question.strip(),  # For backward compatibility
                    )
                else:
                    # Default format if no prompt provided
                    formatted_input = (
                        f"Q: {question.strip()}\n"
                        f"A. {option_a}\n"
                        f"B. {option_b}\n"
                        f"C. {option_c}\n"
                        f"D. {option_d}\n"
                    )
                
                if description:
                    formatted_input = description + "\n\n" + formatted_input
                
                x.append(formatted_input)
                
                # Use answer_idx as target (A, B, C, or D)
                if y_column:
                    y.append(inst.get(y_column, inst.get("answer_idx", "")))
                else:
                    y.append(inst.get("answer_idx", ""))
            
            log.debug(f"Formatted {len(x)} MedQA samples")
        elif "gsm8k-mc" in dataset_name.lower() or (isinstance(dataset_path, str) and "gsm8k-mc" in dataset_path.lower()):
            # Special handling for GSM8K-MC format
            # Format: Question, A, B, C, D, Answer columns (separate columns, not dict)
            log.debug("Detected GSM8K-MC dataset format, formatting question and options")
            x, y = [], []
            for inst in dataset:
                # Get question and options from separate columns
                question = inst.get("Question", inst.get("question", ""))
                option_a = inst.get("A", "")
                option_b = inst.get("B", "")
                option_c = inst.get("C", "")
                option_d = inst.get("D", "")
                
                # Format the prompt
                if prompt:
                    formatted_input = prompt.format(
                        question=question.strip() if question else "",
                        option_a=option_a if option_a else "",
                        option_b=option_b if option_b else "",
                        option_c=option_c if option_c else "",
                        option_d=option_d if option_d else "",
                        text=question.strip() if question else "",  # For backward compatibility
                    )
                else:
                    # Default format if no prompt provided
                    formatted_input = (
                        f"Q: {question.strip() if question else ''}\n"
                        f"A. {option_a if option_a else ''}\n"
                        f"B. {option_b if option_b else ''}\n"
                        f"C. {option_c if option_c else ''}\n"
                        f"D. {option_d if option_d else ''}\n"
                    )
                
                if description:
                    formatted_input = description + "\n\n" + formatted_input
                
                x.append(formatted_input)
                
                # Use Answer column as target (A, B, C, or D)
                if y_column:
                    answer = inst.get(y_column, inst.get("Answer", inst.get("answer", "")))
                    y.append(answer)
                else:
                    y.append(inst.get("Answer", inst.get("answer", "")))
            
            log.debug(f"Formatted {len(x)} GSM8K-MC samples")
        elif (
            dataset_format == "forecastqa"
            or "forecastqa" in dataset_name.lower()
            or (isinstance(dataset_path, str) and "forecastqa" in dataset_path.lower())
            or (isinstance(dataset_path, list) and any("forecastqa" in str(p).lower() for p in dataset_path))
        ):
            # Special handling for ForecastQA multichoice subset.
            # Format: question + choices[{label, choice}] with answer labels in 1-4.
            log.debug("Detected ForecastQA dataset format, formatting question and options")
            x, y = [], []
            skipped_count = 0

            for inst in dataset:
                question = str(inst.get("question", "")).strip()
                answer_raw = inst.get("answer", inst.get(y_column, None))

                choices = inst.get("choices", [])
                label_to_text = {}
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        raw_label = str(choice.get("label", "")).strip().upper()
                        raw_text = str(choice.get("choice", choice.get("text", ""))).strip()
                        if raw_label and raw_text:
                            label_to_text[raw_label] = raw_text

                option_a = label_to_text.get("1", label_to_text.get("A", ""))
                option_b = label_to_text.get("2", label_to_text.get("B", ""))
                option_c = label_to_text.get("3", label_to_text.get("C", ""))
                option_d = label_to_text.get("4", label_to_text.get("D", ""))

                answer_letter = None
                if isinstance(answer_raw, str) and answer_raw.strip() in {"1", "2", "3", "4"}:
                    answer_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}[answer_raw.strip()]
                elif isinstance(answer_raw, int) and answer_raw in {1, 2, 3, 4}:
                    answer_letter = {1: "A", 2: "B", 3: "C", 4: "D"}[answer_raw]
                else:
                    answer_letter = _normalize_multiple_choice_answer(answer_raw)
                if answer_letter is None or not question:
                    skipped_count += 1
                    continue

                choices_str = _format_choices_block(option_a, option_b, option_c, option_d)
                if not choices_str:
                    skipped_count += 1
                    continue

                if prompt:
                    if "{choices}" in prompt:
                        formatted_input = prompt.format(
                            question=question,
                            choices=choices_str,
                            option_a=option_a,
                            option_b=option_b,
                            option_c=option_c,
                            option_d=option_d,
                            text=question,
                        )
                    else:
                        formatted_input = prompt.format(
                            question=question,
                            option_a=option_a,
                            option_b=option_b,
                            option_c=option_c,
                            option_d=option_d,
                            text=question,
                        )
                else:
                    formatted_input = (
                        "Return the label of the correct answer for the question below.\n\n"
                        f"Question: {question}\n"
                        f"Choices:\n{choices_str}\n"
                        "Answer:"
                    )

                if description:
                    formatted_input = description + "\n\n" + formatted_input

                x.append(formatted_input)
                y.append(answer_letter)

            if len(x) == 0:
                raise ValueError(
                    "No valid ForecastQA samples found after processing. "
                    f"Skipped {skipped_count} samples."
                )

            log.debug(f"Formatted {len(x)} ForecastQA samples (skipped {skipped_count} invalid samples)")
        elif "ai2_arc" in dataset_name.lower() or (isinstance(dataset_path, str) and "ai2_arc" in dataset_path.lower()) or (isinstance(dataset_path, list) and any("ai2_arc" in str(p).lower() for p in dataset_path)):
            # Special handling for ARC dataset format
            # Format: question, choices (dict with A, B, C, D, etc.), answerKey
            log.debug("Detected ARC dataset format, formatting question and options")
            x, y = [], []
            valid_answer_keys = {"A", "B", "C", "D"}
            skipped_count = 0
            
            for inst in dataset:
                # Get answer key and filter to only A, B, C, D
                answer_key = inst.get("answerKey", "")
                if answer_key not in valid_answer_keys:
                    skipped_count += 1
                    continue
                
                # Get question and choices
                question = inst.get("question", "")
                choices = inst.get("choices", {})
                
                # Extract options - ARC dataset format: choices is a dict with "label" and "text" lists
                # Format: {"label": ["A", "B", "C", "D"], "text": ["option1", "option2", "option3", "option4"]}
                option_a = ""
                option_b = ""
                option_c = ""
                option_d = ""
                
                if isinstance(choices, dict):
                    # Check if it's the ARC format with "label" and "text" lists
                    if "label" in choices and "text" in choices and isinstance(choices["label"], list):
                        # ARC format: {"label": ["A", "B", "C", "D"], "text": ["...", "...", "...", "..."]}
                        labels = choices.get("label", [])
                        texts = choices.get("text", [])
                        for label, text in zip(labels, texts):
                            if label == "A":
                                option_a = text
                            elif label == "B":
                                option_b = text
                            elif label == "C":
                                option_c = text
                            elif label == "D":
                                option_d = text
                    elif isinstance(choices, list):
                        # Format: [{"label": "A", "text": "..."}, {"label": "B", "text": "..."}, ...]
                        for choice in choices:
                            if isinstance(choice, dict):
                                label = choice.get("label", "")
                                text = choice.get("text", "")
                                if label == "A":
                                    option_a = text
                                elif label == "B":
                                    option_b = text
                                elif label == "C":
                                    option_c = text
                                elif label == "D":
                                    option_d = text
                    else:
                        # Format: {"A": "...", "B": "...", ...}
                        option_a = choices.get("A", "")
                        option_b = choices.get("B", "")
                        option_c = choices.get("C", "")
                        option_d = choices.get("D", "")
                
                # Format the prompt
                # Support both old format (option_a, option_b, etc.) and new bayesian-peft format (choices)
                if prompt:
                    # Check if prompt uses {choices} format (bayesian-peft style)
                    if "{choices}" in prompt:
                        # Format choices as "A) option_text\nB) option_text\n..." to match bayesian-peft
                        choices_list = []
                        for label, text in [("A", option_a), ("B", option_b), ("C", option_c), ("D", option_d)]:
                            if text:  # Only include if option text is not empty
                                choices_list.append(f"{label}) {text}")
                        choices_str = "\n".join(choices_list)
                        formatted_input = prompt.format(
                            question=question.strip() if question else "",
                            choices=choices_str,
                            # Also support old format for backward compatibility
                            option_a=option_a if option_a else "",
                            option_b=option_b if option_b else "",
                            option_c=option_c if option_c else "",
                            option_d=option_d if option_d else "",
                            text=question.strip() if question else "",
                        )
                    else:
                        # Old format with individual options
                        formatted_input = prompt.format(
                            question=question.strip() if question else "",
                            option_a=option_a if option_a else "",
                            option_b=option_b if option_b else "",
                            option_c=option_c if option_c else "",
                            option_d=option_d if option_d else "",
                            text=question.strip() if question else "",  # For backward compatibility
                        )
                else:
                    # Default format: use bayesian-peft style as default for ARC
                    choices_list = []
                    for label, text in [("A", option_a), ("B", option_b), ("C", option_c), ("D", option_d)]:
                        if text:
                            choices_list.append(f"{label}) {text}")
                    choices_str = "\n".join(choices_list)
                    formatted_input = (
                        f"Return the label of the correct answer for the question below.\n\n"
                        f"Question: {question.strip() if question else ''}\n"
                        f"Choices:\n{choices_str}\n"
                        f"Answer:"
                    )
                
                if description:
                    formatted_input = description + "\n\n" + formatted_input
                
                x.append(formatted_input)
                
                # Use answerKey as target (A, B, C, or D)
                if y_column:
                    y.append(inst.get(y_column, answer_key))
                else:
                    y.append(answer_key)
            
            log.debug(f"Formatted {len(x)} ARC samples (skipped {skipped_count} with answer keys other than A, B, C, D)")
        elif "medmcqa" in dataset_name.lower() or (isinstance(dataset_path, str) and "medmcqa" in dataset_path.lower()):
            # Special handling for MedMCQA dataset format
            # Format: question + opa/opb/opc/opd as options, cop (0-3) as correct answer
            log.debug("Detected MedMCQA dataset format, formatting question and options")
            x, y = [], []
            option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            skipped_count = 0
            
            # Log first example to debug structure
            if len(dataset) > 0:
                first_example = dataset[0]
                log.debug(f"MedMCQA dataset sample keys: {list(first_example.keys())}")
                log.debug(f"MedMCQA first example sample: {first_example}")
            
            # Count cop value distribution for debugging
            cop_distribution = {}
            for inst in dataset:
                cop = inst.get("cop", None)
                if cop is not None:
                    cop_distribution[cop] = cop_distribution.get(cop, 0) + 1
            if cop_distribution:
                log.info(f"MedMCQA cop value distribution: {cop_distribution}")
            
            for inst in dataset:
                # Format the question and options
                # Try multiple possible column names
                question = inst.get("question", inst.get("Question", ""))
                
                # Try different option column name variations
                option_a = inst.get("opa", inst.get("option_a", inst.get("A", "")))
                option_b = inst.get("opb", inst.get("option_b", inst.get("B", "")))
                option_c = inst.get("opc", inst.get("option_c", inst.get("C", "")))
                option_d = inst.get("opd", inst.get("option_d", inst.get("D", "")))
                
                # Get correct option (cop is 0, 1, 2, 3 for A, B, C, D)
                # Note: cop = -1 means "no answer" or "unknown" and should be skipped
                # Try multiple possible column names
                cop = inst.get("cop", inst.get("correct_idx", inst.get("correct_option", None)))
                if cop is not None:
                    # Convert 0-3 to A-D
                    # Handle both int and string representations
                    if isinstance(cop, str):
                        try:
                            cop = int(cop)
                        except (ValueError, TypeError):
                            cop = None
                    # Skip if cop is -1 (no answer) or not in valid range (0-3)
                    if cop == -1:
                        skipped_count += 1
                        continue
                    if cop is not None and cop in option_map:
                        answer_letter = option_map[cop]
                    else:
                        skipped_count += 1
                        continue
                else:
                    # Fallback to answer_idx if cop is not available
                    answer_idx = inst.get("answer_idx", inst.get("answer", ""))
                    if isinstance(answer_idx, int):
                        if answer_idx in option_map:
                            answer_letter = option_map[answer_idx]
                        else:
                            skipped_count += 1
                            continue
                    elif isinstance(answer_idx, str) and answer_idx.upper() in ["A", "B", "C", "D"]:
                        answer_letter = answer_idx.upper()
                    else:
                        skipped_count += 1
                        continue
                
                # Skip if we don't have valid question or options
                if not question or not (option_a or option_b or option_c or option_d):
                    skipped_count += 1
                    continue
                
                # Format the prompt
                # Support both old format (option_a, option_b, etc.) and new bayesian-peft format (choices)
                if prompt:
                    # Check if prompt uses {choices} format (bayesian-peft style)
                    if "{choices}" in prompt:
                        # Format choices as "A) option_text\nB) option_text\n..." to match bayesian-peft
                        choices_list = []
                        for label, text in [("A", option_a), ("B", option_b), ("C", option_c), ("D", option_d)]:
                            if text:  # Only include if option text is not empty
                                choices_list.append(f"{label}) {text}")
                        choices_str = "\n".join(choices_list)
                        try:
                            formatted_input = prompt.format(
                                question=question.strip() if question else "",
                                choices=choices_str,
                                # Also support old format for backward compatibility
                                option_a=option_a if option_a else "",
                                option_b=option_b if option_b else "",
                                option_c=option_c if option_c else "",
                                option_d=option_d if option_d else "",
                                text=question.strip() if question else "",
                            )
                        except KeyError as e:
                            log.warning(f"Prompt format error: {e}, using default format")
                            formatted_input = (
                                f"Q: {question.strip() if question else ''}\n"
                                f"A. {option_a if option_a else ''}\n"
                                f"B. {option_b if option_b else ''}\n"
                                f"C. {option_c if option_c else ''}\n"
                                f"D. {option_d if option_d else ''}\n"
                            )
                    else:
                        # Old format with individual options
                        try:
                            formatted_input = prompt.format(
                                question=question.strip() if question else "",
                                option_a=option_a if option_a else "",
                                option_b=option_b if option_b else "",
                                option_c=option_c if option_c else "",
                                option_d=option_d if option_d else "",
                                text=question.strip() if question else "",  # For backward compatibility
                            )
                        except KeyError as e:
                            log.warning(f"Prompt format error: {e}, using default format")
                            formatted_input = (
                                f"Q: {question.strip() if question else ''}\n"
                                f"A. {option_a if option_a else ''}\n"
                                f"B. {option_b if option_b else ''}\n"
                                f"C. {option_c if option_c else ''}\n"
                                f"D. {option_d if option_d else ''}\n"
                            )
                else:
                    # Default format: use bayesian-peft style as default
                    choices_list = []
                    for label, text in [("A", option_a), ("B", option_b), ("C", option_c), ("D", option_d)]:
                        if text:
                            choices_list.append(f"{label}) {text}")
                    choices_str = "\n".join(choices_list)
                    formatted_input = (
                        f"Answer the multiple choice question below by returning the answer label (A to D)\n\n"
                        f"Question: {question.strip() if question else ''}\n"
                        f"Choices:\n{choices_str}\n"
                        f"Answer:"
                    )
                
                if description:
                    formatted_input = description + "\n\n" + formatted_input
                
                x.append(formatted_input)
                
                # Use answer_letter as target (A, B, C, or D)
                y.append(answer_letter)
            
            if len(x) == 0:
                # Check if all samples had cop=-1 (no answer)
                all_cop_neg_one = all(
                    inst.get("cop", None) == -1 
                    for inst in dataset
                )
                if all_cop_neg_one:
                    raise ValueError(
                        f"No valid MedMCQA samples found. All {skipped_count} samples have cop=-1 (no answer). "
                        f"The '{split}' split may not have labels. "
                        f"Try using '--split validation' or '--split train' instead, "
                        f"or check if the dataset split has labeled examples."
                    )
                else:
                    raise ValueError(
                        f"No valid MedMCQA samples found after processing. "
                        f"Skipped {skipped_count} samples. "
                        f"Please check the dataset format and column names. "
                        f"Expected cop values: 0, 1, 2, 3 (for A, B, C, D). "
                        f"cop=-1 indicates no answer and is skipped."
                    )
            log.debug(f"Formatted {len(x)} MedMCQA samples (skipped {skipped_count} invalid samples)")
        elif "race" in dataset_name.lower() or (isinstance(dataset_path, str) and "race" in dataset_path.lower()):
            # Special handling for RACE-style per-article format.
            # We keep only the first question per article and build two prompt variants:
            # 1) with article context, 2) without article context.
            log.debug("Detected RACE dataset format, formatting first question per article")
            x, y = [], []
            race_with_context_prompts = []
            race_without_context_prompts = []
            skipped_count = 0

            for inst in dataset:
                article = str(inst.get("article", "")).strip()

                # EleutherAI/race stores questions under `problems` per article.
                first_problem = None
                problems = inst.get("problems", None)
                if isinstance(problems, str) and problems.strip():
                    parsed_problems = None
                    try:
                        parsed_problems = ast.literal_eval(problems)
                    except (SyntaxError, ValueError):
                        try:
                            parsed_problems = json.loads(problems)
                        except (TypeError, ValueError):
                            parsed_problems = None
                    problems = parsed_problems
                if isinstance(problems, list) and len(problems) > 0:
                    if isinstance(problems[0], dict):
                        first_problem = problems[0]

                if first_problem is not None:
                    question = str(first_problem.get("question", "")).strip()
                    options = first_problem.get("options", [])
                    answer_raw = first_problem.get("answer", None)
                else:
                    # Fallback for flattened RACE variants.
                    question = str(inst.get("question", "")).strip()
                    options = inst.get("options", [])
                    answer_raw = inst.get("answer", None)

                if not isinstance(options, list):
                    skipped_count += 1
                    continue

                option_a = str(options[0]).strip() if len(options) > 0 else ""
                option_b = str(options[1]).strip() if len(options) > 1 else ""
                option_c = str(options[2]).strip() if len(options) > 2 else ""
                option_d = str(options[3]).strip() if len(options) > 3 else ""

                answer_letter = _normalize_multiple_choice_answer(answer_raw)
                if answer_letter is None:
                    skipped_count += 1
                    continue

                if not question or not (option_a or option_b or option_c or option_d):
                    skipped_count += 1
                    continue

                choices_str = _format_choices_block(option_a, option_b, option_c, option_d)

                if prompt:
                    try:
                        with_context_prompt = prompt.format(
                            question=question,
                            article=article,
                            context=article,
                            choices=choices_str,
                            option_a=option_a,
                            option_b=option_b,
                            option_c=option_c,
                            option_d=option_d,
                            text=question,
                        )
                    except KeyError as e:
                        log.warning(f"RACE prompt format error: {e}. Falling back to default with-context prompt.")
                        with_context_prompt = (
                            "Return the label of the correct answer for the question below.\n\n"
                            f"Article:\n{article}\n\n"
                            f"Question: {question}\n"
                            f"Choices:\n{choices_str}\n"
                            "Answer:"
                        )
                else:
                    with_context_prompt = (
                        "Return the label of the correct answer for the question below.\n\n"
                        f"Article:\n{article}\n\n"
                        f"Question: {question}\n"
                        f"Choices:\n{choices_str}\n"
                        "Answer:"
                    )

                if prompt_without_context:
                    try:
                        without_context_prompt = prompt_without_context.format(
                            question=question,
                            choices=choices_str,
                            option_a=option_a,
                            option_b=option_b,
                            option_c=option_c,
                            option_d=option_d,
                            text=question,
                        )
                    except KeyError as e:
                        log.warning(
                            f"RACE prompt_without_context format error: {e}. "
                            "Falling back to default without-context prompt."
                        )
                        without_context_prompt = (
                            "Return the label of the correct answer for the question below.\n\n"
                            f"Question: {question}\n"
                            f"Choices:\n{choices_str}\n"
                            "Answer:"
                        )
                else:
                    without_context_prompt = (
                        "Return the label of the correct answer for the question below.\n\n"
                        f"Question: {question}\n"
                        f"Choices:\n{choices_str}\n"
                        "Answer:"
                    )

                if description:
                    with_context_prompt = description + "\n\n" + with_context_prompt
                    without_context_prompt = description + "\n\n" + without_context_prompt

                # Use no-context prompt by default; model-specific routing can switch to article context.
                x.append(without_context_prompt)
                y.append(answer_letter)
                race_with_context_prompts.append(with_context_prompt)
                race_without_context_prompts.append(without_context_prompt)

            if len(x) == 0:
                raise ValueError(
                    "No valid RACE samples found after processing. "
                    f"Skipped {skipped_count} samples. "
                    "Expected per-article rows with `problems` containing at least one question, "
                    "4 options, and answer labels in A/B/C/D or 0-3 format."
                )

            log.debug(
                f"Formatted {len(x)} RACE samples using first question per article "
                f"(skipped {skipped_count} invalid samples)"
            )
        elif "pubmedqa" in dataset_name.lower() or (isinstance(dataset_path, str) and "pubmedqa" in dataset_path.lower()):
            # Special handling for PubMedQA format
            # Check if this is a no-context variant
            is_pubmedqa_no_context = (
                "no_context" in dataset_name.lower() or
                "no-context" in dataset_name.lower() or
                (isinstance(dataset_path, str) and ("no_context" in dataset_path.lower() or "no-context" in dataset_path.lower()))
            )
            
            if is_pubmedqa_no_context:
                # For no-context variant: only create no-context prompts, skip with-context creation
                log.debug("Detected PubMedQA no-context variant, formatting question/long_answer prompts without context")
            else:
                # For standard PubMedQA: keep two prompt variants:
                #   1) Full: question + context + long answer
                #   2) Reduced: question + long answer
                # The reduced variant is stored in dataset.x by default, and callers can
                # choose the full variant for a sampled model.
                log.debug("Detected PubMedQA dataset format, formatting question/context/long_answer prompts")
            
            x, y = [], []
            pubmedqa_with_context_prompts = []
            pubmedqa_without_context_prompts = []
            pubmedqa_context_only_prompts = []
            pubmedqa_questions = []
            pubmedqa_long_answers = []
            pubmedqa_context_texts = []
            skipped_count = 0

            for inst in dataset:
                question = _stringify_pubmedqa_field(inst.get("question", ""))
                long_answer = _stringify_pubmedqa_field(inst.get("long_answer", inst.get("answer", "")))
                context_text = _format_pubmedqa_context(inst.get("context", ""))

                # Normalize labels to YES/NO and skip unsupported labels (e.g., maybe)
                raw_label = inst.get(y_column, inst.get("final_decision", None))
                normalized_label = _normalize_pubmedqa_label(raw_label)
                if normalized_label is None:
                    skipped_count += 1
                    continue

                if not question or not long_answer:
                    skipped_count += 1
                    continue

                # For no-context variant, skip with-context prompt creation
                if not is_pubmedqa_no_context:
                    if prompt:
                        try:
                            with_context_prompt = prompt.format(
                                question=question,
                                context=context_text,
                                long_answer=long_answer,
                                text=question,
                                answer=long_answer,
                            )
                        except KeyError as e:
                            log.warning(f"PubMedQA prompt format error: {e}. Falling back to default prompt.")
                            with_context_prompt = _build_pubmedqa_prompt(
                                question=question,
                                long_answer=long_answer,
                                context_text=context_text,
                                include_context=True,
                            )
                    else:
                        with_context_prompt = _build_pubmedqa_prompt(
                            question=question,
                            long_answer=long_answer,
                            context_text=context_text,
                            include_context=True,
                        )
                else:
                    with_context_prompt = None

                if prompt_without_context:
                    try:
                        without_context_prompt = prompt_without_context.format(
                            question=question,
                            context=context_text,
                            long_answer=long_answer,
                            text=question,
                            answer=long_answer,
                        )
                    except KeyError as e:
                        log.warning(
                            f"PubMedQA prompt_without_context format error: {e}. Falling back to default reduced prompt."
                        )
                        without_context_prompt = _build_pubmedqa_prompt(
                            question=question,
                            long_answer=long_answer,
                            context_text=context_text,
                            include_context=False,
                        )
                else:
                    without_context_prompt = _build_pubmedqa_prompt(
                        question=question,
                        long_answer=long_answer,
                        context_text=context_text,
                        include_context=False,
                    )

                context_only_prompt = _build_pubmedqa_context_only_prompt(
                    question=question,
                    context_text=context_text,
                )

                if description:
                    with_context_prompt = description + "\n\n" + with_context_prompt if with_context_prompt else with_context_prompt
                    without_context_prompt = description + "\n\n" + without_context_prompt
                    context_only_prompt = description + "\n\n" + context_only_prompt

                # Use reduced prompt by default; caller can switch to full prompt per model.
                x.append(without_context_prompt)
                y.append(normalized_label)
                # For standard pubmedqa, store both variants; for no-context, only store without-context
                if not is_pubmedqa_no_context:
                    pubmedqa_with_context_prompts.append(with_context_prompt)
                    pubmedqa_without_context_prompts.append(without_context_prompt)
                    pubmedqa_context_only_prompts.append(context_only_prompt)
                    pubmedqa_questions.append(question)
                    pubmedqa_long_answers.append(long_answer)
                    pubmedqa_context_texts.append(context_text)

            if len(x) == 0:
                raise ValueError(
                    "No valid PubMedQA samples found after processing. "
                    f"Skipped {skipped_count} samples. "
                    "Expected final_decision labels containing yes/no values."
                )

            log.debug(
                f"Formatted {len(x)} PubMedQA samples (skipped {skipped_count} unsupported samples)"
            )
        elif "cais/mmlu" in dataset_name.lower() or (isinstance(dataset_path, str) and "cais/mmlu" in dataset_path.lower()) or (isinstance(dataset_path, list) and any("cais/mmlu" in str(p).lower() for p in dataset_path)):
            # Special handling for cais/mmlu dataset format
            # Format: question + choices (A, B, C, D) as lists, answer as integer (0-3)
            log.debug("Detected cais/mmlu dataset format, formatting question and options")
            x, y = [], []
            option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            skipped_count = 0
            
            # Log first example to debug structure
            if len(dataset) > 0:
                first_example = dataset[0]
                log.debug(f"cais/mmlu dataset sample keys: {list(first_example.keys())}")
                log.debug(f"cais/mmlu first example sample: {first_example}")
            
            for inst in dataset:
                # Format the question and options
                question = inst.get("question", "")
                choices = inst.get("choices", [])
                answer = inst.get("answer", None)
                
                # Handle answer: can be integer (0-3) or already a letter
                if answer is None:
                    skipped_count += 1
                    continue
                
                if isinstance(answer, int):
                    if answer < 0 or answer >= len(choices):
                        skipped_count += 1
                        continue
                    answer_letter = option_map.get(answer, "")
                    if not answer_letter:
                        skipped_count += 1
                        continue
                elif isinstance(answer, str):
                    answer_letter = answer.upper()
                    if answer_letter not in {"A", "B", "C", "D"}:
                        skipped_count += 1
                        continue
                else:
                    skipped_count += 1
                    continue
                
                # Extract options - handle list of choices
                options = {}
                if isinstance(choices, list) and len(choices) >= 4:
                    options = {
                        "A": choices[0],
                        "B": choices[1],
                        "C": choices[2],
                        "D": choices[3],
                    }
                elif isinstance(choices, dict):
                    options = choices
                else:
                    skipped_count += 1
                    continue
                
                # Format the prompt
                # Support both old format (option_a, option_b, etc.) and new bayesian-peft format (choices)
                if prompt:
                    # Check if prompt uses {option_a} format or {choices} format
                    if "{choices}" in prompt:
                        # Format choices as "A) ... B) ... C) ... D) ..."
                        choices_str = "\n".join([f"{k}) {options.get(k, '')}" for k in ["A", "B", "C", "D"]])
                        formatted_input = prompt.format(
                            question=question,
                            choices=choices_str,
                            option_a=options.get("A", ""),
                            option_b=options.get("B", ""),
                            option_c=options.get("C", ""),
                            option_d=options.get("D", ""),
                        )
                    else:
                        # Try to format with individual options
                        formatted_input = prompt.format(
                            question=question,
                            option_a=options.get("A", ""),
                            option_b=options.get("B", ""),
                            option_c=options.get("C", ""),
                            option_d=options.get("D", ""),
                        )
                else:
                    # Default format if no prompt template provided
                    choices_str = "\n".join([f"{k}) {options.get(k, '')}" for k in ["A", "B", "C", "D"]])
                    formatted_input = f"Question: {question}\n{choices_str}"
                
                if description:
                    formatted_input = description + "\n\n" + formatted_input
                
                x.append(formatted_input)
                
                # Use answer_letter as target
                if y_column:
                    y.append(inst.get(y_column, answer_letter))
                else:
                    y.append(answer_letter)
            
            if len(x) == 0:
                raise ValueError(
                    f"No valid cais/mmlu samples found after processing. "
                    f"Skipped {skipped_count} samples. "
                    f"Please check the dataset format and column names. "
                    f"Expected columns: 'question', 'choices' (list), 'answer' (int 0-3). "
                    f"Split: {split}"
                )
            
            log.debug(f"Formatted {len(x)} cais/mmlu samples (skipped {skipped_count} invalid samples)")
        else:
            # Convert to lists to ensure we only have the selected samples
            # This is important because dataset[x_column] on a lazy dataset might access all rows
            log.debug(f"Converting dataset to lists (current length: {len(dataset)})")
            x = list(dataset[x_column])
            if y_column is not None:
                y = list(dataset[y_column])
            else:
                y = ["" for _ in range(len(x))]
            log.info(f"Extracted {len(x)} samples from dataset")

        images = dataset[im_column] if im_column else None
        if images is not None:
            images = list(images)

        formatted_dataset = Dataset(x, y, batch_size, images=images)

        # Only set mixed_context strategy for standard PubMedQA (not for no-context variant)
        if (pubmedqa_with_context_prompts is not None and 
            pubmedqa_without_context_prompts is not None and
            len(pubmedqa_with_context_prompts) > 0):
            # Store both prompt variants to support model-specific prompt selection.
            formatted_dataset.pubmedqa_with_context_x = pubmedqa_with_context_prompts
            formatted_dataset.pubmedqa_without_context_x = pubmedqa_without_context_prompts
            formatted_dataset.pubmedqa_context_only_x = pubmedqa_context_only_prompts
            formatted_dataset.pubmedqa_prompt_strategy = "mixed_context"
            formatted_dataset.pubmedqa_context_model_path = pubmedqa_context_model_path
            # Preserve the original prompt templates so we can re-render context-bearing prompts
            # (e.g., "wrong context" routing) without parsing prompt text.
            formatted_dataset.pubmedqa_prompt_template_with_context = str(prompt or "")
            formatted_dataset.pubmedqa_prompt_template_without_context = str(prompt_without_context or "")
            # Keep raw fields for context re-rendering.
            if pubmedqa_questions is not None:
                formatted_dataset.pubmedqa_questions = list(pubmedqa_questions)
            if pubmedqa_long_answers is not None:
                formatted_dataset.pubmedqa_long_answers = list(pubmedqa_long_answers)
            if pubmedqa_context_texts is not None:
                formatted_dataset.pubmedqa_context_texts = list(pubmedqa_context_texts)

        if (
            race_with_context_prompts is not None
            and race_without_context_prompts is not None
            and len(race_with_context_prompts) > 0
        ):
            # Store both prompt variants for per-model article-context routing.
            formatted_dataset.race_with_context_x = race_with_context_prompts
            formatted_dataset.race_without_context_x = race_without_context_prompts
            formatted_dataset.race_prompt_strategy = "article_context_mixed"

        return formatted_dataset

    @staticmethod
    def load(path_or_path_and_files: Union[str, List[str]], *args, **kwargs):
        """
        Creates the dataset from either local .csv path (if such exists) or Huggingface datasets.
        See `from_csv` and `from_datasets` static functions for the description of *args and **kwargs arguments.

        Parameters:
            path_or_path_and_files (str or List[str]): local path to .csv table or HF path to dataset.
        """
        if isinstance(path_or_path_and_files, str) and os.path.isfile(
            path_or_path_and_files
        ):
            return Dataset.from_csv(path_or_path_and_files, *args, **kwargs)
        return Dataset.from_datasets(path_or_path_and_files, *args, **kwargs)

    @staticmethod
    def get_images(images: List[Union[Image.Image, str, bytes]]):
        imgs: List[Image.Image] = []
        for image_input in images:
            try:
                if isinstance(image_input, Image.Image):
                    imgs.append(image_input.convert("RGB"))
                elif isinstance(image_input, str) and image_input.startswith("http"):
                    response = requests.get(image_input, stream=True, timeout=10)
                    response.raise_for_status()
                    imgs.append(Image.open(io.BytesIO(response.content)).convert("RGB"))
                elif isinstance(image_input, str):
                    imgs.append(Image.open(image_input).convert("RGB"))
                elif isinstance(image_input, (bytes, bytearray)):
                    imgs.append(Image.open(io.BytesIO(image_input)).convert("RGB"))
                else:
                    log.warning(f"Unsupported image input format: {type(image_input)}")
            except Exception as e:
                log.warning(f"Failed to load image '{image_input}': {e}")
        return imgs
