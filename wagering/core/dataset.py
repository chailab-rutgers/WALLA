import os
import inspect
import importlib
import pandas as pd
import numpy as np
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

_DEFAULT_MC_PROMPT = (
    "Return the label of the correct answer for the question below.\n\n"
    "Question: {question}\nChoices:\n{choices}\nAnswer:"
)


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


def _choices_string(options: dict) -> str:
    return "\n".join(
        f"{label}) {options[label]}"
        for label in ("A", "B", "C", "D")
        if options.get(label)
    )


def _format_mc_prompt(
    question: str,
    options: dict,
    prompt: str,
    description: str = "",
) -> str:
    template = prompt or _DEFAULT_MC_PROMPT
    formatted = template.format(
        question=question.strip(),
        choices=_choices_string(options),
    )
    if description:
        return description + "\n\n" + formatted
    return formatted


def _dataset_path_matches(dataset_path: Union[str, List[str]], needle: str) -> bool:
    needle = needle.lower()
    if isinstance(dataset_path, str):
        return needle in dataset_path.lower()
    return any(needle in str(part).lower() for part in dataset_path)


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

            module = importlib.import_module(module_path)
            helper = getattr(module, function_name, None)
            if helper is None or not callable(helper):
                raise ValueError(
                    f"{key_name} references missing or non-callable '{function_name}' in module '{module_path}'"
                )
            return helper

        def _render_with_helper(helper, row: dict, key_name: str) -> str:
            signature = inspect.signature(helper)
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
                formatted_x.append(prompt.format(**format_kwargs))
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
                formatted_without_context.append(prompt_without_context.format(**format_kwargs))
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

        kwargs.pop("local_files_only", None)
        allow_remote_fallback = bool(kwargs.pop("allow_remote_fallback", True))
        base_download_config = kwargs.pop("download_config", None)

        def _set_offline_env(enabled: bool):
            keys = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
            previous = {k: os.environ.get(k) for k in keys}
            for key in keys:
                if enabled:
                    os.environ[key] = "1"
                elif previous[key] is None:
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
                base_download_config.local_files_only = local_only
                effective_download_config = base_download_config
            try:
                if isinstance(path, str):
                    return path, load_dataset(
                        path,
                        split=split,
                        download_config=effective_download_config,
                        **kwargs,
                    )
                return path[0], load_dataset(
                    *path,
                    split=split,
                    download_config=effective_download_config,
                    **kwargs,
                )
            finally:
                _restore_env(previous_offline)

        try:
            return _load_with_local_setting(local_only=True)
        except Exception:
            if not allow_remote_fallback:
                raise
            return _load_with_local_setting(local_only=False)

    @staticmethod
    def _format_arc_dataset(dataset, prompt: str, description: str, y_column: str):
        x, y = [], []
        valid_answer_keys = {"A", "B", "C", "D"}

        for inst in dataset:
            answer_key = inst.get("answerKey", "")
            if answer_key not in valid_answer_keys:
                continue

            question = inst.get("question", "")
            choices = inst.get("choices", {})
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            options = {
                label: text
                for label, text in zip(labels, texts)
                if label in valid_answer_keys
            }

            x.append(_format_mc_prompt(question, options, prompt, description))
            y.append(inst.get(y_column, answer_key) if y_column else answer_key)

        return x, y

    @staticmethod
    def _format_medmcqa_dataset(dataset, prompt: str, description: str):
        x, y = [], []
        option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        skipped_count = 0

        for inst in dataset:
            question = inst.get("question", "")
            options = {
                "A": inst.get("opa", ""),
                "B": inst.get("opb", ""),
                "C": inst.get("opc", ""),
                "D": inst.get("opd", ""),
            }

            cop = inst.get("cop", None)
            if isinstance(cop, str):
                cop = int(cop)
            if cop == -1 or cop not in option_map:
                skipped_count += 1
                continue

            answer_letter = option_map[cop]
            if not question or not any(options.values()):
                skipped_count += 1
                continue

            x.append(_format_mc_prompt(question, options, prompt, description))
            y.append(answer_letter)

        if not x:
            if all(inst.get("cop", None) == -1 for inst in dataset):
                raise ValueError(
                    f"No valid MedMCQA samples found. All {skipped_count} samples have cop=-1 (no answer). "
                    f"Try a labeled split such as validation or train."
                )
            raise ValueError(
                f"No valid MedMCQA samples found after processing. Skipped {skipped_count} samples."
            )

        return x, y

    @staticmethod
    def _format_pubmedqa_dataset(
        dataset,
        y_column: str,
        prompt: str,
        prompt_without_context: str,
        description: str,
    ):
        x, y = [], []
        with_context_prompts = []
        without_context_prompts = []
        questions = []
        long_answers = []
        context_texts = []
        skipped_count = 0

        if not str(prompt or "").strip():
            raise ValueError("PubMedQA requires a non-empty prompt template")
        if not str(prompt_without_context or "").strip():
            raise ValueError("PubMedQA requires a non-empty prompt_without_context template")

        for inst in dataset:
            question = _stringify_pubmedqa_field(inst.get("question", ""))
            long_answer = _stringify_pubmedqa_field(
                inst.get("long_answer", inst.get("answer", ""))
            )
            context_text = _format_pubmedqa_context(inst.get("context", ""))

            raw_label = inst.get(y_column, inst.get("final_decision", None))
            normalized_label = _normalize_pubmedqa_label(raw_label)
            if normalized_label is None or not question or not long_answer:
                skipped_count += 1
                continue

            with_context_prompt = prompt.format(
                question=question,
                context=context_text,
                long_answer=long_answer,
                text=question,
                answer=long_answer,
            )

            without_context_prompt = prompt_without_context.format(
                question=question,
                context=context_text,
                long_answer=long_answer,
                text=question,
                answer=long_answer,
            )

            if description:
                with_context_prompt = description + "\n\n" + with_context_prompt
                without_context_prompt = description + "\n\n" + without_context_prompt

            x.append(without_context_prompt)
            y.append(normalized_label)
            with_context_prompts.append(with_context_prompt)
            without_context_prompts.append(without_context_prompt)
            questions.append(question)
            long_answers.append(long_answer)
            context_texts.append(context_text)

        if not x:
            raise ValueError(
                "No valid PubMedQA samples found after processing. "
                f"Skipped {skipped_count} samples. "
                "Expected final_decision labels containing yes/no values."
            )

        metadata = {
            "pubmedqa_with_context_prompts": with_context_prompts,
            "pubmedqa_without_context_prompts": without_context_prompts,
            "pubmedqa_questions": questions,
            "pubmedqa_long_answers": long_answers,
            "pubmedqa_context_texts": context_texts,
        }
        return x, y, metadata

    @staticmethod
    def _format_mmlu_dataset(dataset, prompt: str, description: str, y_column: str):
        x, y = [], []
        option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        skipped_count = 0

        for inst in dataset:
            question = inst.get("question", "")
            choices = inst.get("choices", [])
            answer = inst.get("answer", None)
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
            elif isinstance(answer, str) and answer.upper() in {"A", "B", "C", "D"}:
                answer_letter = answer.upper()
            else:
                skipped_count += 1
                continue

            if not isinstance(choices, list) or len(choices) < 4:
                skipped_count += 1
                continue

            options = {
                "A": choices[0],
                "B": choices[1],
                "C": choices[2],
                "D": choices[3],
            }
            x.append(_format_mc_prompt(question, options, prompt, description))
            y.append(inst.get(y_column, answer_letter) if y_column else answer_letter)

        if not x:
            raise ValueError(
                f"No valid cais/mmlu samples found after processing. Skipped {skipped_count} samples."
            )

        return x, y

    @staticmethod
    def from_datasets(
        dataset_path: Union[str, List[str]],
        x_column: str,
        y_column: str,
        batch_size: int,
        im_column: Optional[str] = None,
        prompt: str = "",
        description: str = "",
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
        prompt_without_context = kwargs.pop("prompt_without_context", "")

        dataset_name, dataset = Dataset.load_hf_dataset(dataset_path, split, **kwargs)

        if size is not None and size < len(dataset):
            dataset = dataset.select(range(size))

        dataset_name_lower = dataset_name.lower()
        pubmedqa_metadata = None

        if "ai2_arc" in dataset_name_lower or _dataset_path_matches(dataset_path, "ai2_arc"):
            x, y = Dataset._format_arc_dataset(dataset, prompt, description, y_column)
        elif "medmcqa" in dataset_name_lower or _dataset_path_matches(dataset_path, "medmcqa"):
            x, y = Dataset._format_medmcqa_dataset(dataset, prompt, description)
        elif "pubmedqa" in dataset_name_lower or _dataset_path_matches(dataset_path, "pubmedqa"):
            x, y, pubmedqa_metadata = Dataset._format_pubmedqa_dataset(
                dataset,
                y_column,
                prompt,
                prompt_without_context,
                description,
            )
        elif "cais/mmlu" in dataset_name_lower or _dataset_path_matches(dataset_path, "cais/mmlu"):
            x, y = Dataset._format_mmlu_dataset(dataset, prompt, description, y_column)
        else:
            x = list(dataset[x_column])
            y = list(dataset[y_column]) if y_column is not None else [""] * len(x)

        images = dataset[im_column] if im_column else None
        if images is not None:
            images = list(images)

        formatted_dataset = Dataset(x, y, batch_size, images=images)

        if pubmedqa_metadata is not None:
            formatted_dataset.pubmedqa_with_context_x = pubmedqa_metadata[
                "pubmedqa_with_context_prompts"
            ]
            formatted_dataset.pubmedqa_without_context_x = pubmedqa_metadata[
                "pubmedqa_without_context_prompts"
            ]
            formatted_dataset.pubmedqa_prompt_strategy = "mixed_context"
            formatted_dataset.pubmedqa_prompt_template_with_context = str(prompt or "")
            formatted_dataset.pubmedqa_prompt_template_without_context = str(
                prompt_without_context or ""
            )
            formatted_dataset.pubmedqa_questions = list(
                pubmedqa_metadata["pubmedqa_questions"]
            )
            formatted_dataset.pubmedqa_long_answers = list(
                pubmedqa_metadata["pubmedqa_long_answers"]
            )
            formatted_dataset.pubmedqa_context_texts = list(
                pubmedqa_metadata["pubmedqa_context_texts"]
            )

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
