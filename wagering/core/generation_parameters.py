from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationParameters:
    """Parameters that control generation behavior."""

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = False
    num_beams: int = 1
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop_strings: Optional[list] = None
    allow_newlines: bool = True
    max_new_tokens: int = 100


class GenerationParametersFactory:
    """Factory for creating GenerationParameters from YAML/native config dictionaries."""

    @staticmethod
    def from_params(
        yaml_config: Optional[dict] = None,
        native_config: Optional[dict] = None,
    ) -> GenerationParameters:
        yaml_config = yaml_config or {}
        native_config = native_config or {}
        params: dict = {}

        for name in GenerationParameters.__dataclass_fields__.keys():
            if name in yaml_config and yaml_config[name] is not None:
                params[name] = yaml_config[name]
            elif name in native_config and native_config[name] is not None:
                params[name] = native_config[name]

        return GenerationParameters(**params)
