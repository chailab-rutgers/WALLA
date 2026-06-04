import importlib.util
from typing import Any


def load_external_module(path_to_file: str) -> Any:
    """Load an external Python module from a file path."""
    spec = importlib.util.spec_from_file_location("external_module", path_to_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {path_to_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
