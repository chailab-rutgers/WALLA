"""Shared runtime helpers for wagering CLI scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def ensure_project_venv() -> None:
    """Re-exec into the project venv when available."""
    if not VENV_PYTHON.exists():
        return
    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)


def parse_gpu_ids(csv: str) -> List[str]:
    gpu_ids = [p.strip() for p in str(csv).split(",") if p.strip()]
    if not gpu_ids:
        raise ValueError("No GPUs provided. Example: --gpus 0,1,2,3")
    return gpu_ids


def visible_gpu_count() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and str(raw).strip() != "":
        return len(parse_gpu_ids(str(raw)))
    import torch

    return int(torch.cuda.device_count())


def resolve_visible_gpu_ids(gpus_arg: Optional[str]) -> List[str]:
    if gpus_arg is not None:
        return parse_gpu_ids(gpus_arg)
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and str(raw).strip() != "":
        return parse_gpu_ids(str(raw))
    count = visible_gpu_count()
    return [str(i) for i in range(count)]


def require_visible_gpu_ids(gpus_arg: Optional[str]) -> List[str]:
    ids = resolve_visible_gpu_ids(gpus_arg)
    if not ids:
        raise ValueError(
            "No CUDA devices visible. Set --gpus or CUDA_VISIBLE_DEVICES, or install PyTorch with CUDA."
        )
    return ids


def wandb_disabled_env(base_env: Dict[str, str]) -> Dict[str, str]:
    env = dict(base_env)
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_SILENT", "true")
    return env


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def vary_shuffle_seed(cfg: Dict[str, Any], repeat_idx: int) -> Dict[str, Any]:
    out = dict(cfg)
    base_shuffle = int(out.get("shuffle_seed", 42))
    out["shuffle_seed"] = base_shuffle + int(repeat_idx)
    return out


def run_subprocess(cmd: Sequence[str], *, env: Dict[str, str], cwd: Path) -> int:
    proc = subprocess.run(list(cmd), cwd=str(cwd), env=env)
    return int(proc.returncode)


class ParallelGpuRunner:
    """Run jobs concurrently with one GPU slot per in-flight job."""

    def __init__(
        self,
        *,
        gpu_ids: List[str],
        max_workers_per_gpu: int,
        max_jobs: int,
    ) -> None:
        if max_workers_per_gpu <= 0:
            raise ValueError("max_workers_per_gpu must be > 0")
        if not gpu_ids:
            raise ValueError("gpu_ids must be non-empty")
        self.gpu_queue: Queue[str] = Queue()
        for gpu_id in gpu_ids:
            for _ in range(max_workers_per_gpu):
                self.gpu_queue.put(gpu_id)
        self.max_workers = min(max_jobs, len(gpu_ids) * max_workers_per_gpu)

    def run_all(
        self,
        jobs: Sequence[Any],
        run_one: Callable[[Any, str], int],
        *,
        fail_fast: bool = False,
        on_complete: Optional[Callable[[Any, int], None]] = None,
    ) -> int:
        failures = 0
        stop_submit = False

        def _wrapped(job: Any) -> Tuple[Any, int]:
            gpu_id = str(self.gpu_queue.get())
            try:
                return job, int(run_one(job, gpu_id))
            finally:
                self.gpu_queue.put(gpu_id)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            pending = {}
            next_i = 0

            def _submit() -> None:
                nonlocal next_i
                if stop_submit or next_i >= len(jobs):
                    return
                job = jobs[next_i]
                fut = ex.submit(_wrapped, job)
                pending[fut] = job
                next_i += 1

            for _ in range(min(self.max_workers, len(jobs))):
                _submit()

            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    job = pending.pop(fut)
                    _, rc = fut.result()
                    if on_complete is not None:
                        on_complete(job, rc)
                    if rc != 0:
                        failures += 1
                        if fail_fast:
                            stop_submit = True
                    _submit()

        return failures
