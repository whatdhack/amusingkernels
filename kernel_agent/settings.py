"""Run settings read from the environment (./helper loads .env)."""

from __future__ import annotations

import os
from pathlib import Path


def model() -> str:
    """The model the agent talks to (MODEL_API_MODEL, default muse-spark-1.3-contributor)."""
    return os.environ.get("MODEL_API_MODEL") or "muse-spark-1.3-contributor"


def author() -> str:
    """The solution author tag (AUTHOR, default amusingkernels)."""
    return os.environ.get("AUTHOR", "").strip() or "amusingkernels"


def problem_dir(root: Path) -> Path:
    """The SOL-ExecBench problem folder (PROBLEM_DIR, absolute or relative to the repo root).

    Must contain definition.json and workload.jsonl.
    """
    value = os.environ.get("PROBLEM_DIR", "").strip()
    if not value:
        raise RuntimeError("PROBLEM_DIR is not set: point it at a problem folder in .env")
    path = Path(value).expanduser()
    resolved = path if path.is_absolute() else (Path(root) / path).resolve()
    if not (resolved / "definition.json").is_file():
        raise RuntimeError(f"PROBLEM_DIR has no definition.json: {resolved}")
    if not (resolved / "workload.jsonl").is_file():
        raise RuntimeError(f"PROBLEM_DIR has no workload.jsonl: {resolved}")
    return resolved


def sol_execbench() -> str:
    """The evaluator binary (SOL_EXECBENCH, default sol-execbench on PATH)."""
    return os.environ.get("SOL_EXECBENCH", "").strip() or "sol-execbench"


def eval_backend() -> str:
    """Where run_eval executes (EVAL_BACKEND, default local).

    "local" runs sol-execbench on this machine; "modal" runs it on a Modal
    GPU (MODAL_GPU, default B200) via modal_eval.py, billing that workspace.
    """
    value = os.environ.get("EVAL_BACKEND", "").strip().lower() or "local"
    if value not in ("local", "modal"):
        raise RuntimeError(f"EVAL_BACKEND must be local or modal: {value!r}")
    return value


def solar_dir(root: Path) -> Path | None:
    """The SOLAR checkout for MAC precompute (SOLAR_DIR, default ../SOLAR).

    Returns None when there is no usable checkout; SOL falls back to
    memory-only.
    """
    value = os.environ.get("SOLAR_DIR", "").strip()
    if value:
        path = Path(value).expanduser()
        candidate = path if path.is_absolute() else (Path(root) / path).resolve()
    else:
        candidate = (Path(root) / "../SOLAR").resolve()
    if (candidate / "solar" / "perf").is_dir():
        return candidate
    return None
