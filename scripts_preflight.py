"""Preflight for ./helper setup: key, problem, evaluator (no GPU work)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    failures = []
    if not os.environ.get("MODEL_API_KEY"):
        failures.append("MODEL_API_KEY is not set: add it to .env")
    problem = Path(os.environ.get("PROBLEM_DIR", "").strip())
    if not os.environ.get("PROBLEM_DIR", "").strip():
        failures.append("PROBLEM_DIR is not set: point it at a problem folder in .env")
    else:
        resolved = problem if problem.is_absolute() else (ROOT / problem)
        if not (resolved / "definition.json").is_file():
            failures.append(f"PROBLEM_DIR has no definition.json: {resolved}")
        if not (resolved / "workload.jsonl").is_file():
            failures.append(f"PROBLEM_DIR has no workload.jsonl: {resolved}")
    binary = os.environ.get("SOL_EXECBENCH", "").strip() or "sol-execbench"
    if shutil.which(binary) is None and not Path(binary).is_file():
        failures.append(f"evaluator not found: {binary} (install SOL-ExecBench first)")
    else:
        try:
            subprocess.run(
                [binary, "--help"], capture_output=True, timeout=60, check=True
            )
        except Exception as error:
            failures.append(f"evaluator --help failed: {error}")
    try:
        import openai  # noqa: F401
    except ImportError:
        failures.append("openai package missing: ./helper setup installs it")
    if failures:
        print("preflight failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("GEMINI_API_KEY"):
        critic_note = "critic " + (
            os.environ.get("GEMINI_API_MODEL") or "gemini-4-argon"
        )
    else:
        critic_note = "critic disabled"
    print(f"preflight ok: key, problem, evaluator, openai, {critic_note}")


if __name__ == "__main__":
    main()
