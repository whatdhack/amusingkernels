"""The agent's tools, one file per tool: its DEFINITION (what the model sees) and
run(ctx, args) (what happens when the model calls it).

    kernel_tools.definitions()              # the tools to offer the model
    result = kernel_tools.run(client, root, run_id, call)  # run the tool the model chose
    result.get("status", "running")         # "running", or why the run must stop

A result is what the model sees next. A refused call comes back as {"error": ...} so the
model can correct it. A result carries "status" only when the run must stop:
"paused" (eval timed out; resume to retry). Eval outcomes themselves are reported
under "eval_status" (PASSED, INCORRECT_NUMERICAL, ...) and never stop the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from kernel_agent.record import KernelRecord

from . import (
    build_solution,
    read_problem,
    read_trace,
    record_lesson,
    run_eval,
    write_kernel,
)

MODULES: dict[str, Any] = {
    str(m.DEFINITION["name"]): m
    for m in (
        read_problem,
        write_kernel,
        build_solution,
        run_eval,
        read_trace,
        record_lesson,
    )
}
DEFINITIONS: dict[str, dict] = {
    name: module.DEFINITION for name, module in MODULES.items()
}


@dataclass
class Context:
    """What a tool gets besides its arguments."""

    client: OpenAI
    root: Path
    run_id: str
    record: KernelRecord


def definitions() -> list:
    """The tool definitions to offer the model."""
    return [DEFINITIONS[name] for name in MODULES]


def run(client: OpenAI, root: Path, run_id: str, call: Any) -> dict:
    """Run the tool the model chose (an item of response.output, or the same as a dict)."""
    name, raw = _field(call, "name"), _field(call, "arguments") or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError) as error:
        return {"error": f"tool arguments are not valid JSON: {error}"}
    if not isinstance(args, dict):
        return {"error": "tool arguments must be a JSON object"}
    if name not in MODULES:
        return {"error": f"unknown tool: {name!r}"}
    ctx = Context(client, Path(root), run_id, KernelRecord(root, run_id))
    try:
        result = MODULES[name].run(ctx, args)
    except Exception as error:  # the model reads the error and tries again
        return {"error": str(error) or type(error).__name__}
    return result if isinstance(result, dict) else {"result": result}


def _field(call: Any, key: str) -> Any:
    return call.get(key) if isinstance(call, dict) else getattr(call, key, None)
