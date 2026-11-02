"""read_problem: the SOL-ExecBench problem (signature, shapes, reference, workloads)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from kernel_agent.settings import problem_dir

if TYPE_CHECKING:
    from . import Context

REFERENCE_LIMIT = 8000

DEFINITION = {
    "type": "function",
    "name": "read_problem",
    "description": "Read the SOL-ExecBench problem: definition name, input/output signature "
    "with shapes and dtypes, the PyTorch reference implementation, and workload count.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    },
}


def run(ctx: Context, args: dict) -> dict:
    prob = problem_dir(ctx.root)
    definition = json.loads((prob / "definition.json").read_text())
    workloads = (prob / "workload.jsonl").read_text().splitlines()
    first = json.loads(workloads[0]) if workloads else {}
    inputs = list(definition["inputs"])
    outputs = list(definition["outputs"])
    reference = definition.get("reference", "")
    truncated = len(reference) > REFERENCE_LIMIT
    return {
        "definition": definition["name"],
        "op_type": definition.get("op_type"),
        "description": definition.get("description"),
        "inputs": definition["inputs"],
        "outputs": definition["outputs"],
        "axes": definition.get("axes"),
        "constraints": definition.get("constraints", []),
        "custom_inputs_entrypoint": definition.get("custom_inputs_entrypoint"),
        "dps_signature": f"run({', '.join(inputs + outputs)})",
        "n_workloads": len(workloads),
        "first_workload_axes": first.get("axes", {}),
        "reference": reference[:REFERENCE_LIMIT],
        "reference_truncated": truncated,
    }
