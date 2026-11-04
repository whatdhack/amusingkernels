"""read_trace: the full eval trace back (correctness + latency + log)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import Context

LOG_LIMIT = 4000

DEFINITION = {
    "type": "function",
    "name": "read_trace",
    "description": "Read back the full eval trace for a version: per-workload status, "
    "correctness errors, latency, and the evaluator log tail.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {"type": "string", "description": "Kernel version id, e.g. v1."},
        },
        "required": ["version"],
    },
}


def run(ctx: Context, args: dict) -> dict:
    version_id = str(args.get("version", ""))
    try:
        version = ctx.record.get_version(version_id)
    except KeyError:
        return {"error": f"no such kernel version: {version_id}"}
    evals = version.get("evals") or []
    if not evals:
        return {"error": f"{version_id} has no eval yet: run_eval first"}
    last = evals[-1]
    trace_file = ctx.root / last["trace"] if last.get("trace") else None
    if trace_file is None or not trace_file.is_file():
        return {"error": f"{version_id} has no saved trace file"}
    traces = json.loads(trace_file.read_text())
    traces = traces if isinstance(traces, list) else [traces]
    from kernel_agent.sol import trace_sols

    sols = trace_sols(ctx.root, ctx.run_id, traces)
    workloads = []
    for trace, sol in zip(traces, sols):
        evaluation = trace.get("evaluation", {}) if isinstance(trace, dict) else {}
        workloads.append(
            {
                "axes": (trace.get("workload", {}) or {}).get("axes", {}),
                "status": evaluation.get("status"),
                "sol_ms": sol["sol_ms"] if sol else None,
                "sol_bottleneck": sol["bottleneck"] if sol else None,
                "correctness": evaluation.get("correctness"),
                "performance": evaluation.get("performance"),
                "log_tail": str(evaluation.get("log", ""))[-LOG_LIMIT:],
            }
        )
    return {
        "version": version_id,
        "eval_status": last.get("eval_status"),
        "passed": last.get("passed"),
        "n_workloads": last.get("n_workloads"),
        "workloads": workloads,
    }
