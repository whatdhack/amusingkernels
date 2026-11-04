"""run_eval: run the local sol-execbench eval for a kernel version."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

from kernel_agent.settings import problem_dir, sol_execbench

if TYPE_CHECKING:
    from . import Context

LOG_LIMIT = 2000

DEFINITION = {
    "type": "function",
    "name": "run_eval",
    "description": "Run local sol-execbench eval for a built kernel version and record the trace "
    "(correctness + latency). Needs build_solution first. A timed-out eval pauses the run.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {"type": "string", "description": "Kernel version id, e.g. v1."},
            "timeout": {
                "type": "integer",
                "description": "Evaluation timeout in seconds (default 600).",
            },
            "compile_timeout": {
                "type": "integer",
                "description": "Compilation timeout in seconds (default 120).",
            },
            "benchmark_reference": {
                "type": "boolean",
                "description": "Also time the reference for a speedup number (default false; "
                "can be very slow).",
            },
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
    if not version.get("solution_path"):
        return {"error": f"{version_id} has no solution.json yet: build_solution first"}
    solution = ctx.root / version["solution_path"]
    folder = solution.parent
    trace_path = folder / "trace.json"
    jsonl_path = folder / "trace.jsonl"
    cmd = [
        sol_execbench(),
        str(problem_dir(ctx.root)),
        "--solution",
        str(solution),
        "--json",
        "-o",
        str(jsonl_path),
    ]
    cmd += ["--timeout", str(int(args.get("timeout") or 600))]
    cmd += ["--compile-timeout", str(int(args.get("compile_timeout") or 120))]
    if args.get("benchmark_reference"):
        config = folder / "bench_config.json"
        config.write_text(json.dumps({"benchmark_reference": True}))
        cmd += ["--config", str(config)]
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_timeout(args), cwd=ctx.root
        )
    except FileNotFoundError:
        return {"error": f"evaluator not found: {cmd[0]} (install SOL-ExecBench first)"}
    except subprocess.TimeoutExpired:
        return {
            "error": f"eval timed out after {_timeout(args)}s",
            "status": "paused",
        }
    if completed.returncode != 0:
        return {
            "error": "eval failed: "
            + (completed.stderr.strip() or completed.stdout.strip())[-LOG_LIMIT:],
            "version": version_id,
        }
    traces: list = []
    if jsonl_path.is_file():
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    traces.append(json.loads(line))
                except ValueError:
                    pass
    if not traces:
        for line in completed.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    traces.append(json.loads(line))
                except ValueError:
                    pass
    if not traces:
        return {
            "error": "eval produced no JSON trace: "
            + (completed.stdout + completed.stderr)[-LOG_LIMIT:],
            "version": version_id,
        }
    trace_path.write_text(json.dumps(traces, indent=1) + "\n")
    from kernel_agent.sol import trace_sols

    summary = _summarize(traces, trace_sols(ctx.root, ctx.run_id, traces))
    summary.update(
        version=version_id,
        trace=str(trace_path.relative_to(ctx.root)),
    )
    ctx.record.record_eval(version_id, summary)
    return summary


def _timeout(args: dict) -> int:
    return int(args.get("timeout") or 600) + int(args.get("compile_timeout") or 120) + 60


def _summarize(traces: list, sols: list | None = None) -> dict[str, Any]:
    """One eval verdict across the workload traces."""
    statuses = [_trace(t).get("status") for t in traces]
    passed = sum(1 for s in statuses if s == "PASSED")
    latencies = [
        p.get("latency_ms")
        for t in traces
        if isinstance((p := (_trace(t).get("performance") or {})), dict)
        and isinstance(p.get("latency_ms"), (int, float))
    ]
    ref_latencies = [
        p.get("reference_latency_ms")
        for t in traces
        if isinstance((p := (_trace(t).get("performance") or {})), dict)
        and isinstance(p.get("reference_latency_ms"), (int, float))
        and p.get("reference_latency_ms") > 0  # 0.0 means "not measured"
    ]
    errors = [
        c.get("max_relative_error")
        for t in traces
        if isinstance((c := (_trace(t).get("correctness") or {})), dict)
        and isinstance(c.get("max_relative_error"), (int, float))
    ]
    eval_status = statuses[0] if statuses and all(s == statuses[0] for s in statuses) else (
        "MIXED" if passed else (statuses[0] if statuses else "NO_TRACES")
    )
    if eval_status == "MIXED" and passed == len(traces):
        eval_status = "PASSED"
    latency = sum(latencies) / len(latencies) if latencies else None
    reference = sum(ref_latencies) / len(ref_latencies) if ref_latencies else None
    sol_values = [
        s["sol_ms"]
        for t, s in zip(traces, sols or [])
        if s is not None and _trace(t).get("status") == "PASSED"
    ]
    sol = sum(sol_values) / len(sol_values) if sol_values else None
    passed_lat = [
        p.get("latency_ms")
        for t in traces
        if _trace(t).get("status") == "PASSED"
        and isinstance((p := (_trace(t).get("performance") or {})), dict)
        and isinstance(p.get("latency_ms"), (int, float))
    ]
    pct_sol = (
        100 * sum(sol_values) / sum(passed_lat)
        if sol_values and passed_lat and sum(passed_lat) > 0
        else None
    )
    return {
        "eval_status": eval_status,
        "passed": passed,
        "n_workloads": len(traces),
        "statuses": statuses,
        "latency_ms": latency,
        "sol_ms": sol,
        "pct_sol": pct_sol,
        "reference_latency_ms": reference,
        "speedup_factor": (reference / latency)
        if latency and reference
        else None,
        "max_relative_error": max(errors) if errors else None,
    }


def _trace(trace: Any) -> dict:
    if isinstance(trace, dict):
        evaluation = trace.get("evaluation")
        return evaluation if isinstance(evaluation, dict) else {}
    return {}
