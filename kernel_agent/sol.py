"""GB10 roofline SOL for a problem workload.

SOL(time) = max(compute, memory): fused bytes over peak bandwidth, and MACs
over peak MAC throughput. Bytes come straight from the definition shapes;
MACs come from a SOLAR stages 1-3 precompute, cached per run
(`ensure_macs`). When MACs are unavailable the SOL is memory-only and marked
as a lower bound.
"""

from __future__ import annotations

import ast
import json
import operator
import subprocess
from pathlib import Path
from typing import Any

# Peak rates per architecture. GB10: DRAM bandwidth is the public DGX Spark
# figure (128GB LPDDR5X unified, 273 GB/s); compute peaks are ESTIMATED from
# the public 1000 FP4 TOPS (sparse) figure (dense = half sparse, halving per
# wider precision, FP32 from ~18.4 TFLOPS CUDA-core estimate). B200: SOLAR's
# B200.yaml roofline assumptions (8TB/s-class HBM, 1.5 GHz nominal) — the same
# peaks the leaderboard's SOL-Score methodology uses. Predicted SOL times scale
# linearly with these; the memory side is the solid one.
PEAKS: dict[str, dict[str, Any]] = {
    "GB10": {
        "bw": 273e9,
        "flops": {
            "float64": 9.2e12,
            "float32": 18.4e12,
            "tf32": 62.5e12,
            "float16": 125e12,
            "bfloat16": 125e12,
            "float8_e4m3fn": 250e12,
            "float8_e5m2": 250e12,
            "int8": 250e12,
            "float4_e2m1": 500e12,
            "float4_e2m1fn_x2": 500e12,
        },
    },
    "B200": {
        "bw": 5111.6 * 1.5e9,
        "flops": {
            "float64": 60.0e12,
            "float32": 120.753e12,
            "tf32": 905.625e12,
            "float16": 1811.253e12,
            "bfloat16": 1811.253e12,
            "float8_e4m3fn": 3622.503e12,
            "float8_e5m2": 3622.503e12,
            "int8": 3622.5e12,
            "float4_e2m1": 7245e12,
            "float4_e2m1fn_x2": 7245e12,
        },
    },
}
PEAK_BW_BYTES_S = PEAKS["GB10"]["bw"]
PEAK_FLOPS = PEAKS["GB10"]["flops"]
DEFAULT_FLOPS = PEAK_FLOPS["float32"]

DTYPE_BYTES: dict[str, float] = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float4_e2m1": 0.5,
    "float4_e2m1fn_x2": 0.5,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "bool": 1,
}

TORCH_DTYPES: dict[str, str] = {
    "float64": "torch.float64",
    "float32": "torch.float32",
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "float8_e4m3fn": "torch.float8_e4m3fn",
    "float8_e5m2": "torch.float8_e5m2",
    "int64": "torch.int64",
    "int32": "torch.int32",
    "int16": "torch.int16",
    "int8": "torch.int8",
    "bool": "torch.bool",
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.floordiv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def resolve_dims(definition: dict, workload_axes: dict) -> dict[str, int]:
    """Concrete axis values: workload vars + definition consts + evaluated exprs."""
    dims: dict[str, int] = {}
    for name, axis in definition.get("axes", {}).items():
        if axis.get("type") == "const":
            dims[name] = int(axis["value"])
    dims.update({k: int(v) for k, v in workload_axes.items()})
    for name, axis in definition.get("axes", {}).items():
        if axis.get("type") == "expr":
            dims[name] = _eval_expr(axis["expression"], dims)
    return dims


def tensor_numel(shape: list | None, dims: dict[str, int]) -> int:
    """Element count for a definition shape (None = scalar)."""
    if not shape:
        return 1
    numel = 1
    for axis in shape:
        numel *= dims[axis]
    return numel


def fused_bytes(definition: dict, workload_axes: dict) -> int:
    """Fused traffic: read every input once, write every output once."""
    dims = resolve_dims(definition, workload_axes)
    total = 0.0
    for group in ("inputs", "outputs"):
        for tensor in definition.get(group, {}).values():
            total += (
                tensor_numel(tensor.get("shape"), dims)
                * DTYPE_BYTES[tensor["dtype"]]
            )
    return int(total)


def compute_peak_flops(definition: dict, arch: str = "GB10") -> float:
    """Peak rate for the problem's compute dtype (first input's dtype)."""
    flops = PEAKS.get(arch, PEAKS["GB10"])["flops"]
    inputs = definition.get("inputs", {})
    first = next(iter(inputs.values()), {})
    return flops.get(first.get("dtype", ""), flops["float32"])


def sol_ms(
    definition: dict, workload: dict, macs: float | None, arch: str = "GB10"
) -> dict[str, Any]:
    """Roofline SOL for one workload. macs=None gives a memory-only lower bound."""
    peaks = PEAKS.get(arch, PEAKS["GB10"])
    nbytes = fused_bytes(definition, workload.get("axes", {}))
    mem_ms = nbytes / peaks["bw"] * 1000
    compute_ms = (
        macs * 2 / compute_peak_flops(definition, arch) * 1000
        if macs is not None
        else None
    )
    sol = max([m for m in (mem_ms, compute_ms) if m is not None])
    return {
        "sol_ms": sol,
        "mem_ms": mem_ms,
        "compute_ms": compute_ms,
        "bottleneck": "compute"
        if compute_ms is not None and compute_ms > mem_ms
        else "memory",
        "macs": macs,
        "bytes": nbytes,
    }


def trace_sols(root: Path, run_id: str, traces: list) -> list:
    """GB10 roofline SOL per trace (None when the workload is unknown)."""
    from kernel_agent.settings import problem_dir

    prob = problem_dir(Path(root))
    definition = json.loads((prob / "definition.json").read_text())
    workloads = {}
    for line in (prob / "workload.jsonl").read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            workloads[entry.get("uuid")] = entry
    macs = ensure_macs(root, run_id, prob)
    sols = []
    for trace in traces:
        uuid = (
            (trace.get("workload") or {}).get("uuid")
            if isinstance(trace, dict)
            else None
        )
        workload = workloads.get(uuid)
        if not workload:
            sols.append(None)
            continue
        hardware = ""
        if isinstance(trace, dict):
            hardware = str((trace.get("evaluation") or {}).get("environment", {}).get("hardware", ""))
        arch = "B200" if "B200" in hardware else "GB10"
        sols.append(sol_ms(definition, workload, macs.get(uuid), arch))
    return sols


def ensure_macs(root: Path, run_id: str, problem: Path) -> dict[str, float | None]:
    """MACs per workload uuid via a cached SOLAR stages 1-3 precompute.

    Runs once per run (results cached at runs/<run_id>/solar_macs.json).
    Workloads SOLAR cannot analyze get None (memory-only SOL fallback).
    """
    root, problem = Path(root), Path(problem)
    cache = root / "runs" / run_id / "solar_macs.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    from kernel_agent.settings import solar_dir

    macs: dict[str, float | None] = {}
    definition = json.loads((problem / "definition.json").read_text())
    workloads = [
        json.loads(line)
        for line in (problem / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    checkout = solar_dir(root)
    for index, workload in enumerate(workloads):
        uuid = workload.get("uuid", str(index))
        macs[uuid] = None
        if checkout is None:
            continue
        try:
            macs[uuid] = _solar_macs(root, run_id, checkout, definition, workload, index)
        except Exception:
            macs[uuid] = None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(macs, indent=1))
    return macs


def _eval_expr(expression: str, dims: dict[str, int]) -> int:
    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return int(node.value)
        if isinstance(node, ast.Name):
            return dims[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError(f"unsupported expression: {expression!r}")

    return visit(ast.parse(expression, mode="eval"))


MODEL_TEMPLATE = '''"""SOLAR model synthesized from a SOL-ExecBench reference."""
import torch
import torch.nn as nn

REFERENCE = {reference!r}

_namespace: dict = {{}}
exec(REFERENCE, _namespace)
_run = _namespace["run"]


class Model(nn.Module):
    def forward(self, *args):
        return _run(*args)


def get_inputs():
    torch.manual_seed(0)
    return [{inputs}]
'''


def _solar_macs(
    root: Path,
    run_id: str,
    checkout: Path,
    definition: dict,
    workload: dict,
    index: int,
) -> float:
    """Run SOLAR stages 1-3 for one workload shape; return total MACs."""
    import os

    work = root / "runs" / run_id / "solar" / f"w{index:02d}"
    work.mkdir(parents=True, exist_ok=True)
    dims = resolve_dims(definition, workload.get("axes", {}))
    inputs = ", ".join(
        _synth_input(name, spec, dims, workload) for name, spec in definition["inputs"].items()
    )
    (work / "model.py").write_text(
        MODEL_TEMPLATE.format(reference=definition["reference"], inputs=inputs)
    )
    env = dict(os.environ, PYTHONPATH=str(checkout))
    py = _python()
    _stage(
        [py, "-m", "solar.cli.process_model", "--model-file", str(work / "model.py"),
         "--output-dir", str(work / "graph"), "--force-rerun"], env)
    _stage(
        [py, "-m", "solar.cli.toeinsum_model", "--graph-path",
         str(work / "graph" / "pytorch_graph.yaml"),
         "--output-dir", str(work / "einsum"), "--no-copy-graph"], env)
    _stage(
        [py, "-m", "solar.cli.analyze_model", "--einsum-graph-path",
         str(work / "einsum" / "einsum_graph_renamed.yaml"),
         "--output-dir", str(work / "analysis")], env)
    import yaml

    analysis = yaml.safe_load((work / "analysis" / "analysis.yaml").read_text())
    return float((analysis.get("total") or {}).get("macs", 0))


def _synth_input(name: str, spec: dict, dims: dict, workload: dict) -> str:
    shape = spec.get("shape")
    if not shape:
        entry = (workload.get("inputs") or {}).get(name, {})
        return repr(entry.get("value", 1))
    torch_dtype = TORCH_DTYPES[spec["dtype"]]
    concrete = [dims[axis] for axis in shape]
    return f"torch.randn({concrete}, dtype={torch_dtype})"


def _stage(cmd: list[str], env: dict) -> None:
    subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600, check=True)


def _python() -> str:
    import os
    import sys

    return os.environ.get("PYTHON", "").strip() or sys.executable
