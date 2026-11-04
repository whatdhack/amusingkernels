"""Run sol-execbench evals on Modal B200s.

Validating GPU access (fast, tiny image):

    modal run modal_eval.py::check

Evaluating a frozen solution on a B200 (builds the CUDA+torch image once,
then cached):

    modal run modal_eval.py --problem ../SOL-ExecBench/data/benchmark/L1/085_geglu_activation \
        --solution solutions/085_geglu_activation/solution.json

Works best for random/scalar-input workloads (no dataset mount needed):
problem + solution files are passed as text.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent
SOL_SRC = REPO.parent / "SOL-ExecBench" / "src"

import os

GPU = os.environ.get("MODAL_GPU", "B200")

if modal.is_local() and not (SOL_SRC / "sol_execbench").is_dir():
    raise RuntimeError(
        f"SOL-ExecBench checkout not found at {SOL_SRC.parent} "
        "(expected as a sibling of this repo)"
    )

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04", add_python="3.12"
    )
    .pip_install("torch==2.9.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install(
        [
            "einops",
            "datasets",
            "safetensors",
            "numcodecs",
            "ninja",
            "huggingface-hub",
            "cupti-python>=13.0.1",
            "click",
            "rich",
            "numpy",
            "pydantic",
            "cuda-bindings==13.1.1",
            "cuda-python==13.1.1",
        ]
    )
    .add_local_dir(SOL_SRC, remote_path="/opt/sol_execbench")
)

app = modal.App("amusingkernels-eval")


@app.function(gpu=GPU, image=modal.Image.debian_slim(), timeout=300)
def gpu_check() -> str:
    import subprocess

    return subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


@app.function(gpu=GPU, image=image, timeout=1800)
def eval_kernel(
    definition: str,
    workload: str,
    solution: str,
    timeout: int = 600,
    compile_timeout: int = 120,
) -> dict:
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/opt/sol_execbench")
    prob = "/tmp/prob"
    os.makedirs(prob, exist_ok=True)
    with open(f"{prob}/definition.json", "w") as f:
        f.write(definition)
    with open(f"{prob}/workload.jsonl", "w") as f:
        f.write(workload)
    with open("/tmp/solution.json", "w") as f:
        f.write(solution)
    cmd = [
        sys.executable,
        "-c",
        "from sol_execbench.cli import cli; cli()",
        prob,
        "--solution",
        "/tmp/solution.json",
        "--json",
        "--timeout",
        str(timeout),
        "--compile-timeout",
        str(compile_timeout),
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + compile_timeout + 120,
        )
    except subprocess.TimeoutExpired as error:
        return {"timeout": True, "error": str(error)}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-30000:],
        "stderr": completed.stderr[-8000:],
    }


@app.local_entrypoint()
def check() -> None:
    print(f"{GPU}:", gpu_check.remote())


@app.local_entrypoint()
def main(problem: str, solution: str, timeout: int = 600) -> None:
    import json

    prob = Path(problem).expanduser()
    definition = (prob / "definition.json").read_text()
    workload = (prob / "workload.jsonl").read_text()
    solution_text = Path(solution).expanduser().read_text()
    result = eval_kernel.remote(definition, workload, solution_text, timeout)
    if result.get("timeout"):
        raise SystemExit(f"remote eval timed out: {result}")
    if result["returncode"] != 0:
        print(result["stderr"] or result["stdout"])
        raise SystemExit(f"remote eval failed (exit {result['returncode']})")
    traces = [
        json.loads(line)
        for line in result["stdout"].splitlines()
        if line.strip().startswith("{")
    ]
    statuses = [t.get("evaluation", {}).get("status") for t in traces]
    latencies = [
        t["evaluation"]["performance"]["latency_ms"]
        for t in traces
        if t.get("evaluation", {}).get("status") == "PASSED"
    ]
    print(f"workloads: {len(traces)}, passed: {statuses.count('PASSED')}")
    if latencies:
        print(f"mean latency: {sum(latencies) / len(latencies):.4f}ms (B200)")
    for status in sorted(set(statuses)):
        if status != "PASSED":
            print(f"{status}: {statuses.count(status)}")
