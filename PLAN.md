## Goal

Create a new repo, `amusingkernels`, holding a kernel-optimization agent that reuses the `loop.py` agentic-loop pattern from the legal-agent workshop, and produces SOL-ExecBench leaderboard submissions (a valid `solution.json` per problem, debugged locally first). No Docker anywhere: native Python environment on the host GPU machine.

## Success Criteria

- `amusingkernels/` repo exists with this plan, an agent loop, kernel tools, and docs.
- Local eval works natively: `sol-execbench <problem_dir> --solution solution.json` runs in the host venv on our GPU.
- One L1 problem goes end-to-end: reference baseline → agent-written kernel → correct trace → `solution.json`.
- Leaderboard submission path is documented (account, upload, SOL-Score reading).

## Submission types (researched answer)

- The site takes **leaderboard submissions only**: solutions run on NVIDIA B200 GPUs and are ranked by SOL-Score; an NVIDIA account is required to submit and track progress.
- There is **no separate debug-eval submission track** on the site. Debugging is **local**: install the evaluator natively, download the datasets, and run the `sol-execbench` CLI against a problem directory (`definition.json` + `workload.jsonl`) with your `solution.json`.
- Official runs lock B200 SM clocks at 1500 MHz and still show ~5% run-to-run variance (up to ~10% for many-small-kernel launches), so local scores are approximate unless our GPU and clocks match.

## Approach

Reuse only the domain-free skeleton from `legal_agent/loop.py` (model picks one tool per turn, result fed back, progress saved each turn, pause/resume). Replace everything domain-specific:

- Instructions: profile-driven kernel workflow (baseline → correctness first → optimize → re-time), Triton-first.
- Tools (new, kernel-specific): read problem (`definition.json`/`workload.jsonl`/reference), write kernel source, build a `solution.json`, run local `sol-execbench` eval, read back the trace (correctness + latency).
- Record: kernel versions with their traces and timings (replaces the legal case record).
- Verification: correctness must pass before any timing claim counts; reward-hacking checks noted by the evaluator must stay clean.
- Environment is native (no containers): the `rsi` conda env (`/home/sgoswami/miniforge3/envs/rsi/bin/python`) with the SOL-ExecBench checkout installed (`pip install -e . --no-deps`, keeping torch 2.13.0+cu130 instead of the pinned 2.9.0), plus small runtime deps (`einops datasets safetensors numcodecs ninja huggingface-hub[cli] cupti-python`), plus the HF datasets on disk.
- Start with one small L1 problem and the PyTorch reference as baseline; add `run_dataset.py` batch runs only after single-problem works.

## Steps

1. Scaffold `amusingkernels/`: README, agent loop (ported skeleton), `kernel_tools/`, `runs/` progress dir, submission notes doc.
2. Environment (native, no Docker) — DONE 2026-11-02: `rsi` conda env (Python 3.13.14, torch 2.13.0+cu130, triton 3.7.1, openai 2.44.0); SOL-ExecBench checkout as sibling `../SOL-ExecBench` (not forked), installed `--no-deps`; HF CLI installed; benchmark data downloaded (`data/benchmark`: L1 94, L2 82, Quant 33, FlashInfer-Bench 26 problems); driver 580.95.05 on NVIDIA GB10; `sol-execbench --help` works; shipped Triton example `examples/triton/rmsnorm` PASSES 14/14 natively.
3. Baseline: run the shipped example eval end-to-end natively and record a reference trace for the chosen L1 problem.
4. Build kernel tools: problem reader, kernel writer, `solution.json` builder (DPS in-place convention, `name`/`definition`/`author`/`spec`/`sources`), local-eval runner (subprocess `sol-execbench`), trace reader.
5. Wire the loop: instructions + tool definitions + per-turn save, last-turn "submit what you have", pause on eval-timeout/API failure with resume.
6. First campaign: agent optimizes the one L1 problem until correct and faster than reference; freeze the winning `solution.json`.
7. Submission packaging: per-problem solution files plus a checklist (account, upload, expected SOL-Score, variance caveat).

## Validation Plan

- Env: `sol-execbench examples/cute_dsl/jamba_attn_proj --solution examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json` returns a correct trace, run natively in the venv.
- Single problem: `sol-execbench <L1-problem-dir> --solution <ours>.json --json` prints correctness pass + latency; re-run twice to eyeball variance.
- Batch (later): `scripts/run_dataset.py ../SOL-ExecBench/data/benchmark --limit 5 --max-workloads 3 -o ./results` completes with traces (note: actual data layout is `data/benchmark`, not `data/SOL-ExecBench/benchmark` as the upstream README says).
- Submission readiness: `solution.json` validates against the solution schema (required `name`, `definition`, `author`, `spec`, `sources`; DPS in-place signature).

## Risks / Open Questions

- Our local GPU is a GB10 (driver 580.95.05), not a B200, so local timings won't match official SOL-Score; treat local results as directional (no B200 access confirmed yet).
- Without Docker, the native env deliberately diverges from the pinned stack: torch 2.13.0+cu130 (vs pinned 2.9.0), numpy 2.5.1 (vs 2.2.6), numcodecs 0.17.1 (vs 0.16.5), cuda-bindings 13.0.3 (vs 13.1.1); heavy deps (cupy, cutlass-dsl, cupynumeric, cudnn-frontend) not installed — Triton/PyTorch paths verified working, CUDA C++/cuDNN/CUTLASS paths not yet attempted. System gcc 13.3 + ptxas available; `ninja` pip package installed.
- Reference timing can dominate (sometimes >1h); keep `benchmark_reference` off except when a speedup number is needed.
- Reward-hacking checks are evaluator-side and not fully documented in what we read; keep kernels honest (real computation, no shape specialization beyond the workload spec).
- Starter L1 problem: `085_geglu_activation` (tiny elementwise reference, 16 random-input workloads). PyTorch-port baseline recorded 2026-11-02: 16/16 PASSED, port 1.61ms mean vs reference 1.18ms (speedup 0.73). Triton-first assumed, CUDA C++ as fallback.
- `flashinfer-trace` dataset NOT downloaded (only needed for the FlashInfer-Bench subset; L1/L2/Quant workloads use random/custom/scalar inputs).

## Sources

- https://raw.githubusercontent.com/NVIDIA/SOL-ExecBench/main/README.md
- https://raw.githubusercontent.com/NVIDIA/SOL-ExecBench/main/docs/solution.md
- https://raw.githubusercontent.com/NVIDIA/SOL-ExecBench/main/pyproject.toml
- https://research.nvidia.com/benchmarks.md
