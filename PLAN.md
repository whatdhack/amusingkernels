## Goal

Create a new repo, `amusingkernels`, holding a kernel-optimization agent that reuses the `loop.py` agentic-loop pattern from the legal-agent workshop, and produces SOL-ExecBench leaderboard submissions (a valid `solution.json` per problem, debugged locally first). No Docker anywhere: native Python environment on the host GPU machine.

Extended goal (Nov 2026): two models work the loop together (Muse drives, Gemini reviews failures and proposes fresh approaches); stalls pull KernelWiki techniques; a GPU-less B200 latency model predicts Triton GEMMs and is calibrated against real Modal B200 traces; campaigns cover FlashInfer-Bench GEMMs as well as L1.

## Success Criteria

- `amusingkernels/` repo exists with this plan, an agent loop, kernel tools, and docs.
- Local eval works natively: `sol-execbench <problem_dir> --solution solution.json` runs in the host venv on our GPU.
- One L1 problem goes end-to-end: reference baseline → agent-written kernel → correct trace → `solution.json`.
- Leaderboard submission path is documented (account, upload, SOL-Score reading).
- Dual-model loop: failed evals get a critic review, stalls get a wiki technique plus a fresh-eyes proposal; versions record their authoring model.
- B200 prediction: Triton GEMM evals carry an advisory latency prediction and log predicted-vs-measured pairs; `./helper calibrate` compares them.
- A FlashInfer-Bench GEMM collection (`amusingkernels-fib-001`) is built from frozen Modal-verified solutions.

## Submission types (researched answer)

- The site takes **leaderboard submissions only**: solutions run on NVIDIA B200 GPUs and are ranked by SOL-Score; an NVIDIA account is required to submit and track progress.
- There is **no separate debug-eval submission track** on the site. Debugging is **local**: install the evaluator natively, download the datasets, and run the `sol-execbench` CLI against a problem directory (`definition.json` + `workload.jsonl`) with your `solution.json`.
- Official runs lock B200 SM clocks at 1500 MHz and still show ~5% run-to-run variance (up to ~10% for many-small-kernel launches), so local scores are approximate unless our GPU and clocks match.

## Approach

Reuse only the domain-free skeleton from `legal_agent/loop.py` (model picks one tool per turn, result fed back, progress saved each turn, pause/resume). Replace everything domain-specific:

- Instructions: profile-driven kernel workflow (baseline → correctness first → optimize → re-time), Triton-first.
- Tools (new, kernel-specific): read problem (`definition.json`/`workload.jsonl`/reference), write kernel source, build a `solution.json`, run `sol-execbench` eval (local subprocess or Modal B200), read back the trace (correctness + latency), record durable lessons.
- Second model: the driver loop is single-conversation; Gemini joins via stateless calls — a review after each failed eval, a fresh-eyes proposal when stalled. Neither call can stop the run.
- Stagnation: after N evals without a new best (default 3), the loop appends one guidance message — a KernelWiki technique chosen by measured symptoms (roofline bottleneck, else problem kind) plus the fresh-eyes proposal — rotating past tried pages.
- B200 prediction (advisory, Triton GEMMs only): tile config from the launch site plus per-workload shapes feed a timed Petri-net model; the prediction never blocks an eval, and every attempt logs a calibration pair for later comparison and fitting.
- Record: kernel versions with their traces, timings, authoring model, and predictions (replaces the legal case record).
- Verification: correctness must pass before any timing claim counts; reward-hacking checks noted by the evaluator must stay clean.
- Environment is native (no containers): the `rsi` conda env (`/home/sgoswami/miniforge3/envs/rsi/bin/python`) with the SOL-ExecBench checkout installed (`pip install -e . --no-deps`, keeping torch 2.13.0+cu130 instead of the pinned 2.9.0), plus small runtime deps (`einops datasets safetensors numcodecs ninja huggingface-hub[cli] cupti-python`), plus the HF datasets on disk.
- Start with one small L1 problem and the PyTorch reference as baseline; add `run_dataset.py` batch runs only after single-problem works.

## Steps

1. Scaffold `amusingkernels/`: README, agent loop (ported skeleton), `kernel_tools/`, `runs/` progress dir, submission notes doc.
2. Environment (native, no Docker) — DONE 2026-11-02: `rsi` conda env (Python 3.13.14, torch 2.13.0+cu130, triton 3.7.1, openai 2.44.0); SOL-ExecBench checkout as sibling `../SOL-ExecBench` (not forked), installed `--no-deps`; HF CLI installed; benchmark data downloaded (`data/benchmark`: L1 94, L2 82, Quant 33, FlashInfer-Bench 26 problems); driver 580.95.05 on NVIDIA GB10; `sol-execbench --help` works; shipped Triton example `examples/triton/rmsnorm` PASSES 14/14 natively.
3. Baseline: run the shipped example eval end-to-end natively and record a reference trace for the chosen L1 problem.
4. Build kernel tools: problem reader, kernel writer, `solution.json` builder (DPS in-place convention, `name`/`definition`/`author`/`spec`/`sources`), local-eval runner (subprocess `sol-execbench`), trace reader.
5. Wire the loop: instructions + tool definitions + per-turn save, last-turn "submit what you have", pause on eval-timeout/API failure with resume.
6. First campaign — DONE 2026-11-02: 30-turn run on `085_geglu_activation` (model `muse-spark-1.3-contributor`); winner v3 Triton PASSED 16/16 at 0.711ms mean (~2.3x vs the 1.61ms PyTorch-port baseline); frozen at `solutions/085_geglu_activation/` and independently re-verified 16/16 PASSED.
7. Submission packaging: per-problem solution files plus a checklist (account, upload, expected SOL-Score, variance caveat).
8. Dual-model loop — DONE 2026-11-08: Gemini critic client, version authorship, critic-on-failure reviews; proven on a live 10-turn 005 Modal campaign.
9. Stagnation + KernelWiki — DONE 2026-11-08: stagnation counter in progress notes, trace-based symptoms, technique rotation; same live campaign.
10. B200 prediction + calibration — DONE 2026-11-08: evaluator port, SOL adapter, advisory predictions with pair logging, `./helper calibrate` report; first pair 123µs predicted vs 50.5µs measured.
11. FlashInfer-Bench GEMM campaigns — IN PROGRESS: manual 005 winner + 20-turn 004 agent campaign frozen into `collections/amusingkernels-fib-001/`; remaining GEMMs (006–011) follow the same pattern.

## Validation Plan

- Env: `sol-execbench examples/cute_dsl/jamba_attn_proj --solution examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json` returns a correct trace, run natively in the venv.
- Single problem: `sol-execbench <L1-problem-dir> --solution <ours>.json --json` prints correctness pass + latency; re-run twice to eyeball variance.
- Batch (later): `scripts/run_dataset.py ../SOL-ExecBench/data/benchmark --limit 5 --max-workloads 3 -o ./results` completes with traces (note: actual data layout is `data/benchmark`, not `data/SOL-ExecBench/benchmark` as the upstream README says).
- Submission readiness: `solution.json` validates against the solution schema (required `name`, `definition`, `author`, `spec`, `sources`; DPS in-place signature).

## Risks / Open Questions

- Our local GPU is a GB10 (driver 580.95.05), not a B200, so local timings won't match official SOL-Score; treat local results as directional. B200 access confirmed via Modal (`EVAL_BACKEND=modal`, `whatdhack` workspace); this box runs Modal-only (the `rsi` env no longer exists here). B200 microsecond kernels flake CUPTI timing (RUNTIME_ERROR "kernel activity sequence not found") — re-run, don't rewrite.
- Without Docker, the native env deliberately diverges from the pinned stack: torch 2.13.0+cu130 (vs pinned 2.9.0), numpy 2.5.1 (vs 2.2.6), numcodecs 0.17.1 (vs 0.16.5), cuda-bindings 13.0.3 (vs 13.1.1); heavy deps (cupy, cutlass-dsl, cupynumeric, cudnn-frontend) not installed — Triton/PyTorch paths verified working, CUDA C++/cuDNN/CUTLASS paths not yet attempted. System gcc 13.3 + ptxas available; `ninja` pip package installed.
- Reference timing can dominate (sometimes >1h); keep `benchmark_reference` off except when a speedup number is needed.
- Reward-hacking checks are evaluator-side and not fully documented in what we read; keep kernels honest (real computation, no shape specialization beyond the workload spec).
- Starter L1 problem: `085_geglu_activation` (tiny elementwise reference, 16 random-input workloads). PyTorch-port baseline recorded 2026-11-02: 16/16 PASSED, port 1.61ms mean vs reference 1.18ms (speedup 0.73). Triton-first assumed, CUDA C++ as fallback.
- `flashinfer-trace` dataset only partially downloaded (GEMM/norm workloads use random inputs; only attention/MoE need trace safetensors). FlashInfer-Bench definitions ship with `"hf_id": ""`, which the evaluator rejects — null them after download (README step 3b).

## Sources

- https://raw.githubusercontent.com/NVIDIA/SOL-ExecBench/main/README.md
- https://raw.githubusercontent.com/NVIDIA/SOL-ExecBench/main/docs/solution.md
- https://raw.githubusercontent.com/NVIDIA/SOL-ExecBench/main/pyproject.toml
- https://research.nvidia.com/benchmarks.md
