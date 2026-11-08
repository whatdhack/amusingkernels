# amusingkernels

**Goal: generate GPU kernels and make them faster with every run.**

A kernel-optimization agent for [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench): it writes GPU kernels (Triton-first), debugs them against the local or Modal B200 evaluator, and produces leaderboard-ready `solution.json` files. Muse drives the loop (model picks one tool per turn, progress saved each turn, pause/resume); Gemini reviews failures and proposes fresh approaches; a GPU-less B200 model predicts Triton GEMM latency for calibration.

![amusingkernels loop: an agent writes kernel versions for a SOL-ExecBench problem, each version is signature-checked and evaled for correctness then time vs roofline SOL, a failed eval gets a Gemini second opinion, a stall pulls a KernelWiki technique plus a fresh proposal, Triton GEMMs get an advisory B200 latency prediction whose pairs calibrate the model, the fastest full pass is frozen as solution.json, and durable findings go to learnings.md for the next run](assets/demo.gif)

Native Python on the host GPU — no Docker anywhere.

## Layout

- `PLAN.md` — goal, approach, steps, validation, risks.
- `SUBMISSION.md` — leaderboard submission path (account, upload, SOL-Score reading).
- `kernel_agent/` — the loop (`loop.py`), Muse + Gemini clients, critic (`critic.py`), stagnation tracking, settings, per-run progress, kernel record.
- `kernel_tools/` — the six tools: `read_problem`, `write_kernel`, `build_solution`, `run_eval`, `read_trace`, `record_lesson`.
- `knowledge/` — KernelWiki retrieval driven by measured symptoms.
- `evaluators/` — GPU-less B200 latency model, SOL adapter, calibration comparison.
- `runs/` — one folder per campaign: progress, kernels, solutions, traces, calibration pairs (gitignored).
- `solutions/` — frozen per-problem `solution.json` winners with notes.
- `collections/` — leaderboard submission packs built from `solutions/`.
- `learnings.md` — durable lessons across runs (agent writes via `record_lesson`, next runs read it in their objective).
- `tests/` — offline tests (no API calls, no GPU).
- `assets/make_demo_gif.py` — regenerates the loop diagram above (needs pillow).

## Quickstart

Needs API keys for both models plus a problem and an eval backend (see Setup below for the two supported layouts: full GPU host, or Modal-only with no local GPU).

```bash
cp .env.example .env   # then set MODEL_API_KEY, GEMINI_API_KEY, PROBLEM_DIR
./helper setup          # install agent deps, check keys + problem + evaluator
./helper test           # offline tests
./helper run --turns 30 # optimize PROBLEM_DIR; resume with --resume RUN_ID
./helper calibrate      # predicted-vs-measured B200 report over all runs
```

`./helper` uses `PYTHON` from `.env` (default `python3`). With `EVAL_BACKEND=local` it must be an interpreter with torch and the `sol-execbench` binary on PATH; with `EVAL_BACKEND=modal` any Python with `openai` + `modal` works and no local GPU is needed.

## Second opinion (dual model)

Muse drives the tool loop. When `GEMINI_API_KEY` is set, a failed `run_eval`
also triggers one stateless critic call (default `gemini-4-argon` via
`GEMINI_API_BASE_URL`): it reads the kernel plus the trace and appends a
diagnosis for the driver's next turn. The critic never calls tools and never
stops the run, and each version records its authoring model in `record.json`.

## Stagnation (KernelWiki + fresh eyes)

After `STAGNATION_THRESHOLD` evals (default 3) without a new best version, the
loop appends one guidance message: a KernelWiki technique page chosen by the
run's measured symptoms (best kernel's roofline bottleneck, else the problem
kind), plus — when the critic is configured — a fresh-eyes proposal for a
different approach. Each firing rotates to a technique not yet tried in the
run. The wiki checkout defaults to `../KernelWiki` (`KERNELWIKI_DIR`
overrides); a missing checkout silently skips the wiki block.

## B200 prediction (advisory)

Every `run_eval` of a Triton kernel on a GEMM-like problem also runs the
GPU-less B200 evaluator (ported from mini-kerngen): tile sizes come from the
kernel's launch site, shapes from the definition workloads, and the timed
Petri-net model predicts B200 latency and bottleneck. The prediction is
advisory only — it never blocks or fails an eval — and each attempt appends a
predicted-vs-measured pair to `runs/<run-id>/calibration.jsonl`, the dataset
for calibrating the model against real Modal B200 traces. Turn lines show it
as `[pred N.NNNms]`. Tier-1 exact arithmetic stays dormant (no sample input
tiles are supplied yet).

## Calibration report

`./helper calibrate` compares those pairs across runs: coverage, latency
agreement (geometric-mean and median predicted/measured ratio, fraction
within 2x), Spearman rank correlation, roofline-bottleneck agreement, and
hard-gate false rejects. `PASSED` pairs and `MIXED` pairs with no
correctness failure inform the comparison (missing B200 timings only shrink
the subset); use `--all` to lift the hardware filter, `--run` to scope to
one run, and `--json` for machine-readable output.

## Setup (native, no Docker)

```bash
# 1. SOL-ExecBench checkout as a sibling (already done once):
git clone https://github.com/NVIDIA/SOL-ExecBench.git ../SOL-ExecBench

# 2. Install into rsi without the pinned heavy stack (keeps torch 2.13):
/home/sgoswami/miniforge3/envs/rsi/bin/python -m pip install -e ../SOL-ExecBench --no-deps
/home/sgoswami/miniforge3/envs/rsi/bin/python -m pip install einops datasets safetensors numcodecs ninja "huggingface-hub[cli]" "cupti-python>=13.0.1"

# 3. Data (from the SOL-ExecBench checkout):
cd ../SOL-ExecBench && ./scripts/download_data.sh
# 3b. FlashInfer-Bench definitions ship with "hf_id": "", which the
# evaluator rejects (it wants null or non-empty): null them out.
python -c "
import json
from pathlib import Path
for p in Path('data/benchmark/FlashInfer-Bench').iterdir():
    f = p / 'definition.json'
    d = json.loads(f.read_text())
    if d.get('hf_id') == '':
        d['hf_id'] = None
        f.write_text(json.dumps(d, indent=4) + chr(10))
"

# 4. Sanity:
sol-execbench --help
sol-execbench examples/cute_dsl/jamba_attn_proj \
  --solution examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json
```

Local timings are directional: the leaderboard runs on B200s with locked clocks; expect ~5% run-to-run variance here (up to ~10% for many-small-kernel launches).

## Setup (Modal-only, no local GPU)

When evals run on Modal, the agent needs no torch, no CUDA, and no local
evaluator — only the benchmark data for problem definitions:

```bash
# 1. SOL-ExecBench checkout as a sibling (definitions + workloads only):
git clone https://github.com/NVIDIA/SOL-ExecBench.git ../SOL-ExecBench

# 2. Data + FlashInfer hf_id fix (same 3/3b as above):
cd ../SOL-ExecBench && ./scripts/download_data.sh
# ... then null the "" hf_id values (see step 3b above)

# 3. Agent env: any Python with the API clients:
python3 -m venv .venv && .venv/bin/pip install openai modal
```

Set `EVAL_BACKEND=modal` and `PYTHON` (to the venv interpreter) in `.env`.
`./helper setup` still expects a local `sol-execbench` binary, so on a
Modal-only box skip it and go straight to `./helper test` and `./helper run`.

## Modal B200 evals

`modal_eval.py` runs a frozen solution on a Modal B200 (same leaderboard-class GPU).
Needs `modal` configured (`modal setup`, `whatdhack` workspace):

```bash
modal run modal_eval.py::check   # validate GPU access (MODAL_GPU=T4 for a cheap check)
modal run modal_eval.py::main --problem <problem_dir> --solution <solution.json>
```

The image (CUDA 13 + torch 2.9 cu130 + evaluator from the sibling checkout) builds
once and caches. Works best for random/scalar-input workloads — problem and
solution are passed as text, no dataset mount. 085 reference: 16/16 PASSED,
0.027ms mean on B200 vs 0.711ms on the GB10. 005 GEMM reference
(hand-written Triton 128×256×64): 17/17 PASSED at 0.0505ms mean; the B200
model predicted 123µs for it.

Set `EVAL_BACKEND=modal` in `.env` to run the agent loop's evals on the Modal
GPU instead of the local one (each `run_eval` bills the workspace; turn lines
show `[modal]`). Partial traces are kept when some workloads fail: on B200,
microsecond kernels flake ~0-2 workloads per run with CUPTI
"kernel activity sequence not found" timing errors (evaluator-side, not the
kernel). SOL uses B200 peaks for B200 traces, GB10 peaks for local ones.
