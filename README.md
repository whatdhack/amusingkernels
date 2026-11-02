# amusingkernels

A kernel-optimization agent for [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench): it writes GPU kernels (Triton-first), debugs them against the local evaluator, and produces leaderboard-ready `solution.json` files. The agentic loop is ported from the legal-agent workshop (`loop.py` skeleton: model picks one tool per turn, progress saved each turn, pause/resume); everything domain-specific is new.

Native Python on the host GPU — no Docker anywhere.

## Layout

- `PLAN.md` — goal, approach, steps, validation, risks.
- `SUBMISSION.md` — leaderboard submission path (account, upload, SOL-Score reading).
- `kernel_agent/` — the loop (`loop.py`), model client, settings, per-run progress, kernel record.
- `kernel_tools/` — the five tools: `read_problem`, `write_kernel`, `build_solution`, `run_eval`, `read_trace`.
- `runs/` — one folder per campaign: progress, kernels, solutions, traces (gitignored).
- `tests/` — offline tests (no API calls, no GPU).

## Quickstart

Uses the `rsi` conda env (Python 3.13, torch 2.13 cu130, GB10 host). The SOL-ExecBench checkout lives next to this repo at `../SOL-ExecBench` and is installed into `rsi` with `pip install -e . --no-deps` plus the small runtime deps (see Setup below).

```bash
cp .env.example .env   # then set MODEL_API_KEY and PROBLEM_DIR
./helper setup          # install agent deps, check key + problem + evaluator
./helper test           # offline tests
./helper run --turns 30 # optimize PROBLEM_DIR; resume with --resume RUN_ID
```

`./helper` uses `PYTHON` from `.env` (default `python3`); point it at the `rsi` interpreter. The evaluator binary is `SOL_EXECBENCH` (default `sol-execbench` on PATH).

## Setup (native, no Docker)

```bash
# 1. SOL-ExecBench checkout as a sibling (already done once):
git clone https://github.com/NVIDIA/SOL-ExecBench.git ../SOL-ExecBench

# 2. Install into rsi without the pinned heavy stack (keeps torch 2.13):
/home/sgoswami/miniforge3/envs/rsi/bin/python -m pip install -e ../SOL-ExecBench --no-deps
/home/sgoswami/miniforge3/envs/rsi/bin/python -m pip install einops datasets safetensors numcodecs ninja "huggingface-hub[cli]" "cupti-python>=13.0.1"

# 3. Data (from the SOL-ExecBench checkout):
cd ../SOL-ExecBench && ./scripts/download_data.sh

# 4. Sanity:
sol-execbench --help
sol-execbench examples/cute_dsl/jamba_attn_proj \
  --solution examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json
```

Local timings are directional: the leaderboard runs on B200s with locked clocks; expect ~5% run-to-run variance here (up to ~10% for many-small-kernel launches).
