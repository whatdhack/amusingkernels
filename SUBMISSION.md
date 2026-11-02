# Submission notes

## What the site takes

- **Leaderboard submissions only**, via the [SOL-ExecBench leaderboard](https://research.nvidia.com/benchmarks/sol-execbench). Solutions run on NVIDIA B200 GPUs and are ranked by SOL-Score (roofline-based, see `sol_score.py` in the evaluator).
- An **NVIDIA account** is required to submit and track progress.
- There is **no debug-eval track** on the site: debugging is local with the `sol-execbench` CLI against a problem directory (`definition.json` + `workload.jsonl`).

## Per-problem artifact

One `solution.json` per problem (as built by `build_solution`):

- Required top-level: `name`, `definition`, `author`, `spec`, `sources`.
- `spec`: `languages`, `target_hardware` (include `"B200"`), `entry_point` (`"file::run"`), `destination_passing_style: true`, `dependencies`.
- `run()` takes definition inputs then pre-allocated outputs, writes in-place, returns nothing.
- Freeze the winning file per problem, e.g. `runs/<run-id>/kernels/v3/solution.json`.

## Reading the result

- Official runs lock B200 SM clocks at 1500 MHz; still ~5% run-to-run variance (up to ~10% for many-small-kernel launches). Treat local GB10 numbers as directional.
- Keep kernels honest: real computation, no shape specialization beyond the workload spec — reward-hacking checks run evaluator-side.

## Checklist

- [ ] NVIDIA account created, leaderboard submission page located (upload UI not yet verified — confirm file format there: single `solution.json` vs archive).
- [x] `solution.json` validates (required fields, DPS signature matches `definition.json`).
- [x] Local trace PASSED on all workloads (`run_eval` / `sol-execbench --json`).
- [x] Expected SOL-Score recorded from the local trace (latency + hardware note: GB10, unlocked clocks) — see `solutions/085_geglu_activation/NOTES.md`.
