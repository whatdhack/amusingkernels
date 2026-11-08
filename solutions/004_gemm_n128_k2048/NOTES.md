# 004_gemm_n128_k2048 — frozen submission

- Run: `run-20261108T004300Z-6d221c` (20-turn dual-model campaign, Modal B200)
- Winner: v2, Triton ("single-kernel Triton GEMM baseline"), PASSED 25/25 workloads
- Measured latency (Modal B200): 0.0198ms mean
- B200 roofline SOL (memory-only, no SOLAR MACs): 0.0019ms mean → 9.3% of SOL
- B200 model prediction for v2: 36µs, within 1.8x of measured (see calibration pairs)
- Note: v1 PyTorch hit 0.0091ms but never fully passed (B200 CUPTI flake);
  v3 was faster than v2 (0.0134ms) but dropped one workload to the same
  flake in both of its evals, so v2 stays the fastest *full* pass
- Files: `solution.json` (submission artifact), `kernel.py` (source as shipped inside it)

Re-verify on Modal with:

```bash
modal run modal_eval.py::main --problem ../SOL-ExecBench/data/benchmark/FlashInfer-Bench/004_gemm_n128_k2048 \
  --solution solutions/004_gemm_n128_k2048/solution.json
```

Leaderboard caveat: official scoring runs on B200s with locked 1500 MHz SM
clocks; treat Modal latency as directional and expect ~5% run-to-run variance
plus occasional CUPTI timing flakes on microsecond kernels (re-run; a
RUNTIME_ERROR with "kernel activity sequence not found" is the profiler, not
the kernel).
