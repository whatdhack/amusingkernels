# 005_gemm_n256_k7168 — frozen submission

- Run: `run-manual-005-01` (manual9426 run: assistant-written kernel, CPU-interpreter-verified, no LLM driver)
- Winner: v1, Triton ("masked GEMM C=A@B.T, 128x256x64, 8 warps, 4 stages"), PASSED 17/17 workloads
- Measured latency (Modal B200): 0.0505ms mean
- B200 roofline SOL (memory-only, no SOLAR MACs): 0.0036ms mean → 7.2% of SOL
- B200 model prediction for v1: 123.3µs, SMEM_CAPACITY_CONSTRAINED (first calibration pair)
- Note: the agent's PyTorch v1 hit 0.012ms but never fully passed (B200 CUPTI flake); this Triton v1 is the fastest *full* pass
- Files: `solution.json` (submission artifact), `kernel.py` (source as shipped inside it)

Re-verify on Modal with:

```bash
modal run modal_eval.py::main --problem ../SOL-ExecBench/data/benchmark/FlashInfer-Bench/005_gemm_n256_k7168 \
  --solution solutions/005_gemm_n256_k7168/solution.json
```

Leaderboard caveat: official scoring runs on B200s with locked 1500 MHz SM
clocks; treat Modal latency as directional and expect ~5% run-to-run variance
plus occasional CUPTI timing flakes on microsecond kernels (re-run; a
RUNTIME_ERROR with "kernel activity sequence not found" is the profiler, not
the kernel).
