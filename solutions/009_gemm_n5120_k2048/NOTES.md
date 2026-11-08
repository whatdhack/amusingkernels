# 009_gemm_n5120_k2048 — frozen submission

- Run: `run-20261108T013827Z-3e1efd` (20-turn dual-model campaign, Modal B200)
- Winner: v1, PyTorch ("pytorch out matmul baseline"), PASSED 25/25 workloads
- Measured latency (Modal B200): 0.0436ms mean, max relative error 0.0
- B200 roofline SOL (memory-only, no SOLAR MACs): 0.0086ms mean → 19.8% of SOL
- Note: custom Triton v2 was ~1.6x slower (0.068ms) — cuBLAS wins here, so
  the winner is the plain `torch.matmul(A, B.T, out=C)` baseline. v1 needed
  4 evals to land a clean pass (B200 CUPTI flake on the first three).
- Files: `solution.json` (submission artifact), `kernel.py` (source as shipped inside it)

Re-verify on Modal with:

```bash
modal run modal_eval.py::main --problem ../SOL-ExecBench/data/benchmark/FlashInfer-Bench/009_gemm_n5120_k2048 \
  --solution solutions/009_gemm_n5120_k2048/solution.json
```

Leaderboard caveat: official scoring runs on B200s with locked 1500 MHz SM
clocks; treat Modal latency as directional and expect ~5% run-to-run variance
plus occasional CUPTI timing flakes on microsecond kernels (re-run; a
RUNTIME_ERROR with "kernel activity sequence not found" is the profiler, not
the kernel).
