# 083_attention_score_value_matmul — frozen submission

- Run: `run-20261104T075019Z-625190` (30 turns, model `muse-spark-1.3-contributor`)
- Winner: v3, Triton, PASSED 16/16 workloads
- Local latency (GB10, unlocked clocks): 0.334ms mean
- GB10 roofline SOL (fused, SOLAR MACs + 273 GB/s): 0.223ms mean → 66.8% of SOL
- PyTorch-port baseline (v1, same run): 0.524ms → winner is ~1.6x faster
- Files: `solution.json` (submission artifact), `kernel.py` (source as shipped inside it)

Re-verify locally with:

```bash
sol-execbench ../SOL-ExecBench/data/benchmark/L1/083_attention_score_value_matmul \
  --solution solutions/083_attention_score_value_matmul/solution.json --json
```

Leaderboard caveat: official scoring runs on B200s with locked 1500 MHz SM
clocks; treat the GB10 latency as directional and expect ~5% run-to-run variance.
Compute-peak side of the SOL uses estimated GB10 FLOPS (memory side is the solid one).
