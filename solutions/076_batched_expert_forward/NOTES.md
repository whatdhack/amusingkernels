# 076_batched_expert_forward — frozen submission

- Run: `run-20261104T171406Z-ceaeb7` (30 turns, model `muse-spark-1.3-contributor`)
- Winner: v6, Triton, PASSED 14/14 workloads
- Local latency (GB10, unlocked clocks): 422.98ms mean
- GB10 roofline SOL (calibrated FP32 peak): 404.33ms mean → 95.6% of SOL
- Baseline (v2 PyTorch port, same run): 635.64ms → winner is ~1.5x faster
- Files: `solution.json` (submission artifact), `kernel.py` (source as shipped inside it)

Re-verify locally with:

```bash
sol-execbench ../SOL-ExecBench/data/benchmark/L1/076_batched_expert_forward \
  --solution solutions/076_batched_expert_forward/solution.json --json
```

Leaderboard caveat: official scoring runs on B200s with locked 1500 MHz SM
clocks; treat the GB10 latency as directional and expect ~5% run-to-run variance.
