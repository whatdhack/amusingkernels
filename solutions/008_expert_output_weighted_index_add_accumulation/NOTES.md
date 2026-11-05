# 008_expert_output_weighted_index_add_accumulation — frozen submission

- Run: `run-20261104T190439Z-f4e939` (30 turns, model `muse-spark-1.3-contributor`)
- Winner: v7, Triton, PASSED 16/16 workloads
- Local latency (GB10, unlocked clocks): 1.348ms mean
- GB10 roofline SOL: 0.612ms mean → 45.4% of SOL
- Baseline (v1 PyTorch port, same run): 1.884ms → winner is ~1.4x faster
- Files: `solution.json` (submission artifact), `kernel.py` (source as shipped inside it)

Re-verify locally with:

```bash
sol-execbench ../SOL-ExecBench/data/benchmark/L1/008_expert_output_weighted_index_add_accumulation \
  --solution solutions/008_expert_output_weighted_index_add_accumulation/solution.json --json
```

Leaderboard caveat: official scoring runs on B200s with locked 1500 MHz SM
clocks; treat the GB10 latency as directional and expect ~5% run-to-run variance.
