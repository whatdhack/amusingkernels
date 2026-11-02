# 085_geglu_activation — frozen submission

- Run: `run-20261102T173923Z-a1e9bd` (30 turns, model `muse-spark-1.3-contributor`)
- Winner: v3, Triton ("1D grid manual tanh 8 warps"), PASSED 16/16 workloads
- Local latency (GB10, unlocked clocks): 0.711ms mean
- Baseline PyTorch port: 1.614ms mean (same machine) → ~2.3x faster
- Reference (torch, `benchmark_reference`): 1.181ms mean
- Files: `solution.json` (submission artifact), `kernel.py` (source as shipped inside it)

Re-verify locally with:

```bash
sol-execbench ../SOL-ExecBench/data/benchmark/L1/085_geglu_activation \
  --solution solutions/085_geglu_activation/solution.json --json
```

Leaderboard caveat: official scoring runs on B200s with locked 1500 MHz SM
clocks; treat the GB10 latency as directional and expect ~5% run-to-run variance.
