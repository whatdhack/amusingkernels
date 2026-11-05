# 031_repeat_kv_attention_matmul — frozen submission

- Run: `run-20261104T162224Z-27adbd` (30 turns, model `muse-spark-1.3-contributor`)
- Winner: v3, PyTorch, PASSED 16/16 workloads
- Local latency (GB10, unlocked clocks): 1.844ms mean
- GB10 roofline SOL: 1.198ms mean → 65.0% of SOL
- Baseline (v1 PyTorch port, same run): 18.172ms → winner is ~9.9x faster
- Files: `solution.json` (submission artifact), `kernel.py` (source as shipped inside it)

Re-verify locally with:

```bash
sol-execbench ../SOL-ExecBench/data/benchmark/L1/031_repeat_kv_attention_matmul \
  --solution solutions/031_repeat_kv_attention_matmul/solution.json --json
```

Leaderboard caveat: official scoring runs on B200s with locked 1500 MHz SM
clocks; treat the GB10 latency as directional and expect ~5% run-to-run variance.
