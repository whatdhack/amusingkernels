# 063_attention_output_reshape_and_projection — frozen submission

- Run: `run-20261104T165631Z-4d19a2` (30 turns, model `muse-spark-1.3-contributor`)
- Winner: v8, PyTorch, PASSED 16/16 workloads
- Local latency (GB10, unlocked clocks): 5.455ms mean
- GB10 roofline SOL: 3.545ms mean → 65.0% of SOL
- Baseline (v1 PyTorch port, same run): 5.477ms → winner is ~1.0x (all five
  Triton attempts v3–v7 failed eval; the fused reshape+projection needs work)
- Files: `solution.json` (submission artifact), `kernel.py` (source as shipped inside it)

Re-verify locally with:

```bash
sol-execbench ../SOL-ExecBench/data/benchmark/L1/063_attention_output_reshape_and_projection \
  --solution solutions/063_attention_output_reshape_and_projection/solution.json --json
```

Leaderboard caveat: official scoring runs on B200s with locked 1500 MHz SM
clocks; treat the GB10 latency as directional and expect ~5% run-to-run variance.
