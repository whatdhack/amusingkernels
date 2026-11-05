# 077_whisper_decoder_output_projection — frozen submission

- Run: `run-20261104T163318Z-4505f2` (30 turns, model `muse-spark-1.3-contributor`)
- Winner: v8, Triton, PASSED 16/16 workloads
- Local latency (GB10, unlocked clocks): 9.489ms mean
- GB10 roofline SOL: 5.914ms mean → 62.3% of SOL
- Baseline (v1 PyTorch port, same run): 11.476ms → winner is ~1.2x faster
- Files: `solution.json` (submission artifact), `kernel.py` (source as shipped inside it)

Re-verify locally with:

```bash
sol-execbench ../SOL-ExecBench/data/benchmark/L1/077_whisper_decoder_output_projection \
  --solution solutions/077_whisper_decoder_output_projection/solution.json --json
```

Leaderboard caveat: official scoring runs on B200s with locked 1500 MHz SM
clocks; treat the GB10 latency as directional and expect ~5% run-to-run variance.
