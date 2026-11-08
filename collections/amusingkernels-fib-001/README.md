# amusingkernels-fib-001

First FlashInfer-Bench collections submission: 2 fp16 GEMMs
(`C = A @ B.T`), verified on Modal B200. Built by
`collections/build.py --name amusingkernels-fib-001` from `solutions/` (one
`solution.json` per definition + `manifest.json`).

| Definition | Cat. | Winner | Time (B200) | SOL | % SOL | Passed |
|---|---|---|---|---|---|---|
| 004_gemm_n128_k2048 | gemm | v2 triton | 0.0198ms | 0.0019ms | 9.3% | 25/25 |
| 005_gemm_n256_k7168 | gemm | v1 triton | 0.0505ms | 0.0036ms | 7.2% | 17/17 |

SOL here is memory-only (no SOLAR MACs on the Modal box), so the percentages
are lower bounds. 004's winner came from a 20-turn dual-model campaign;
005's is a manual kernel (see its NOTES.md).

Upload: pending — the site's collections format is unverified (guide page is
JS-rendered). If the site wants a different layout, adjust `build.py`.
