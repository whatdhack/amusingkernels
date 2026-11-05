# amusingkernels-l1-001

First collections submission: 7 L1 kernels (2 attention, 2 GEMM, 2 MoE, 1 activation),
all agent-optimized and locally verified on GB10. Built by `collections/build.py`
from `solutions/` (one `solution.json` per definition + `manifest.json`).

| Definition | Cat. | Winner | Time (GB10) | SOL | % SOL | Passed |
|---|---|---|---|---|---|---|
| 085_geglu_activation | activation | v3 triton | 0.711ms | 0.599ms | 84.2% | 16/16 |
| 083_attention_score_value_matmul | attention | v3 triton | 0.334ms | 0.223ms | 66.8% | 16/16 |
| 031_repeat_kv_attention_matmul | attention | v3 pytorch | 1.844ms | 1.198ms | 65.0% | 16/16 |
| 077_whisper_decoder_output_projection | gemm | v8 triton | 9.489ms | 5.914ms | 62.3% | 16/16 |
| 063_attention_output_reshape_and_projection | gemm | v8 pytorch | 5.455ms | 3.545ms | 65.0% | 16/16 |
| 076_batched_expert_forward | moe | v6 triton | 422.98ms | 404.33ms | 95.6% | 14/14 |
| 008_expert_output_weighted_index_add_accumulation | moe | v7 triton | 1.348ms | 0.612ms | 45.4% | 16/16 |

Upload: pending — the site's collections format is unverified (guide page is
JS-rendered). If the site wants a different layout, adjust `build.py`.
