# Kernel learnings

Durable lessons across runs. Injected into each run's objective.

## 2026-11-02 (run-20261102T173923Z-a1e9bd)
- On 085_geglu_activation (GB10, triton 3.7.1): a Triton kernel using the tanh intrinsic failed eval; rewriting with a manual tanh passed. Prefer manual tanh over the intrinsic here.
- In this env, torch.compile and CUDA C++/inline-CUDA solutions failed eval (compile/toolchain issues); Triton-first is validated, CUDA paths need toolchain work before retrying.
- GEGLU winner: 1D-grid Triton, manual tanh, 8 warps — PASSED 16/16 at 0.711ms mean, ~84% of GB10 roofline SOL. Small workloads top out at 54-70% of SOL (launch overhead); optimize for the large shapes.

## 2026-11-04 (run-20261104T162224Z-27adbd)
- On fused Q@K^T*scale (031): single torch.baddbmm with alpha=scaling, beta=0, out=preallocated hits ~65% SOL and 8.5x over fp32 reference; separate bmm+scale or fp32 upcast passes create a full extra read/write of the huge [B,H,S,S] output and are far slower. Keep bf16 Tensor-Core path, never materialize expanded K.

## 2026-11-08 (operator notes from the 004/005 B200 campaigns)
- On Modal B200, microsecond kernels flake CUPTI timing: a workload with status RUNTIME_ERROR and log "kernel activity sequence not found" is the profiler, not the kernel — re-run the eval instead of rewriting the kernel. MIXED results from flakes still carry valid timings for the passed workloads.
- FlashInfer-Bench GEMMs are `C = A @ B.T` with B stored `[N, K]` (no transpose in global memory); read B as-is and transpose in registers. A PyTorch `out=` matmul baseline is a strong opener (0.009–0.012ms on B200 for 004/005).
- FlashInfer-Bench definitions ship with `"hf_id": ""`, which the evaluator rejects — null it after downloading (README step 3b).

## 2026-11-08 (run-20261108T013827Z-3e1efd)
- On FlashInfer GEMM 009 (C=A@B.T, N=5120 K=2048) B200: plain torch.matmul(A,B.T,out=C) PASSED 25/25 at 0.0436ms mean; custom Triton NT kernel was ~2x slower (0.068ms) so cuBLAS wins here. 'Expected kernel activity sequence not found ... nvjet...' RUNTIME_ERRORs on random large-M workloads are CUPTI profiler flakes, not kernel bugs — re-run the same version instead of rewriting.
