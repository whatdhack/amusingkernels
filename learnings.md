# Kernel learnings

Durable lessons across runs. Injected into each run's objective.

## 2026-11-02 (run-20261102T173923Z-a1e9bd)
- On 085_geglu_activation (GB10, triton 3.7.1): a Triton kernel using the tanh intrinsic failed eval; rewriting with a manual tanh passed. Prefer manual tanh over the intrinsic here.
- In this env, torch.compile and CUDA C++/inline-CUDA solutions failed eval (compile/toolchain issues); Triton-first is validated, CUDA paths need toolchain work before retrying.
- GEGLU winner: 1D-grid Triton, manual tanh, 8 warps — PASSED 16/16 at 0.711ms mean, ~84% of GB10 roofline SOL. Small workloads top out at 54-70% of SOL (launch overhead); optimize for the large shapes.
