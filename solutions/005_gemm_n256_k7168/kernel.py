"""005 v1: Triton GEMM C = A @ B.T, fp16 in/out, fp32 accumulate.

A[M, K], B[N, K] row-major; B is read as-is (no transpose in global memory),
the transpose happens in registers via tl.trans. Masked on M and the K tail.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_nt(a_ptr, b_ptr, c_ptr, M, N, K,
             stride_am, stride_ak, stride_bn, stride_bk,
             stride_cm, stride_cn,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining))
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < k_remaining))
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def run(A, B, C):
    M, K = A.shape
    N, KB = B.shape
    assert K == KB, f"K mismatch: {K} vs {KB}"
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 256))
    _gemm_nt[grid](A, B, C, M, N, K,
                   A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                   C.stride(0), C.stride(1),
                   BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,
                   num_warps=8, num_stages=4)
