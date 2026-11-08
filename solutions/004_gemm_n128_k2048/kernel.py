import torch
import triton
import triton.language as tl

@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rn[:, None] * stride_bn + offs_k[None, :] * stride_bk,
                    mask=(rn[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
    c = acc.to(tl.float16)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, c,
             mask=(rm[:, None] < M) & (rn[None, :] < N))

def run(A, B, C):
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[0]
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
