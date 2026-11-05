import torch
import triton
import triton.language as tl

@triton.jit
def _gemm_nt(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cur_k = k * BLOCK_K + offs_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (cur_k[None, :] < K))
        b = tl.load(b_ptrs, mask=(cur_k[:, None] < K) & (offs_n[None, :] < N))
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def run(hidden_states, weight, logits, **kwargs):
    B, S, K = hidden_states.shape
    N = weight.shape[0]
    M = B * S
    hs2 = hidden_states.reshape(M, K)
    lg2 = logits.reshape(M, N)
    if M >= 128:
        BM, BN, BK, W, ST, GM = 128, 128, 64, 8, 3, 8
    elif M >= 32:
        BM, BN, BK, W, ST, GM = 64, 128, 64, 8, 3, 8
    else:
        BM, BN, BK, W, ST, GM = 16, 128, 32, 4, 4, 8
    grid = lambda META: (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _gemm_nt[grid](
        hs2, weight, lg2, M, N, K,
        hs2.stride(0), hs2.stride(1),
        weight.stride(0), weight.stride(1),
        lg2.stride(0), lg2.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM, NUM_STAGES=ST,
        num_warps=W,
    )
