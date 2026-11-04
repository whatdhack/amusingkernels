import torch
import triton
import triton.language as tl

@triton.jit
def _av_kernel(
    A_ptr, V_ptr, O_ptr,
    s_ab, s_ah, s_aq, s_ak,
    s_vb, s_vh, s_vk, s_vd,
    s_ob, s_oq,
    Q, K,
    BQ: tl.constexpr, BK: tl.constexpr, D: tl.constexpr = 64,
):
    pid = tl.program_id(0)
    nq = tl.cdiv(Q, BQ)
    bh = pid // nq
    qb = pid % nq
    b = bh // 20
    h = bh % 20
    q0 = qb * BQ

    off_q = tl.arange(0, BQ)
    off_k = tl.arange(0, BK)
    off_d = tl.arange(0, D)
    q_idx = q0 + off_q
    q_mask = q_idx < Q

    A_base = A_ptr + b * s_ab + h * s_ah
    V_base = V_ptr + b * s_vb + h * s_vh
    O_base = O_ptr + b * s_ob + h * D

    acc = tl.zeros((BQ, D), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        k_idx = k0 * BK + off_k
        k_mask = k_idx < K
        a = tl.load(A_base + (q_idx[:, None] * s_aq + k_idx[None, :] * s_ak),
                    mask=q_mask[:, None] & k_mask[None, :], other=0.0)
        v = tl.load(V_base + (k_idx[:, None] * s_vk + off_d[None, :] * s_vd),
                    mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, v, acc)

    o = acc.to(tl.bfloat16)
    tl.store(O_base + (q_idx[:, None] * s_oq + off_d[None, :]), o, mask=q_mask[:, None])


def run(attention_weights, value, output, **kwargs):
    B = attention_weights.shape[0]
    Q = attention_weights.shape[2]
    K = attention_weights.shape[3]
    s_ab, s_ah, s_aq, s_ak = attention_weights.stride()
    s_vb, s_vh, s_vk, s_vd = value.stride()
    s_ob, s_oq, s_ol = output.stride()
    # adaptive tiling
    if Q <= 32:
        BQ, w, st = 32, 4, 3
    elif Q <= 256:
        BQ, w, st = 64, 4, 3
    else:
        BQ, w, st = 128, 8, 4
    if K <= 32:
        BK = 32
    elif K <= 128:
        BK = 64
    else:
        BK = 64
    # for small-K cross-attn with big Q, larger BQ still best; keep warps 8
    nq = (Q + BQ - 1) // BQ
    grid = (B * 20 * nq,)
    _av_kernel[grid](
        attention_weights, value, output,
        s_ab, s_ah, s_aq, s_ak,
        s_vb, s_vh, s_vk, s_vd,
        s_ob, s_oq,
        Q, K,
        BQ=BQ, BK=BK, D=64,
        num_warps=w, num_stages=st,
    )
