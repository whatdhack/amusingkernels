import torch

@torch.no_grad()
def run(query, key, attn_weights, **kwargs):
    # query: [B,H,S,D], key: [B,Hk,S,D], out: [B,H,S,S]; H=Hk=32, D=128, groups=1.
    # Single cuBLAS strided-batched GEMM: out = alpha*(Q @ K^T) + 0*out,
    # alpha=scaling applied in fp32 accumulator, bf16 written once.
    # Exactly matches reference numerics (fp32 accumulate, scale, round to bf16).
    B, H, S, D = query.shape
    Bk, Hk, Sk, Dk = key.shape
    scale = Dk ** -0.5
    q = query.reshape(B * H, S, D)
    if Hk != H:
        rep = H // Hk
        k = key.reshape(B * Hk, Sk, Dk).transpose(1, 2).repeat_interleave(rep, dim=0)
    else:
        k = key.reshape(B * Hk, Sk, Dk).transpose(1, 2)
    out = attn_weights.reshape(B * H, S, S)
    torch.baddbmm(out, q, k, alpha=scale, beta=0.0, out=out)
