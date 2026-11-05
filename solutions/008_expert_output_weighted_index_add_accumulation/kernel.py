import torch
import triton
import triton.language as tl

@triton.jit
def _scatter_atomic_kernel(expert_ptr, out_ptr, tok_ptr, H, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    e = pid // 3
    hb = pid % 3
    offs = hb * BLOCK + tl.arange(0, BLOCK)
    t = tl.load(tok_ptr + e).to(tl.int32)
    v = tl.load(expert_ptr + e * H + offs)
    tl.atomic_add(out_ptr + t * H + offs, v)

@triton.jit
def _fill_perm_kernel(tok_ptr, cur_ptr, perm_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    t = tl.load(tok_ptr + offs, mask=mask).to(tl.int32)
    pos = tl.atomic_add(cur_ptr + t, 1, mask=mask)
    tl.store(perm_ptr + pos, offs, mask=mask)

@triton.jit
def _gather_kernel(final_ptr, expert_ptr, perm_ptr, starts_ptr, counts_ptr, out_ptr,
                   H, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // 3
    hb = pid % 3
    offs = hb * BLOCK + tl.arange(0, BLOCK)
    row = t * H
    acc = tl.load(final_ptr + row + offs).to(tl.float32)
    start = tl.load(starts_ptr + t).to(tl.int32)
    cnt = tl.load(counts_ptr + t).to(tl.int32)
    for k in range(16):
        if k < cnt:
            e_idx = tl.load(perm_ptr + start + k).to(tl.int32)
            e_val = tl.load(expert_ptr + e_idx * H + offs).to(tl.float32)
            acc = acc + e_val
    k = 16
    while k < cnt:
        e_idx = tl.load(perm_ptr + start + k).to(tl.int32)
        e_val = tl.load(expert_ptr + e_idx * H + offs).to(tl.float32)
        acc = acc + e_val
        k += 1
    tl.store(out_ptr + row + offs, acc.to(tl.bfloat16))

def run(final_hidden_states, expert_outputs, token_indices, output):
    B = final_hidden_states.shape[0]
    N = expert_outputs.shape[0]
    if N <= 32768:
        output.copy_(final_hidden_states)
        _scatter_atomic_kernel[(N * 3,)](expert_outputs, output, token_indices, 3072, BLOCK=1024, num_warps=4)
    else:
        counts = torch.bincount(token_indices, minlength=B).to(torch.int32)
        starts = torch.cumsum(counts, dim=0, dtype=torch.int32) - counts
        cur = starts.clone()
        perm = torch.empty(N, dtype=torch.int32, device=token_indices.device)
        BF = 2048
        _fill_perm_kernel[((N + BF - 1) // BF,)](token_indices, cur, perm, N, BLOCK=BF, num_warps=4)
        _gather_kernel[(B * 3,)](final_hidden_states, expert_outputs, perm, starts, counts,
                                 output, 3072, BLOCK=1024, num_warps=8)
