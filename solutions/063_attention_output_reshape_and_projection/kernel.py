import torch
import triton
import triton.language as tl

def run(attn_output, o_proj_weight, output):
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    hidden_size = o_proj_weight.shape[0]
    a2d = attn_output.transpose(1, 2).reshape(bsz * seq_len, num_heads * v_head_dim)
    out2d = output.reshape(bsz * seq_len, hidden_size)
    torch.mm(a2d, o_proj_weight.t(), out=out2d)
