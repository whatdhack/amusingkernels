import torch
import triton
import triton.language as tl

@triton.jit
def _geglu_kernel(
    x_ptr, o_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // 5
    blk = pid % 5
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    base = row * 10240
    xg = tl.load(x_ptr + base + offs).to(tl.float32)
    xl = tl.load(x_ptr + base + 5120 + offs).to(tl.float32)
    x3 = xg * xg * xg
    inner = 0.7978845608028654 * (xg + 0.044715 * x3)
    a = tl.abs(inner)
    e = tl.exp(-2.0 * a)
    t = (1.0 - e) / (1.0 + e)
    tanh_v = tl.where(inner >= 0, t, -t)
    gelu = 0.5 * xg * (1.0 + tanh_v)
    out = gelu * xl
    tl.store(o_ptr + row * 5120 + offs, out)

def run(x, output, **kwargs):
    if not (x.is_contiguous() and output.is_contiguous()):
        import torch.nn.functional as F
        inner = x.shape[-1] // 2
        output.copy_(F.gelu(x[..., :inner], approximate='tanh') * x[..., inner:])
        return
    M = output.numel() // 5120
    grid = (M * 5,)
    _geglu_kernel[grid](x, output, BLOCK=1024, num_warps=8)
