import torch
import triton
import triton.language as tl

@triton.jit
def _fused_act_kernel(
    gu_ptr, gated_ptr, bias_ptr,
    T, DD,
    stride_gu_e, stride_gated_e, stride_bias_e,
    alpha, limit,
    BLOCK: tl.constexpr,
):
    pid_row = tl.program_id(0)  # e * T + t  (e is LOCAL chunk index)
    pid_col = tl.program_id(1)  # d-block
    e = pid_row // T
    t = pid_row % T
    d = pid_col * BLOCK + tl.arange(0, BLOCK)
    mask = d < DD
    dd2 = DD * 2
    row_base = e * stride_gu_e + t * dd2
    g = tl.load(gu_ptr + row_base + 2 * d, mask=mask)
    u = tl.load(gu_ptr + row_base + 2 * d + 1, mask=mask)
    bbase = e * stride_bias_e
    bg = tl.load(bias_ptr + bbase + 2 * d, mask=mask)
    bu = tl.load(bias_ptr + bbase + 2 * d + 1, mask=mask)
    g = g + bg
    u = u + bu
    g = tl.minimum(g, limit)
    u = tl.minimum(tl.maximum(u, -limit), limit)
    s = 1.0 / (1.0 + tl.exp(-g * alpha))
    out = (u + 1.0) * (g * s)
    tl.store(gated_ptr + e * stride_gated_e + t * DD + d, out, mask=mask)

_USE_TRITON = {}

def _torch_act(gu, bgu, alpha, limit):
    gu += bgu.unsqueeze(1)
    gate = gu[..., ::2]
    up = gu[..., 1::2]
    gate_c = torch.clamp(gate, max=limit)
    up_c = torch.clamp(up, min=-limit, max=limit)
    gate_c.mul_(torch.sigmoid(gate_c * alpha))
    up_c.add_(1.0)
    return up_c * gate_c

def run(hidden_states, routing_weights, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, alpha, limit, output, **kwargs):
    with torch.no_grad():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        B, S, H = hidden_states.shape
        T = B * S
        E = gate_up_proj.shape[0]
        D = down_proj.shape[1]
        hidden_flat = hidden_states.reshape(T, H)
        out_flat = output.reshape(T, H)
        if isinstance(alpha, torch.Tensor):
            alpha_f = float(alpha.reshape(-1)[0].item()) if alpha.numel() else 0.0
        else:
            alpha_f = float(alpha)
        if isinstance(limit, torch.Tensor):
            limit_f = float(limit.reshape(-1)[0].item()) if limit.numel() else 0.0
        else:
            limit_f = float(limit)
        # Fold all down-bias terms into one GEMM: sum_e w_te * bd_e
        out_flat.copy_(torch.mm(routing_weights, down_proj_bias))
        rw_t = routing_weights.transpose(0, 1).contiguous()  # (E,T)
        # gu tile (cc,T,2D) fp32 <= ~470MB  =>  cc*T <= 20000
        C = 20000 // max(T, 1)
        if C < 1:
            C = 1
        if C > 64:
            C = 64
        if C > E:
            C = E
        BLOCK = 1024
        nblk = (D + BLOCK - 1) // BLOCK
        key = (hidden_flat.device.index if hidden_flat.is_cuda else -1)
        use_triton = _USE_TRITON.get(key, True)
        for e0 in range(0, E, C):
            e1 = e0 + C if e0 + C < E else E
            cc = e1 - e0
            Wgu = gate_up_proj[e0:e1]
            Wd = down_proj[e0:e1]
            hid = hidden_flat.unsqueeze(0).expand(cc, T, H).contiguous()
            gu = torch.bmm(hid, Wgu)
            del hid
            gated = None
            if use_triton:
                try:
                    gated = torch.empty((cc, T, D), device=gu.device, dtype=gu.dtype)
                    grid = (cc * T, nblk)
                    _fused_act_kernel[grid](
                        gu, gated, gate_up_proj_bias[e0:e1].contiguous(),
                        T, D,
                        gu.stride(0), gated.stride(0), 2 * D,
                        alpha_f, limit_f,
                        BLOCK=BLOCK,
                    )
                except Exception:
                    gated = None
                    _USE_TRITON[key] = False
                    use_triton = False
            if gated is None:
                gated = _torch_act(gu, gate_up_proj_bias[e0:e1], alpha_f, limit_f)
            del gu
            out_e = torch.bmm(gated, Wd)
            del gated
            w = rw_t[e0:e1].transpose(0, 1)  # (T,cc)
            out_flat += torch.einsum('tc,cth->th', w, out_e)
            del out_e, w
