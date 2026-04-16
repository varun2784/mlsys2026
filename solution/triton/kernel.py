"""
FP8 Block-Scale Fused MoE Kernel  –  v33
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

v29: ~2.0x geomean on Modal (TF32 GEMMs)
v31: torch._scaled_mm fp8 GEMMs → RUNTIME_ERROR (non-contig mat2)
v32: scalar scales + .contiguous() → INCORRECT_NUMERICAL abs_err~1e8 (scalar too coarse)
v33: rowwise/colwise scales + .contiguous() on mat2:
  - scale_a: [Tk, 1] per-row of A (max over H-blocks per token)
  - scale_b: [1, G1] per-col of W13.T (max over H-blocks per G1-block)
  - GEMM2: quantize C → fp8 with per-row exact scale [Tk, 1] + [1, H] W2 col scale
  - mat2 forced contiguous (required by cuBLAS fp8 GEMM)
  - _quantize_fp8 in eager mode (avoids inductor fp8 cast issues)
"""

import torch

_H      = 7168
_I      = 2048
_G1     = 4096      # 2 * _I
_E_LOC  = 32
_E_GLB  = 256
_TOP_K  = 8
_N_GRP  = 8
_TK_GRP = 4
_BLKSZ  = 128       # FP8 block-scale granularity
_FP8_MAX = 448.0    # torch.finfo(torch.float8_e4m3fn).max

# Precomputed local expert index range (reused across calls, moved to GPU lazily)
_LE_RANGE_CPU = torch.arange(_E_LOC)   # [0..31] on CPU
_LE_RANGE_GPU: torch.Tensor | None = None

# ─── Compiled fused kernels ───────────────────────────────────────────────────

@torch.compile(fullgraph=True, dynamic=True)
def _prep_A_scale(scale: torch.Tensor) -> torch.Tensor:
    """[56, T] → [T, 56] f32. Fuse cast + transpose + contiguous."""
    return scale.to(torch.float32).permute(1, 0).contiguous()


@torch.compile(fullgraph=True, dynamic=True)
def _finalize(out_f32: torch.Tensor, output: torch.Tensor) -> None:
    """Fuse fp32→bf16 cast + copy into 1 kernel."""
    output.copy_(out_f32.to(torch.bfloat16))


@torch.compile(fullgraph=True, dynamic=True)
def _swiglu(C_full: torch.Tensor) -> torch.Tensor:
    """Fuse sigmoid + 2 muls. Non-contiguous slices → strided Triton loads."""
    C_gate = C_full[:, :_I]
    C_up   = C_full[:, _I:]
    return (C_up * torch.sigmoid(C_up)) * C_gate


def _quantize_fp8(x: torch.Tensor):
    """fp32 [Tk, N] → (fp8 [Tk, N], per-row scale [Tk, 1]).
    Eager mode (not compiled) for fp8 cast compatibility."""
    amax  = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)   # [Tk, 1]
    scale = amax / _FP8_MAX
    xq    = (x / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return xq, scale                                               # [Tk, I], [Tk, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Routing – eager
# ─────────────────────────────────────────────────────────────────────────────
def _route(routing_logits, routing_bias, device, T):
    s   = torch.sigmoid(routing_logits.to(torch.float32))
    swb = s + routing_bias.to(torch.float32)

    grouped  = swb.view(T, _N_GRP, _E_GLB // _N_GRP)
    top2, _  = grouped.topk(2, dim=2, largest=True, sorted=False)
    gscores  = top2.sum(dim=2)

    _, gidx  = gscores.topk(_TK_GRP, dim=1, largest=True, sorted=False)
    gmask    = torch.zeros(T, _N_GRP, device=device)
    gmask.scatter_(1, gidx, 1.0)
    score_mask = (gmask.unsqueeze(2)
                       .expand(T, _N_GRP, _E_GLB // _N_GRP)
                       .reshape(T, _E_GLB))

    pruned      = swb.masked_fill(score_mask == 0, torch.finfo(torch.float32).min)
    _, topk_idx = pruned.topk(_TOP_K, dim=1, largest=True, sorted=False)

    M_tok = torch.zeros_like(s)
    M_tok.scatter_(1, topk_idx, 1.0)
    w    = s * M_tok
    wsum = w.sum(dim=1, keepdim=True) + 1e-20
    return topk_idx, w / wsum


def kernel(
    routing_logits:        torch.Tensor,   # f32   [T, 256]
    routing_bias:          torch.Tensor,   # bf16  [256]
    hidden_states:         torch.Tensor,   # fp8   [T, 7168]
    hidden_states_scale:   torch.Tensor,   # f32   [56, T]
    gemm1_weights:         torch.Tensor,   # fp8   [32, 4096, 7168]
    gemm1_weights_scale:   torch.Tensor,   # f32   [32, 32, 56]
    gemm2_weights:         torch.Tensor,   # fp8   [32, 7168, 2048]
    gemm2_weights_scale:   torch.Tensor,   # f32   [32, 56, 16]
    local_expert_offset:   int,
    routed_scaling_factor: float,
    output:                torch.Tensor,   # bf16  [T, 7168]
):
    T      = routing_logits.shape[0]
    device = hidden_states.device

    # ── Routing ───────────────────────────────────────────────────────────────
    topk_idx, weights_norm = _route(routing_logits, routing_bias, device, T)
    weights_scaled = weights_norm * routed_scaling_factor

    # A_scale: [56, T] → [T, 56] f32
    A_scale = _prep_A_scale(hidden_states_scale)   # [T, 56]

    out_f32     = torch.zeros(T, _H, dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    # ── Batch expert selection: one broadcast pass → 1 GPU-CPU sync ──────────
    global _LE_RANGE_GPU
    if _LE_RANGE_GPU is None or _LE_RANGE_GPU.device != device:
        _LE_RANGE_GPU = _LE_RANGE_CPU.to(device)
    ge_range = _LE_RANGE_GPU + local_start                        # [E_LOC]
    all_sel  = (topk_idx.unsqueeze(0) == ge_range.view(_E_LOC, 1, 1)).any(dim=2)
    active   = all_sel.any(dim=1)                                 # [E_LOC]

    for le in active.nonzero(as_tuple=False).squeeze(1).tolist():
        ge      = local_start + le
        tok_idx = all_sel[le].nonzero(as_tuple=False).squeeze(1)

        # ── GEMM1: A_fp8[Tk,H] @ W13_fp8.T[H,G1] ─────────────────────────
        # Row-wise A scale: max over H-blocks per token → [Tk, 1]
        s_a = A_scale[tok_idx].amax(dim=1, keepdim=True)          # [Tk, 1]
        # Col-wise scale for W13.T [H,G1]: per G1-block, max over H-blocks
        # gemm1_weights_scale[le]: [G1//128, H//128] = [32, 56]
        s_w13 = (gemm1_weights_scale[le]            # [32, 56]
                 .amax(dim=1)                        # [32]
                 .repeat_interleave(_BLKSZ)          # [G1=4096]
                 .unsqueeze(0))                      # [1, G1]

        # .contiguous() required by cuBLAS fp8 GEMM (non-contig .t() fails)
        W13_T = gemm1_weights[le].t().contiguous()   # [H, G1] fp8

        C_full = torch._scaled_mm(
            hidden_states[tok_idx],     # [Tk, H] fp8 e4m3fn
            W13_T,                      # [H, G1] fp8 e4m3fn
            scale_a=s_a,
            scale_b=s_w13,
            out_dtype=torch.float32,
            use_fast_accum=False,
        )                                           # [Tk, G1]

        C = _swiglu(C_full)                         # [Tk, I]

        # ── Quantize SwiGLU output → fp8 for GEMM2 ────────────────────────
        C_fp8, s_c = _quantize_fp8(C)               # [Tk, I] fp8, [Tk, 1]

        # ── GEMM2: C_fp8[Tk,I] @ W2_fp8.T[I,H] ───────────────────────────
        # Col-wise scale for W2.T [I,H]: per H-block, max over I-blocks
        # gemm2_weights_scale[le]: [H//128, I//128] = [56, 16]
        s_w2 = (gemm2_weights_scale[le]             # [56, 16]
                .amax(dim=1)                         # [56]
                .repeat_interleave(_BLKSZ)           # [H=7168]
                .unsqueeze(0))                       # [1, H]

        W2_T = gemm2_weights[le].t().contiguous()   # [I, H] fp8

        O = torch._scaled_mm(
            C_fp8,                      # [Tk, I] fp8
            W2_T,                       # [I, H] fp8
            scale_a=s_c,
            scale_b=s_w2,
            out_dtype=torch.float32,
            use_fast_accum=False,
        )                                           # [Tk, H]

        w_tok = weights_scaled[tok_idx, ge]
        out_f32.index_add_(0, tok_idx, O * w_tok.unsqueeze(1))

    _finalize(out_f32, output)
