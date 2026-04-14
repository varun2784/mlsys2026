"""
FP8 Block-Scale Fused MoE Kernel  –  v21 (= v18c restored)
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Version history:
v14 (no cache, no TF32):     19/19 PASSED, 1.0-4.33x peak
v17 (torch.compile dequant): 19/19 PASSED, 1.10-4.52x, geomean ~1.66x
v18c (+ compiled SwiGLU):    19/19 PASSED, 1.20-7.16x, geomean ~2.54x  ← BEST
v19 (full expert compiled):  19/19 PASSED, WORSE — inductor GEMM < cuBLAS
v20 (+ compiled accum):      19/19 PASSED, WORSE — dynamic=True hurts small-T
v21: restore v18c (confirmed best configuration)

Key techniques:
1. Expert skipping: if not sel.any(): continue (critical for small-T speedup)
2. view+broadcast dequant: no cache, no ABA, per-call for W13/W2
3. torch.compile fused dequant: fuses fp8→fp32 + scale mul into 1 Triton kernel
4. torch.compile SwiGLU with dynamic=True: fuses sigmoid + 2 muls, handles
   non-contiguous C_full slices via strided Triton loads (saves ~1.4ms large-T)
5. cuBLAS TF32 for GEMM1 + GEMM2 (tensor cores, not WGMMA which crashes on B200)
"""

import torch

_H     = 7168
_I     = 2048
_G1    = 4096   # 2 * _I
_E_LOC = 32
_E_GLB = 256
_TOP_K = 8
_N_GRP = 8
_TK_GRP= 4
_BLKSZ = 128   # FP8 block-scale granularity

# ─── Compiled fused kernels ───────────────────────────────────────────────────

@torch.compile(fullgraph=True)
def _dequant_A(A_fp8: torch.Tensor, A_scale: torch.Tensor) -> torch.Tensor:
    """[Tk, H] fp8 + [Tk, 56] scale → [Tk, H] fp32. One HBM pass."""
    Tk = A_fp8.shape[0]
    return (A_fp8.to(torch.float32).view(Tk, 56, _BLKSZ)
            * A_scale.unsqueeze(2)).view(Tk, _H)


@torch.compile(fullgraph=True)
def _dequant_W13(W13_fp8: torch.Tensor, S13: torch.Tensor) -> torch.Tensor:
    """[G1, H] fp8 + [32, 56] scale → [G1, H] fp32. Fixed shape, compiled once."""
    return (W13_fp8.to(torch.float32)
            .view(_G1 // _BLKSZ, _BLKSZ, _H // _BLKSZ, _BLKSZ)
            * S13.unsqueeze(1).unsqueeze(3)).view(_G1, _H)


@torch.compile(fullgraph=True)
def _dequant_W2(W2_fp8: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """[H, I] fp8 + [56, 16] scale → [H, I] fp32. Fixed shape, compiled once."""
    return (W2_fp8.to(torch.float32)
            .view(_H // _BLKSZ, _BLKSZ, _I // _BLKSZ, _BLKSZ)
            * S2.unsqueeze(1).unsqueeze(3)).view(_H, _I)


@torch.compile(fullgraph=True, dynamic=True)
def _swiglu(C_full: torch.Tensor) -> torch.Tensor:
    """Fuse sigmoid + 2 muls into 1 kernel. Non-contiguous slices → strided Triton loads.
    Saves ~1.4ms on large-T by eliminating 2 extra HBM passes over C_up."""
    C_gate = C_full[:, :_I]
    C_up   = C_full[:, _I:]
    return (C_up * torch.sigmoid(C_up)) * C_gate


# ─────────────────────────────────────────────────────────────────────────────
# Routing – eager (scatter_ causes graph breaks; compile hurts small-T)
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

    # A_scale: [56, T] → [T, 56]
    A_scale = (hidden_states_scale.to(torch.float32)
                                  .permute(1, 0)
                                  .contiguous())   # [T, 56]

    out_f32     = torch.zeros(T, _H, dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(_E_LOC):
        ge = local_start + le
        if ge < 0 or ge >= _E_GLB:
            continue

        sel = (topk_idx == ge).any(dim=1)
        if not sel.any():
            continue

        tok_idx = sel.nonzero(as_tuple=False).squeeze(1)

        A_fp32   = _dequant_A(hidden_states[tok_idx], A_scale[tok_idx])
        W13_fp32 = _dequant_W13(gemm1_weights[le], gemm1_weights_scale[le])
        C_full   = torch.mm(A_fp32, W13_fp32.t())     # GEMM1: cuBLAS TF32
        C        = _swiglu(C_full)
        W2_fp32  = _dequant_W2(gemm2_weights[le], gemm2_weights_scale[le])
        O        = torch.mm(C, W2_fp32.t())            # GEMM2: cuBLAS TF32

        w_tok = weights_scaled[tok_idx, ge]
        out_f32.index_add_(0, tok_idx, O * w_tok.unsqueeze(1))

    output.copy_(out_f32.to(torch.bfloat16))
