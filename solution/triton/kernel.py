"""
FP8 Block-Scale Fused MoE Kernel  –  v19
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

v17: torch.compile dequant W13/W2 → 2.5ms savings, 4.52x peak
v18c: + compiled SwiGLU → 7.16x peak, geomean ~2.54x (BREAKTHROUGH)
v19: compile entire per-expert computation as one function:
     dequant_A + dequant_W13 + GEMM1 + SwiGLU + dequant_W2 + GEMM2
     → inductor can optimize full computation graph, possibly eliminate
       intermediate materializations and schedule memory more efficiently.
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


@torch.compile(fullgraph=True, dynamic=True)
def _expert_fwd(
    A_fp8:   torch.Tensor,   # [Tk, H] fp8
    A_sc:    torch.Tensor,   # [Tk, 56] fp32
    W13_fp8: torch.Tensor,   # [G1, H] fp8
    S13:     torch.Tensor,   # [32, 56] fp32
    W2_fp8:  torch.Tensor,   # [H, I] fp8
    S2:      torch.Tensor,   # [56, 16] fp32
) -> torch.Tensor:
    """Full per-expert forward: dequant + GEMM1 + SwiGLU + dequant + GEMM2."""
    Tk = A_fp8.shape[0]

    # A dequant
    A = (A_fp8.to(torch.float32).view(Tk, 56, _BLKSZ) * A_sc.unsqueeze(2)).view(Tk, _H)

    # W13 dequant
    W13 = (W13_fp8.to(torch.float32)
           .view(_G1 // _BLKSZ, _BLKSZ, _H // _BLKSZ, _BLKSZ)
           * S13.unsqueeze(1).unsqueeze(3)).view(_G1, _H)

    # GEMM1: cuBLAS via torch.mm
    Cf = torch.mm(A, W13.t())   # [Tk, G1]

    # SwiGLU (non-contiguous slices: inductor handles strided loads natively)
    Cg = Cf[:, :_I]
    Cu = Cf[:, _I:]
    C  = (Cu * torch.sigmoid(Cu)) * Cg  # [Tk, I]

    # W2 dequant
    W2 = (W2_fp8.to(torch.float32)
          .view(_H // _BLKSZ, _BLKSZ, _I // _BLKSZ, _BLKSZ)
          * S2.unsqueeze(1).unsqueeze(3)).view(_H, _I)

    # GEMM2: cuBLAS via torch.mm
    return torch.mm(C, W2.t())   # [Tk, H]


# ─────────────────────────────────────────────────────────────────────────────
# Routing – eager (scatter_ ops cause graph breaks in torch.compile)
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

        # ── Compiled full expert forward (dequant + GEMM1 + SwiGLU + dequant + GEMM2) ──
        O = _expert_fwd(
            hidden_states[tok_idx], A_scale[tok_idx],
            gemm1_weights[le], gemm1_weights_scale[le],
            gemm2_weights[le], gemm2_weights_scale[le],
        )   # [Tk, H]

        w_tok = weights_scaled[tok_idx, ge]
        out_f32.index_add_(0, tok_idx, O * w_tok.unsqueeze(1))

    output.copy_(out_f32.to(torch.bfloat16))
