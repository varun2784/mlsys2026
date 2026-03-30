"""
FP8 Block-Scale Fused MoE Kernel  –  v5 (reference-exact for correctness baseline)
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

This version is a direct port of the reference implementation wrapped in DPS.
Goal: confirm correctness before adding optimisations back.

Differences from reference:
  - DPS signature (output tensor passed in, not returned)
  - Vectorised routing (no per-expert sel_mask.any() GPU→CPU sync)
  - Single dispatch sort upfront (avoids per-expert index_select)
  - Per-expert matmul uses slices (not index_select) → still GPU-async
  - bf16 matmul (reference uses fp32)
"""

import torch
import torch.nn.functional as F

# ── Problem constants ──────────────────────────────────────────────────────────
_H      = 7168
_I      = 2048
_G1     = 4096
_E_LOC  = 32
_E_GLB  = 256
_TOP_K  = 8
_N_GRP  = 8
_TK_GRP = 4
_BLKSZ  = 128


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

    # ── FP8 dequantisation (exact reference style) ─────────────────────────────
    A_fp32 = hidden_states.to(torch.float32)                         # [T, H]
    A_scale_TH = hidden_states_scale.to(torch.float32).permute(1, 0).contiguous()  # [T, 56]
    A_scale_exp = (A_scale_TH.unsqueeze(-1)
                              .repeat(1, 1, _BLKSZ)
                              .reshape(T, _H))
    A = A_fp32 * A_scale_exp                                         # [T, H] f32

    W13_fp32 = gemm1_weights.to(torch.float32)                       # [E, G1, H]
    S13 = gemm1_weights_scale.to(torch.float32)                      # [E, 32, 56]
    S13_exp = (torch.repeat_interleave(S13, _BLKSZ, dim=1)
                    .repeat_interleave(_BLKSZ, dim=2))               # [E, G1, H]
    W13 = W13_fp32 * S13_exp                                         # [E, G1, H] f32

    W2_fp32 = gemm2_weights.to(torch.float32)                        # [E, H, I]
    S2 = gemm2_weights_scale.to(torch.float32)                       # [E, 56, 16]
    S2_exp = (torch.repeat_interleave(S2, _BLKSZ, dim=1)
                   .repeat_interleave(_BLKSZ, dim=2))                # [E, H, I]
    W2 = W2_fp32 * S2_exp                                            # [E, H, I] f32

    # ── Routing (vectorised, matches reference exactly) ────────────────────────
    logits = routing_logits.to(torch.float32)
    bias   = routing_bias.to(torch.float32)

    s   = torch.sigmoid(logits)            # [T, 256]
    swb = s + bias                         # [T, 256]

    g_sz    = _E_GLB // _N_GRP            # 32
    grouped = swb.view(T, _N_GRP, g_sz)   # [T, 8, 32]
    top2, _ = grouped.topk(2, dim=2, largest=True, sorted=False)
    gscores = top2.sum(dim=2)             # [T, 8]

    _, gidx = gscores.topk(_TK_GRP, dim=1, largest=True, sorted=False)
    gmask   = torch.zeros(T, _N_GRP, device=device)
    gmask.scatter_(1, gidx, 1.0)
    score_mask = gmask.unsqueeze(2).expand(T, _N_GRP, g_sz).reshape(T, _E_GLB)

    neg_inf = torch.finfo(torch.float32).min
    pruned  = swb.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = pruned.topk(_TOP_K, dim=1, largest=True, sorted=False)  # [T, 8]

    M    = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    w    = s * M
    wsum = w.sum(dim=1, keepdim=True) + 1e-20
    weights = (w / wsum) * routed_scaling_factor                     # [T, 256]

    # ── Per-expert compute (reference loop, vectorised dispatch) ───────────────
    out_f32    = torch.zeros(T, _H, dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(_E_LOC):
        ge = local_start + le
        if ge < 0 or ge >= _E_GLB:
            continue

        # Which tokens selected this expert?
        sel = (topk_idx == ge).any(dim=1)     # [T] bool  (GPU→CPU sync via .any())
        if not sel.any():
            continue

        tok_idx = sel.nonzero(as_tuple=False).squeeze(1)   # [Tk]

        A_e   = A.index_select(0, tok_idx)                 # [Tk, H]
        W13_e = W13[le]                                     # [G1, H]
        W2_e  = W2[le]                                      # [H, I]

        G1_out = A_e.matmul(W13_e.t())                     # [Tk, G1]

        X1 = G1_out[:, :_I]
        X2 = G1_out[:, _I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))              # silu(X2)
        C = silu_X2 * X1                                    # [Tk, I]

        O = C.matmul(W2_e.t())                              # [Tk, H]

        w_tok = weights.index_select(0, tok_idx)[:, ge]    # [Tk]
        out_f32.index_add_(0, tok_idx, O * w_tok.unsqueeze(1))

    output.copy_(out_f32.to(torch.bfloat16))
