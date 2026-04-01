"""
FP8 Block-Scale Fused MoE Kernel  –  v6c
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

v5  (reference-exact):  all 19 PASSED, abs_err=0, ~1.0x speedup
v6  (fp8 in Triton):    RUNTIME_ERROR – float8 Triton broken on B200
v6b (fp32 Triton 162x): RemoteError – compile OOM/timeout, expand().reshape() bug

v6c:
  - No Triton, no torch.compile (avoids all compilation issues)
  - Lazy per-expert FP8 dequant: only active experts expanded (~177 MB each)
  - A_scale expanded with repeat_interleave (contiguous, no reshape bug)
  - SwiGLU via slice views – no intermediate buffer write
  - Routing identical to v5 (verified correct)

Performance vs v5
-----------------
  v5: expands all 32 experts upfront (5.6 GB peak, 3-4 ms of HBM traffic)
  v6c: per-expert lazy expand, skips empty experts entirely
  Slice views for SwiGLU split: OS-free, avoids a separate elementwise kernel
"""

import torch
import torch.nn.functional as F

_H     = 7168
_I     = 2048
_G1    = 4096   # 2 * _I
_E_LOC = 32
_E_GLB = 256
_TOP_K = 8
_N_GRP = 8
_TK_GRP= 4
_BLKSZ = 128


def _dequant_2d(fp8_t, scale_t):
    """FP8 [R, C] + block-scales [R//128, C//128] → fp32 [R, C]."""
    fp32 = fp8_t.to(torch.float32)
    s    = (scale_t.to(torch.float32)
                   .repeat_interleave(_BLKSZ, dim=0)
                   .repeat_interleave(_BLKSZ, dim=1))
    return fp32 * s


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
    return topk_idx, w / wsum   # [T, 8], [T, 256]


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
    weights_scaled = weights_norm * routed_scaling_factor   # [T, 256]

    # ── Pre-dequant hidden states (T×H – relatively small) ───────────────────
    A_scale = (hidden_states_scale.to(torch.float32)
                                  .permute(1, 0)                     # [T, 56]
                                  .contiguous()
                                  .repeat_interleave(_BLKSZ, dim=1)) # [T, H]
    A = hidden_states.to(torch.float32) * A_scale   # [T, H] fp32

    # ── Output accumulator ────────────────────────────────────────────────────
    out_f32     = torch.zeros(T, _H, dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(_E_LOC):
        ge = local_start + le
        if ge < 0 or ge >= _E_GLB:
            continue

        sel = (topk_idx == ge).any(dim=1)
        if not sel.any():
            continue

        tok_idx = sel.nonzero(as_tuple=False).squeeze(1)   # [Tk]
        A_e     = A[tok_idx]                               # [Tk, H] fp32

        # ── GEMM1: lazy per-expert dequant + cuBLAS ───────────────────────────
        W13_e   = _dequant_2d(gemm1_weights[le], gemm1_weights_scale[le])   # [G1, H]
        G1_out  = torch.mm(A_e, W13_e.t())    # [Tk, G1]

        # SwiGLU (slice views – no HBM copy for the split)
        gate = G1_out[:, :_I]                  # [Tk, I] view
        up   = G1_out[:, _I:]                  # [Tk, I] view
        C    = F.silu(up) * gate               # [Tk, I]

        # ── GEMM2: lazy per-expert dequant + cuBLAS ───────────────────────────
        W2_e = _dequant_2d(gemm2_weights[le], gemm2_weights_scale[le])      # [H, I]
        O    = torch.mm(C, W2_e.t())           # [Tk, H]

        # Routing weight scale + scatter-add
        w_tok = weights_scaled[tok_idx, ge]
        out_f32.index_add_(0, tok_idx, O * w_tok.unsqueeze(1))

    output.copy_(out_f32.to(torch.bfloat16))
