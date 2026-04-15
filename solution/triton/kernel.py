"""
FP8 Block-Scale Fused MoE Kernel  –  v28
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

v22: geomean ~1.69x
v25: batch expert selection (1 GPU-CPU sync) → geomean ~1.97x  ← best
v26/v26b: scatter → ~1.88x (regression, reverted)
v27: compiled _batch_select → ~1.92x (dynamic=True overhead, reverted)
v28: module-level _LE_RANGE to avoid torch.arange allocation per call
  - Precompute [0..31] arange once, reuse via .to(device) (no-op if on GPU)
  - Inline batch selection (no compile overhead)

Key techniques (all from v17/v18c):
1. Expert skipping: one batched broadcast finds all active experts, 1 sync
2. view+broadcast per-call dequant (no cache → no ABA)
3. torch.compile dequant: fp8→fp32 + scale mul fused
4. torch.compile SwiGLU with dynamic=True: strided loads handle non-contiguous
5. cuBLAS TF32 GEMMs (tensor cores; Triton WGMMA crashes on B200 Modal)
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

# Precomputed local expert index range (reused across calls, moved to GPU lazily)
_LE_RANGE_CPU = torch.arange(_E_LOC)   # [0..31] on CPU
_LE_RANGE_GPU: torch.Tensor | None = None

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
def _prep_A_scale(scale: torch.Tensor) -> torch.Tensor:
    """[56, T] bf16/f32 → [T, 56] f32. Fuse cast + transpose + contiguous."""
    return scale.to(torch.float32).permute(1, 0).contiguous()


@torch.compile(fullgraph=True, dynamic=True)
def _finalize(out_f32: torch.Tensor, output: torch.Tensor) -> None:
    """Fuse fp32→bf16 cast + copy into 1 kernel. Avoids intermediate bf16 tensor."""
    output.copy_(out_f32.to(torch.bfloat16))


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

    # A_scale: [56, T] → [T, 56] (fused cast + transpose + contiguous)
    A_scale = _prep_A_scale(hidden_states_scale)   # [T, 56]

    out_f32     = torch.zeros(T, _H, dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    # ── Batch expert selection: one broadcast pass → 1 GPU-CPU sync ──────────
    global _LE_RANGE_GPU
    if _LE_RANGE_GPU is None or _LE_RANGE_GPU.device != device:
        _LE_RANGE_GPU = _LE_RANGE_CPU.to(device)
    ge_range  = _LE_RANGE_GPU + local_start                       # [E_LOC]
    valid     = (ge_range >= 0) & (ge_range < _E_GLB)            # [E_LOC]
    all_match = (topk_idx.unsqueeze(0) == ge_range.view(_E_LOC, 1, 1))  # [E, T, 8]
    all_sel   = all_match.any(dim=2) & valid.unsqueeze(1)         # [E_LOC, T]
    active    = all_sel.any(dim=1)                                # [E_LOC]

    # One sync: get list of active local expert indices
    for le in active.nonzero(as_tuple=False).squeeze(1).tolist():
        ge      = local_start + le
        tok_idx = all_sel[le].nonzero(as_tuple=False).squeeze(1)

        A_fp32   = _dequant_A(hidden_states[tok_idx], A_scale[tok_idx])
        W13_fp32 = _dequant_W13(gemm1_weights[le], gemm1_weights_scale[le])
        C_full   = torch.mm(A_fp32, W13_fp32.t())     # GEMM1: cuBLAS TF32
        C        = _swiglu(C_full)
        W2_fp32  = _dequant_W2(gemm2_weights[le], gemm2_weights_scale[le])
        O        = torch.mm(C, W2_fp32.t())            # GEMM2: cuBLAS TF32

        w_tok = weights_scaled[tok_idx, ge]
        out_f32.index_add_(0, tok_idx, O * w_tok.unsqueeze(1))

    _finalize(out_f32, output)
