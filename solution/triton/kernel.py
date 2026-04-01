"""
FP8 Block-Scale Fused MoE Kernel  –  v6b
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

v5 (reference-exact): all 19 PASSED, abs_err=0, ~1.0x speedup
v6  (fp8 in Triton) : RUNTIME_ERROR – float8_e4m3fn not loadable in Triton on B200

v6b fixes:
  - Dequant FP8 → fp32 in Python (per-expert, lazily – avoids v5's 5.6GB upfront expansion)
  - Triton handles fp32 inputs only (no dtype issues)
  - GEMM1 + SwiGLU fused in Triton (eliminates [Tk, G1] intermediate buffer)
  - GEMM2 via cuBLAS (torch.mm) with per-expert lazy dequant
  - Routing identical to v5

Wins vs v5
----------
  1. Lazy weight dequant: only active experts expanded (~180MB/expert vs 5.6GB upfront)
  2. SwiGLU fused: [Tk, G1] write+read eliminated (~2*Tk*4096*4 bytes/expert)
  3. Triton GEMM1 autotuned for B200 tile shapes
"""

import torch
import triton
import triton.language as tl

_H     = 7168
_I     = 2048
_G1    = 4096   # = 2 * _I
_E_LOC = 32
_E_GLB = 256
_TOP_K = 8
_N_GRP = 8
_TK_GRP= 4
_BLKSZ = 128


# ─────────────────────────────────────────────────────────────────────────────
# GEMM1 + SwiGLU fused (fp32 inputs)
#   A : [M, K]   fp32   (K = H = 7168)
#   W : [2N, K]  fp32   (N = I = 2048, rows 0..N-1 = gate, rows N..2N-1 = up)
#   C : [M, N]   fp32   result = silu(up) * gate
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                      num_warps=nw, num_stages=ns)
        for bm in [32, 64, 128]
        for bn in [32, 64, 128]
        for bk in [32, 64, 128]
        for nw in [4, 8]
        for ns in [2, 3, 4]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm1_swiglu(
    A_ptr, W_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_gate = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # gate cols [0..N)
    n_up   = n_gate + N                                  # up cols   [N..2N)

    acc_g = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_u = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_tile = tl.load(a_ptrs,
                         mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                         other=0.0)

        # W_gate transposed tile [BLOCK_K, BLOCK_N]  – load B in column-major order
        wg_ptrs = W_ptr + n_gate[None, :] * stride_wn + k_offs[:, None] * stride_wk
        wg_tile = tl.load(wg_ptrs,
                          mask=(n_gate[None, :] < N) & (k_offs[:, None] < K),
                          other=0.0)

        # W_up transposed tile [BLOCK_K, BLOCK_N]
        wu_ptrs = W_ptr + n_up[None, :] * stride_wn + k_offs[:, None] * stride_wk
        wu_tile = tl.load(wu_ptrs,
                          mask=(n_up[None, :] < 2 * N) & (k_offs[:, None] < K),
                          other=0.0)

        acc_g += tl.dot(a_tile, wg_tile, allow_tf32=False)
        acc_u += tl.dot(a_tile, wu_tile, allow_tf32=False)

    # SwiGLU: silu(up) * gate
    result = (acc_u * tl.sigmoid(acc_u)) * acc_g

    c_ptrs = C_ptr + m_offs[:, None] * stride_cm + n_gate[None, :] * stride_cn
    tl.store(c_ptrs, result,
             mask=(m_offs[:, None] < M) & (n_gate[None, :] < N))


# ─────────────────────────────────────────────────────────────────────────────
# Routing – vectorised, identical to v5 reference
# ─────────────────────────────────────────────────────────────────────────────
def _route(routing_logits, routing_bias, device, T):
    s   = torch.sigmoid(routing_logits.to(torch.float32))
    swb = s + routing_bias.to(torch.float32)

    grouped = swb.view(T, _N_GRP, _E_GLB // _N_GRP)
    top2, _ = grouped.topk(2, dim=2, largest=True, sorted=False)
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


def _dequant(fp8_tensor, scale, blksz=128):
    """Expand fp8 + block scales → fp32. Works for 2-D tensors."""
    fp32 = fp8_tensor.to(torch.float32)
    s_exp = (scale.to(torch.float32)
                  .repeat_interleave(blksz, dim=0)
                  .repeat_interleave(blksz, dim=1))
    return fp32 * s_exp


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

    # ── Pre-dequant hidden states (small – [T, H]) ────────────────────────────
    A_scale_TH = hidden_states_scale.to(torch.float32).permute(1, 0).contiguous()  # [T, 56]
    A_scale_exp = (A_scale_TH.unsqueeze(-1)
                              .expand(T, _H // _BLKSZ, _BLKSZ)
                              .reshape(T, _H))
    A = hidden_states.to(torch.float32) * A_scale_exp   # [T, H] fp32

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
        Tk      = tok_idx.shape[0]

        A_e = A[tok_idx].contiguous()   # [Tk, H] fp32

        # ── Lazy dequant W13 for this expert only ─────────────────────────────
        # W13[le]: [G1, H] fp8 → fp32,  S13[le]: [32, 56] → [G1, H]
        W13_e = _dequant(gemm1_weights[le], gemm1_weights_scale[le]).contiguous()
        # W13_e: [G1, H] = [4096, 7168] fp32

        # ── GEMM1 + SwiGLU via Triton ─────────────────────────────────────────
        C = torch.empty(Tk, _I, dtype=torch.float32, device=device)

        grid = lambda meta: (
            triton.cdiv(Tk, meta["BLOCK_M"]),
            triton.cdiv(_I, meta["BLOCK_N"]),
        )

        _gemm1_swiglu[grid](
            A_e, W13_e, C,
            Tk, _I, _H,
            A_e.stride(0),   A_e.stride(1),
            W13_e.stride(0), W13_e.stride(1),
            C.stride(0),     C.stride(1),
        )

        # ── GEMM2: lazy dequant + cuBLAS ─────────────────────────────────────
        # W2[le]: [H, I] fp8 → fp32,  S2[le]: [56, 16] → [H, I]
        W2_e = _dequant(gemm2_weights[le], gemm2_weights_scale[le]).contiguous()
        # W2_e: [H, I] = [7168, 2048] fp32

        O = torch.mm(C, W2_e.t())   # [Tk, I] @ [I, H] → [Tk, H]

        w_tok = weights_scaled[tok_idx, ge]                         # [Tk]
        out_f32.index_add_(0, tok_idx, O * w_tok.unsqueeze(1))

    output.copy_(out_f32.to(torch.bfloat16))
