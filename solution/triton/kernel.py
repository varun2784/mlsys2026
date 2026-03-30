"""
FP8 Block-Scale Fused MoE Kernel  –  v4
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Design philosophy for this version
====================================
Get correctness locked before micro-optimising.

Pipeline
--------
1.  Routing     (vectorised PyTorch, no per-expert GPU→CPU sync)
2.  Dequant     (PyTorch: hidden_states → bf16 A; W13, W2 → bf16 upfront)
3.  Dispatch    (vectorised sort by expert; one CPU sync for counts)
4.  GEMM1       (Triton batched: bf16 × bf16 → f32, sorted-token layout)
5.  SwiGLU      (inline: silu(X2) * X1)
6.  GEMM2       (Triton batched: bf16 × bf16 → f32, weighted atomic-add)

Key correctness fixes vs v1/v2/v3
-----------------------------------
• Routing over all 256 global experts (not just 32 local)
• Group size = 32 (256/8), not 4
• SwiGLU: silu(second_half) * first_half  (matches reference)
• Weights normalised from s (no bias); selection uses s+bias
• Per-tile valid count (tvc) replaces broken global num_valid mask
• No FP8 loads inside Triton  (avoids fp8-pointer issues in some Triton builds)

Performance vs reference
-------------------------
• bf16 matmul (~2× vs reference fp32)
• No per-expert GPU→CPU sync (32 eliminated, only 1 upfront)
• Batched dispatch with single gather  (vs index_select per expert)
• Triton batched GEMM (all experts in parallel, one kernel launch each)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ── Problem constants ──────────────────────────────────────────────────────────
_H      = 7168   # hidden_size
_I      = 2048   # intermediate_size
_G1     = 4096   # 2*I (gate+up fused)
_E_LOC  = 32     # local experts per rank
_E_GLB  = 256    # global experts
_TOP_K  = 8
_N_GRP  = 8      # routing groups
_TK_GRP = 4      # top-k groups selected
_BLKSZ  = 128    # FP8 quantisation block size

# Tile sizes for Triton GEMMs
_BM = 64
_BN = 128
_BK = 128


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FP8 dequantisation (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

def _dequant_A(x, scale):
    """fp8 [T,H] + scale[56,T]  →  bf16 [T,H]"""
    T = x.shape[0]
    sc = scale.float().T.contiguous()               # [T, 56]
    sc = sc.unsqueeze(-1).expand(T, 56, _BLKSZ).reshape(T, _H)
    return x.float().mul_(sc).to(torch.bfloat16)    # [T, H]


def _dequant_W13(w, scale):
    """fp8 [E,G1,H] + scale[E,32,56]  →  bf16 [E,G1,H]"""
    sc = scale.float()
    sc = sc.repeat_interleave(_BLKSZ, dim=1).repeat_interleave(_BLKSZ, dim=2)  # [E,G1,H]
    return w.float().mul_(sc).to(torch.bfloat16)


def _dequant_W2(w, scale):
    """fp8 [E,H,I] + scale[E,56,16]  →  bf16 [E,H,I]"""
    sc = scale.float()
    sc = sc.repeat_interleave(_BLKSZ, dim=1).repeat_interleave(_BLKSZ, dim=2)  # [E,H,I]
    return w.float().mul_(sc).to(torch.bfloat16)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Routing  (vectorised, matches reference exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _routing(logits, bias, rsf, loffset, device):
    """
    Returns
        topk_idx  int64  [T, TOP_K]   global expert IDs
        weights   f32    [T, E_GLB]   sparse weight matrix
    """
    T    = logits.shape[0]
    logf = logits.float()
    biasf = bias.float()

    s   = torch.sigmoid(logf)          # [T, 256]
    swb = s + biasf                    # [T, 256]

    g_sz    = _E_GLB // _N_GRP        # 32 experts per group
    grouped = swb.view(T, _N_GRP, g_sz)
    top2, _ = grouped.topk(2, dim=2)
    gscores = top2.sum(dim=2)         # [T, 8]

    _, gidx = gscores.topk(_TK_GRP, dim=1)
    gmask   = torch.zeros(T, _N_GRP, device=device)
    gmask.scatter_(1, gidx, 1.0)
    emask   = gmask.unsqueeze(2).expand(T, _N_GRP, g_sz).reshape(T, _E_GLB)

    neg_inf     = torch.finfo(torch.float32).min
    pruned      = swb.masked_fill(emask == 0, neg_inf)
    _, topk_idx = pruned.topk(_TOP_K, dim=1)   # [T, TOP_K]

    # Weights: s (no bias), normalised
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    w    = s * M
    wsum = w.sum(dim=1, keepdim=True) + 1e-20
    weights = (w / wsum) * rsf                 # [T, 256]

    return topk_idx, weights


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Dispatch  (vectorised, one GPU→CPU sync for counts)
# ══════════════════════════════════════════════════════════════════════════════

def _dispatch(topk_idx, weights, loffset, blk_m, device):
    """
    Returns
        s_tok   int32  [total_padded]   original token indices (padded with last valid)
        s_wgt   f32    [total_padded]   routing weight (0.0 for padding slots)
        tile_e  int32  [num_tiles]      local expert per Triton tile
        tvc     int32  [num_tiles]      valid-token count per tile
    """
    T  = topk_idx.shape[0]
    lo = loffset
    hi = lo + _E_LOC

    tok_rep  = torch.arange(T, device=device, dtype=torch.int64).repeat_interleave(_TOP_K)
    ge_flat  = topk_idx.reshape(-1)
    loc_mask = (ge_flat >= lo) & (ge_flat < hi)

    ge_loc  = ge_flat[loc_mask]
    tok_loc = tok_rep[loc_mask]
    le_loc  = (ge_loc - lo).to(torch.int32)
    wgt_loc = weights[tok_loc, ge_loc].float()

    if tok_loc.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return empty, empty.float(), empty, empty

    order    = torch.argsort(le_loc.long(), stable=True)
    s_le     = le_loc[order]
    s_tok    = tok_loc[order].to(torch.int32)
    s_wgt    = wgt_loc[order]

    counts     = torch.bincount(s_le.long(), minlength=_E_LOC)    # [E_LOC]
    counts_cpu = counts.cpu().tolist()                             # one sync

    padded = [((c + blk_m - 1) // blk_m) * blk_m for c in counts_cpu]
    total_p = sum(padded)
    n_tiles  = total_p // blk_m

    out_tok  = torch.zeros(total_p, dtype=torch.int32,   device=device)
    out_wgt  = torch.zeros(total_p, dtype=torch.float32, device=device)
    tile_e   = torch.zeros(n_tiles, dtype=torch.int32,   device=device)
    tvc      = torch.zeros(n_tiles, dtype=torch.int32,   device=device)

    src = dst = tile_off = 0
    for e in range(_E_LOC):              # 32 iterations, GPU-async
        c = counts_cpu[e]
        p = padded[e]
        if p == 0:
            src += c
            continue
        out_tok[dst : dst + c] = s_tok[src : src + c]
        out_wgt[dst : dst + c] = s_wgt[src : src + c]
        if c < p:
            fill = s_tok[src + c - 1] if c > 0 else torch.zeros(1, dtype=torch.int32, device=device)
            out_tok[dst + c : dst + p] = fill
        n  = p // blk_m
        tile_e[tile_off : tile_off + n] = e
        rem = c
        for ti in range(n):                  # ≤ ceil(c / blk_m) ≤ 64 iters
            tvc[tile_off + ti] = min(rem, blk_m)
            rem = max(0, rem - blk_m)
        src      += c
        dst      += p
        tile_off += n

    return out_tok, out_wgt, tile_e, tvc


# ══════════════════════════════════════════════════════════════════════════════
# 4.  GEMM1  (Triton, bf16 × bf16 → f32, sorted-token layout)
# ══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _gemm1_kernel(
    A_ptr,    # bf16  [T_orig, H]   — dequantised activations
    W_ptr,    # bf16  [E_loc, G1, H]
    tok_ptr,  # int32 [total_padded]
    exp_ptr,  # int32 [num_tiles]
    tvc_ptr,  # int32 [num_tiles]   per-tile valid count
    Out_ptr,  # f32   [total_padded, G1]
    sa_m, sa_k,
    sw_e, sw_n, sw_k,
    so_m, so_n,
    H:  tl.constexpr,
    G1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    eid        = tl.load(exp_ptr + pid_m)
    tile_valid = tl.load(tvc_ptr + pid_m)

    lane_m = tl.arange(0, BM)
    vm     = lane_m < tile_valid                    # per-tile validity mask

    orig   = tl.load(tok_ptr + pid_m * BM + lane_m, mask=vm, other=0)

    lane_n = pid_n * BN + tl.arange(0, BN)
    vn     = lane_n < G1

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for kb in tl.range(H // BK, num_stages=4):
        lane_k = kb * BK + tl.arange(0, BK)

        a = tl.load(A_ptr + orig[:, None] * sa_m + lane_k[None, :] * sa_k,
                    mask=vm[:, None], other=0.0)

        w = tl.load(W_ptr + eid * sw_e + lane_n[:, None] * sw_n + lane_k[None, :] * sw_k,
                    mask=vn[:, None], other=0.0)

        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float32)

    o_ptrs = Out_ptr + (pid_m * BM + lane_m)[:, None] * so_m + lane_n[None, :] * so_n
    tl.store(o_ptrs, acc, mask=vm[:, None] & vn[None, :])


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GEMM2  (Triton, bf16 × bf16 → f32, weighted atomic-add)
# ══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _gemm2_kernel(
    A_ptr,    # bf16  [total_padded, I]  — post-SwiGLU intermediate
    W_ptr,    # bf16  [E_loc, H, I]
    tok_ptr,  # int32 [total_padded]
    exp_ptr,  # int32 [num_tiles]
    wgt_ptr,  # f32   [total_padded]   routing weight (0 for padding)
    tvc_ptr,  # int32 [num_tiles]
    Out_ptr,  # f32   [T_orig, H]       zeroed before launch
    sa_m, sa_k,
    sw_e, sw_n, sw_k,
    so_m, so_n,
    I:  tl.constexpr,
    H:  tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    eid        = tl.load(exp_ptr + pid_m)
    tile_valid = tl.load(tvc_ptr + pid_m)

    lane_m = tl.arange(0, BM)
    vm     = lane_m < tile_valid

    orig = tl.load(tok_ptr + pid_m * BM + lane_m, mask=vm, other=0)
    rw   = tl.load(wgt_ptr + pid_m * BM + lane_m, mask=vm, other=0.0)

    lane_n = pid_n * BN + tl.arange(0, BN)
    vn     = lane_n < H

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for kb in tl.range(I // BK, num_stages=4):
        lane_k = kb * BK + tl.arange(0, BK)

        a = tl.load(A_ptr + (pid_m * BM + lane_m)[:, None] * sa_m + lane_k[None, :] * sa_k,
                    mask=vm[:, None], other=0.0)

        w = tl.load(W_ptr + eid * sw_e + lane_n[:, None] * sw_n + lane_k[None, :] * sw_k,
                    mask=vn[:, None], other=0.0)

        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float32)

    acc    = acc * rw[:, None]
    o_ptrs = Out_ptr + orig[:, None] * so_m + lane_n[None, :] * so_n
    tl.atomic_add(o_ptrs, acc, mask=vm[:, None] & vn[None, :])


# ══════════════════════════════════════════════════════════════════════════════
# Entry point (DPS)
# ══════════════════════════════════════════════════════════════════════════════

def kernel(
    routing_logits:        torch.Tensor,  # f32   [T, 256]
    routing_bias:          torch.Tensor,  # bf16  [256]
    hidden_states:         torch.Tensor,  # fp8   [T, 7168]
    hidden_states_scale:   torch.Tensor,  # f32   [56, T]
    gemm1_weights:         torch.Tensor,  # fp8   [32, 4096, 7168]
    gemm1_weights_scale:   torch.Tensor,  # f32   [32, 32, 56]
    gemm2_weights:         torch.Tensor,  # fp8   [32, 7168, 2048]
    gemm2_weights_scale:   torch.Tensor,  # f32   [32, 56, 16]
    local_expert_offset:   int,
    routed_scaling_factor: float,
    output:                torch.Tensor,  # bf16  [T, 7168]
):
    T   = hidden_states.shape[0]
    dev = hidden_states.device

    # ── 1. Dequantise (PyTorch, safe, no fp8 in Triton) ───────────────────────
    A    = _dequant_A(hidden_states, hidden_states_scale)       # bf16 [T, H]
    W13  = _dequant_W13(gemm1_weights, gemm1_weights_scale)     # bf16 [32, G1, H]
    W2   = _dequant_W2(gemm2_weights, gemm2_weights_scale)      # bf16 [32, H, I]

    # ── 2. Routing ─────────────────────────────────────────────────────────────
    topk_idx, weights = _routing(
        routing_logits, routing_bias,
        routed_scaling_factor, int(local_expert_offset), dev,
    )

    # ── 3. Dispatch ─────────────────────────────────────────────────────────────
    s_tok, s_wgt, tile_e, tvc = _dispatch(
        topk_idx, weights, int(local_expert_offset), _BM, dev,
    )

    if tile_e.shape[0] == 0:
        output.zero_()
        return

    total_p = s_tok.shape[0]
    n_tiles  = tile_e.shape[0]

    # ── 4. GEMM1: [total_p, H] @ [expert, G1, H].T → [total_p, G1] ────────────
    gate_up = torch.empty(total_p, _G1, dtype=torch.float32, device=dev)

    grid1 = (n_tiles, triton.cdiv(_G1, _BN))
    _gemm1_kernel[grid1](
        A,      W13,
        s_tok,  tile_e, tvc,
        gate_up,
        A.stride(0),    A.stride(1),
        W13.stride(0),  W13.stride(1),  W13.stride(2),
        gate_up.stride(0), gate_up.stride(1),
        H=_H, G1=_G1, BM=_BM, BN=_BN, BK=_BK,
        num_warps=4, num_stages=4,
    )

    # ── 5. SwiGLU: silu(X2) * X1  (X1=first half, X2=second half) ─────────────
    X1    = gate_up[:, :_I]
    X2    = gate_up[:, _I:]
    inter = F.silu(X2) * X1                                     # [total_p, I]
    inter_bf16 = inter.to(torch.bfloat16)

    # ── 6. GEMM2 + weighted reduce ───────────────────────────────────────────────
    out_f32 = torch.zeros(T, _H, dtype=torch.float32, device=dev)

    grid2 = (n_tiles, triton.cdiv(_H, _BN))
    _gemm2_kernel[grid2](
        inter_bf16,  W2,
        s_tok, tile_e, s_wgt, tvc,
        out_f32,
        inter_bf16.stride(0), inter_bf16.stride(1),
        W2.stride(0),  W2.stride(1),  W2.stride(2),
        out_f32.stride(0),    out_f32.stride(1),
        I=_I, H=_H, BM=_BM, BN=_BN, BK=_BK,
        num_warps=4, num_stages=4,
    )

    output.copy_(out_f32.to(torch.bfloat16))
