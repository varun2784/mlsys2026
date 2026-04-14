"""
Helion fused GEMM1 + SwiGLU kernel for MoE.

Uses dot_precision='ieee' (FFMA) to avoid WGMMA crash on B200 Modal Triton.
Fuses: A @ W13.T → [Tk, G1], then SwiGLU → [Tk, I]
Saves one HBM round-trip vs separate GEMM1 + SwiGLU passes.
"""

import helion
import helion.language as hl
import torch


@helion.kernel(dot_precision='ieee')
def gemm1_swiglu_fused(
    A: torch.Tensor,   # [M, K]   fp32  (dequanted hidden states)
    W: torch.Tensor,   # [2N, K]  fp32  (W13: gate rows 0..N-1, up rows N..2N-1)
    C: torch.Tensor,   # [M, N]   fp32  output (after SwiGLU)
) -> torch.Tensor:
    M, K = A.shape
    twoN, _ = W.shape
    N = twoN // 2
    for tile_m, tile_n in hl.tile([M, N]):
        acc_gate = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc_up   = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            a_t  = A[tile_m, tile_k]
            wg_t = W[tile_n, tile_k]          # gate rows
            wu_t = W[tile_n + N, tile_k]      # up rows (offset by N)
            acc_gate = hl.dot(a_t, wg_t, acc_gate)
            acc_up   = hl.dot(a_t, wu_t, acc_up)
        result = (acc_up * torch.sigmoid(acc_up)) * acc_gate
        C[tile_m, tile_n] = result
    return C
