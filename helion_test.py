"""
Test Helion GEMM1+SwiGLU kernel codegen.
Checks if we can force 'ieee' precision (FFMA) to avoid WGMMA crash on B200.
"""
import helion
import helion.language as hl
import torch


@helion.jit
def gemm1_swiglu(
    A: torch.Tensor,    # [M, K] fp32
    W: torch.Tensor,    # [2N, K] fp32  (gate=rows 0..N-1, up=rows N..2N-1)
    C: torch.Tensor,    # [M, N] fp32 output
) -> torch.Tensor:
    M, K = A.shape
    twoN, _ = W.shape
    N = twoN // 2
    for tile_m, tile_n in hl.tile([M, N]):
        acc_gate = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc_up   = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            a_t  = A[tile_m, tile_k]
            wg_t = W[tile_n, tile_k]           # gate rows
            wu_t = W[tile_n + N, tile_k]       # up rows
            acc_gate = hl.dot(a_t, wg_t, acc_gate)
            acc_up   = hl.dot(a_t, wu_t, acc_up)
        result = (acc_up * torch.sigmoid(acc_up)) * acc_gate
        C[tile_m, tile_n] = result
    return C


if __name__ == "__main__":
    M, K, N = 64, 7168, 2048

    A = torch.randn(M, K, dtype=torch.float32)
    W = torch.randn(2 * N, K, dtype=torch.float32)
    C = torch.empty(M, N, dtype=torch.float32)

    bound = gemm1_swiglu.bind((A, W, C))
    print("Config spec:", bound.config_spec)

    # Check what precision Helion uses by default
    cfg_tf32 = helion.Config(block_sizes=[64, 128, 128], num_warps=8, num_stages=4)
    src = bound.to_triton_code(cfg_tf32)

    # Check precision setting in generated code
    import re
    prec_hits = re.findall(r"input_precision='[^']*'", src)
    print("Precision settings in generated code:", prec_hits)

    # Try indexing — the gate offset (tile_n + N) is the key issue
    print("\n=== First 80 lines of generated Triton ===")
    for i, line in enumerate(src.split('\n')[:80]):
        print(f"{i+1:3}: {line}")
