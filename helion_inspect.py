import helion
import helion.language as hl
import torch

@helion.jit
def simple_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty(m, n, dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc)
        out[tile_m, tile_n] = acc
    return out

A = torch.randn(64, 128, dtype=torch.float32)
B = torch.randn(128, 64, dtype=torch.float32)
bound = simple_mm.bind((A, B))

cfg = helion.Config(block_sizes=[32, 32, 32], num_warps=4, num_stages=3)
# to_triton_code is the right method
src = bound.to_triton_code(cfg)
print("=== Generated Triton Code ===")
print(src)
