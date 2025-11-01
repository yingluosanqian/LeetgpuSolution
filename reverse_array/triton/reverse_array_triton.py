import torch
import triton
import triton.language as tl


@triton.jit
def reverse_kernel(
    input,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    crd = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_mask = crd < N // 2
    rev_crd = N - 1 - crd
    out_mask = rev_crd >= (N + 1) // 2

    lhs = tl.load(input + crd, mask=in_mask)
    rhs = tl.load(input + rev_crd, mask=out_mask)
    tl.store(input + crd, rhs, mask=in_mask)
    tl.store(input + rev_crd, lhs, mask=out_mask)


# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)

    reverse_kernel[grid](
        input,
        N,
        BLOCK_SIZE
    )
