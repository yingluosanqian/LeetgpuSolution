# 0.2812ms
import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    input,
    output,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    alpha: tl.constexpr = 0.01
    pid = tl.program_id(0)
    crd = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = crd < n_elements
    data = tl.load(input + crd, mask=mask)
    data = tl.where(data > 0, data, alpha * data)
    tl.store(output + crd, data, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](
        input,
        output,
        N,
        BLOCK_SIZE
    )
