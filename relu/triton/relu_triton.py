# time = 0.1598ms
import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    crd = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = crd < n_elements
    data = tl.load(input + crd, mask=mask)
    data = max(data, 0.0)
    tl.store(output + crd, data, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)
