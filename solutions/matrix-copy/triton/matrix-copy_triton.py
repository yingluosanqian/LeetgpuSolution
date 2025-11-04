# 0.1222ms
import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(
    input, output, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    stride_m, stride_n = n_elements, 1
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    crd_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    crd_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    crd = crd_m[:, None] * stride_m + crd_n[None, :] * stride_n
    mask = (crd_m[:, None] < n_elements) & (crd_n[None, :] < n_elements)
    data = tl.load(input + crd, mask=mask)
    tl.store(output + crd, data, mask=mask)


# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE),)
    copy_kernel[grid](
        a, b, N,
        BLOCK_SIZE
    )
