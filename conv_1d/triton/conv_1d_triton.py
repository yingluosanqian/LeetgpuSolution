import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    input, kernel, output,
    input_size, kernel_size,
    BLOCK_SIZE: tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    kernel = kernel.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    output_size = input_size - kernel_size + 1
    crd = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = crd < output_size
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for k in tl.range(0, kernel_size):
        kernel_val = tl.load(kernel + k)
        input_val = tl.load(input + crd + k, mask=mask, other=0.0)
        acc += input_val * kernel_val

    tl.store(output + crd, acc, mask=mask)


# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE),)

    conv1d_kernel[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )
