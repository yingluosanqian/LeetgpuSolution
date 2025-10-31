import torch
import triton
import triton.language as tl


@triton.jit
def invert_kernel(
    image,
    width, height,
    BLOCK_SIZE: tl.constexpr
):
    image = image.to(tl.pointer_type(tl.uint32))
    pid_m = tl.program_id(0)
    crd = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = crd < width * height
    data = tl.load(image + crd, mask=mask, other=0)
    data = data ^ 0x00ffffff
    tl.store(image + crd, data, mask=mask)


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](
        image,
        width, height,
        BLOCK_SIZE
    )
