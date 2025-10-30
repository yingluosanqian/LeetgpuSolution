import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    p_id = tl.program_id(0)
    offset = p_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_val = tl.load(a_ptr + offset, mask=mask, other=0.0)
    b_val = tl.load(b_ptr + offset, mask=mask, other=0.0)
    tl.store(c_ptr + offset, a_val + b_val, mask=mask)
   
# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
