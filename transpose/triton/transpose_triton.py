import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(
    input, output,
    rows, cols,
    stride_ir, stride_ic,
    stride_or, stride_oc,
    CTA_M: tl.constexpr,
    CTA_N: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    m_crd = pid_m * CTA_M + tl.arange(0, CTA_M)
    n_crd = pid_n * CTA_N + tl.arange(0, CTA_N)

    src_crd = input + m_crd[:, None] * stride_ir + n_crd[None, :] * stride_ic
    src_mask = (m_crd[:, None] < rows) & (n_crd[None, :] < cols)
    data = tl.load(src_crd, mask=src_mask, other=0.0)
    data = tl.trans(data)

    dst_crd = output + n_crd[:, None] * stride_or + m_crd[None, :] * stride_oc
    dst_mask = (n_crd[:, None] < cols) & (m_crd[None, :] < rows)
    tl.store(dst_crd, data, mask=dst_mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    CTA_M = 128
    CTA_N = 128
    grid = ((rows + CTA_M - 1) // CTA_M, (cols + CTA_N - 1) // CTA_N)
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc,
        CTA_M, CTA_N
    )
