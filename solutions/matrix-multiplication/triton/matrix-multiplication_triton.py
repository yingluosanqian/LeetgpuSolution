# 23.0866ms
import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    CTA_M: tl.constexpr,
    CTA_N: tl.constexpr,
    CTA_K: tl.constexpr,
):
    pid_m, pid_k = tl.program_id(0), tl.program_id(1)
    m_crd = pid_m * CTA_M + tl.arange(0, CTA_M)
    n_crd = tl.arange(0, CTA_N)
    k_crd = pid_k * CTA_K + tl.arange(0, CTA_K)
    a_offset = m_crd[:, None] * stride_am + n_crd[None, :] * stride_an
    b_offset = n_crd[:, None] * stride_bn + k_crd[None, :] * stride_bk
    acc = tl.zeros((CTA_M, CTA_K), dtype=tl.float32)
    for n_tile in range(tl.cdiv(N, CTA_N)):
        a_mask = (m_crd[:, None] < M) & (n_tile * CTA_N + n_crd[None, :] < N)
        b_mask = (n_tile * CTA_N + n_crd[:, None] < N) & (k_crd[None, :] < K)
        a_val = tl.load(a + a_offset, mask=a_mask, other=0.0)
        b_val = tl.load(b + b_offset, mask=b_mask, other=0.0)
        acc += tl.dot(a_val, b_val, allow_tf32=False)
        a_offset += CTA_N * stride_an
        b_offset += CTA_N * stride_bn
    c_offset = m_crd[:, None] * stride_cm + k_crd[None, :] * stride_ck
    c_mask = (m_crd[:, None] < M) & (k_crd[None, :] < K)
    tl.store(c + c_offset, acc, mask=c_mask)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    CTA_M = 128
    CTA_N = 16
    CTA_K = 64
    grid = ((M + CTA_M - 1) // CTA_M, (K + CTA_K - 1) // CTA_K)
    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        CTA_M, CTA_N, CTA_K
    )
