import pytest
import torch
import utils
from sgl_kernel import hc_pre_gemm_sqrsum

device = utils.get_device()


# ===== HC PRE GEMM SQRSUM TESTS =====


@pytest.mark.parametrize("T", [16, 128, 512])
@pytest.mark.parametrize("K", [4096, 16384])
@pytest.mark.parametrize("N", [24, 32])
def test_hc_pre_gemm_sqrsum_correctness(T, K, N):
    """Test HC Pre GEMM with SqrSum against PyTorch reference"""

    # Create inputs
    A = torch.randn(T, K, dtype=torch.bfloat16, device=f"{device}:0")
    B_colmajor = torch.randn(N, K, dtype=torch.float32, device=f"{device}:0")
    B = B_colmajor.t()  # [K, N] with column-major stride
    C = torch.empty(T, N, dtype=torch.float32, device=f"{device}:0")
    sqrsum = torch.empty(T, dtype=torch.float32, device=f"{device}:0")

    # Run kernel
    hc_pre_gemm_sqrsum(A, B, C, sqrsum)

    # Reference
    A_fp32 = A.to(torch.float32)
    C_ref = torch.matmul(A_fp32, B)
    sqrsum_ref = (C_ref**2).sum(dim=1)

    # Compare C
    diff_C = (C - C_ref).abs()
    max_diff_C = diff_C.max().item()
    print(f"\n[SqrSum] Shape: [{T}, {K}] @ [{K}, {N}]")
    print(f"C max diff: {max_diff_C:.6f}")

    torch.testing.assert_close(
        C,
        C_ref,
        rtol=2e-2,
        atol=1.5,
        msg=f"GEMM+SqrSum: C mismatch - max_diff={max_diff_C:.6f}",
    )

    # Compare sqrsum
    diff_sqrsum = (sqrsum - sqrsum_ref).abs()
    max_diff_sqrsum = diff_sqrsum.max().item()
    print(f"sqrsum max diff: {max_diff_sqrsum:.6f}")

    torch.testing.assert_close(
        sqrsum,
        sqrsum_ref,
        rtol=2e-2,
        atol=5.0,  # Relaxed: bf16 + sum of squares accumulates more error
        msg=f"GEMM+SqrSum: sqrsum mismatch - max_diff={max_diff_sqrsum:.6f}",
    )
    print(f"✓ SqrSum test passed for T={T}, K={K}, N={N}")


def test_hc_pre_gemm_sqrsum_sqrsum_dtype():
    """Test sqrsum dtype check"""
    T, K, N = 16, 1024, 24

    with pytest.raises(RuntimeError, match="sqrsum must be float32"):
        A = torch.randn(T, K, dtype=torch.bfloat16, device=f"{device}:0")
        B = torch.randn(K, N, dtype=torch.float32, device=f"{device}:0")
        C = torch.empty(T, N, dtype=torch.float32, device=f"{device}:0")
        sqrsum = torch.empty(T, dtype=torch.bfloat16, device=f"{device}:0")  # Wrong!
        hc_pre_gemm_sqrsum(A, B, C, sqrsum)

    print("✓ SqrSum dtype check passed")


def test_hc_pre_gemm_sqrsum_dimension():
    """Test sqrsum dimension check"""

    with pytest.raises(RuntimeError, match="sqrsum dimension mismatch"):
        A = torch.randn(16, 1024, dtype=torch.bfloat16, device=f"{device}:0")
        B = torch.randn(1024, 24, dtype=torch.float32, device=f"{device}:0")
        C = torch.empty(16, 24, dtype=torch.float32, device=f"{device}:0")
        sqrsum = torch.empty(32, dtype=torch.float32, device=f"{device}:0")  # Wrong T
        hc_pre_gemm_sqrsum(A, B, C, sqrsum)

    print("✓ SqrSum dimension check passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
