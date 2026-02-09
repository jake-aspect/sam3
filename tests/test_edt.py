"""Tests for the Euclidean Distance Transform (EDT) implementations.

Ground truth is cv2.distanceTransform (core dependency, always available).
"""

import numpy as np
import pytest
import torch

from sam3.model.edt import _HAS_CUPY, _HAS_TRITON, _edt_cv2, edt

_skip_no_cupy = pytest.mark.skipif(not _HAS_CUPY, reason="cupy not installed")
_skip_no_triton = pytest.mark.skipif(not _HAS_TRITON, reason="triton not available")
_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cv2_edt_batch(data: torch.Tensor) -> torch.Tensor:
    """
    Unoptimized ground truth implemention from OpenCV, the target.
    Note that this is equivalent to the current implementation of sam3.model.edt._edt_cv2.
    """
    import cv2

    B, H, W = data.shape
    result = torch.zeros(B, H, W, dtype=torch.float32)
    for b in range(B):
        src = data[b].numpy().astype(np.uint8)
        dt = cv2.distanceTransform(src, cv2.DIST_L2, 0)
        result[b] = torch.from_numpy(dt)
    return result


# ---------------------------------------------------------------------------
# 1. _edt_cv2 (primary CPU backend -- opencv is a core dependency)
# ---------------------------------------------------------------------------

class TestEdtCv2:
    """Validate the OpenCV-based CPU backend."""

    def test_all_zeros(self):
        """All-zero image => all distances are 0."""
        data = torch.zeros(1, 8, 8, dtype=torch.float32)
        result = _edt_cv2(data)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_all_ones(self):
        """All-foreground image => no background, distances are large."""
        data = torch.ones(1, 4, 4, dtype=torch.float32)
        result = _edt_cv2(data)
        assert (result > 0).all()

    def test_single_foreground_pixel(self):
        """One foreground pixel surrounded by background."""
        data = torch.zeros(1, 5, 5, dtype=torch.float32)
        data[0, 2, 2] = 1.0
        result = _edt_cv2(data)
        expected = _cv2_edt_batch(data)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_single_background_pixel(self):
        """All foreground except one background pixel."""
        data = torch.ones(1, 5, 5, dtype=torch.float32)
        data[0, 2, 2] = 0.0
        result = _edt_cv2(data)
        expected = _cv2_edt_batch(data)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        data = torch.zeros(4, 16, 20, dtype=torch.float32)
        assert _edt_cv2(data).shape == data.shape

    def test_zero_distances_at_background(self):
        torch.manual_seed(99)
        data = (torch.rand(2, 8, 8) > 0.5).float()
        result = _edt_cv2(data)
        assert (result[data == 0] == 0).all()

    def test_positive_distances_at_foreground(self):
        torch.manual_seed(7)
        data = (torch.rand(2, 8, 8) > 0.5).float()
        data[:, 0, 0] = 0.0  # guarantee at least one bg pixel
        result = _edt_cv2(data)
        assert (result[data != 0] > 0).all()

    def test_random(self):
        torch.manual_seed(42)
        data = (torch.rand(3, 16, 16) > 0.6).float()
        result = _edt_cv2(data)
        expected = _cv2_edt_batch(data)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_non_square(self):
        torch.manual_seed(77)
        data = (torch.rand(2, 12, 30) > 0.5).float()
        result = _edt_cv2(data)
        expected = _cv2_edt_batch(data)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_large(self):
        torch.manual_seed(0)
        data = (torch.rand(2, 64, 64) > 0.5).float()
        result = _edt_cv2(data)
        expected = _cv2_edt_batch(data)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)



# ---------------------------------------------------------------------------
# 2. Public edt() dispatcher
# ---------------------------------------------------------------------------

class TestEdtDispatch:
    """Verify the public ``edt()`` function works on CPU."""

    def test_edt_on_cpu(self):
        data = torch.zeros(1, 5, 5, dtype=torch.float32)
        data[0, 2, 2] = 1.0
        result = edt(data)
        expected = _cv2_edt_batch(data)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_edt_output_shape(self):
        data = torch.zeros(4, 16, 20, dtype=torch.float32)
        assert edt(data).shape == data.shape

    def test_edt_random(self):
        torch.manual_seed(123)
        data = (torch.rand(2, 20, 20) > 0.5).float()
        result = edt(data)
        expected = _cv2_edt_batch(data)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_edt_2d_input(self):
        """2D input should be auto-promoted to 3D."""
        data = torch.zeros(8, 8, dtype=torch.float32)
        data[3, 3] = 1.0
        result = edt(data)
        assert result.shape == (1, 8, 8)

    def test_edt_threaded(self):
        """Threaded path should produce same result as single-threaded."""
        torch.manual_seed(55)
        data = (torch.rand(4, 16, 16) > 0.5).float()
        result_single = edt(data, workers=1)
        result_threaded = edt(data, workers=None)
        torch.testing.assert_close(result_single, result_threaded)


# ---------------------------------------------------------------------------
# 3. Triton vs OpenCV agreement (skipped if triton/CUDA unavailable)
# ---------------------------------------------------------------------------

@_skip_no_triton
@_skip_no_cuda
class TestTritonAgreement:
    """When triton + CUDA are available, triton must match OpenCV."""

    def _compare(self, data_cpu: torch.Tensor, atol: float = 1e-4):
        from sam3.model.edt import _edt_triton

        expected = _cv2_edt_batch(data_cpu)
        result = _edt_triton(data_cpu.cuda()).cpu()
        torch.testing.assert_close(result, expected, atol=atol, rtol=1e-4)

    def test_single_pixel(self):
        data = torch.zeros(1, 8, 8, dtype=torch.float32)
        data[0, 4, 4] = 1.0
        self._compare(data)

    def test_random_small(self):
        torch.manual_seed(123)
        data = (torch.rand(2, 16, 16) > 0.6).float()
        self._compare(data)

    def test_random_medium(self):
        torch.manual_seed(456)
        data = (torch.rand(4, 32, 32) > 0.5).float()
        self._compare(data)

    def test_non_square(self):
        torch.manual_seed(789)
        data = (torch.rand(2, 20, 40) > 0.5).float()
        self._compare(data)

    def test_all_background(self):
        data = torch.zeros(1, 8, 8, dtype=torch.float32)
        self._compare(data)


# ---------------------------------------------------------------------------
# 4. CuPy vs OpenCV agreement (skipped if cupy/CUDA unavailable)
# ---------------------------------------------------------------------------

@_skip_no_cupy
@_skip_no_cuda
class TestCupyAgreement:
    """When cupy + CUDA are available, cupy must match OpenCV."""

    def _compare(self, data_cpu: torch.Tensor, atol: float = 1e-4):
        from sam3.model.edt import _edt_cupy

        expected = _cv2_edt_batch(data_cpu)
        result = _edt_cupy(data_cpu.cuda()).cpu()
        torch.testing.assert_close(result, expected, atol=atol, rtol=1e-4)

    def test_single_pixel(self):
        data = torch.zeros(1, 8, 8, dtype=torch.float32)
        data[0, 4, 4] = 1.0
        self._compare(data)

    def test_random_small(self):
        torch.manual_seed(123)
        data = (torch.rand(2, 16, 16) > 0.6).float()
        self._compare(data)

    def test_random_medium(self):
        torch.manual_seed(456)
        data = (torch.rand(4, 32, 32) > 0.5).float()
        self._compare(data)

    def test_non_square(self):
        torch.manual_seed(789)
        data = (torch.rand(2, 20, 40) > 0.5).float()
        self._compare(data)

    def test_all_background(self):
        data = torch.zeros(1, 8, 8, dtype=torch.float32)
        self._compare(data)
