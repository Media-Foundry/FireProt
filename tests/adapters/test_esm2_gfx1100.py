"""Guard the exact ESM2 gfx1100 capability boundary."""
import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from fireprot.adapters.plm.runtime import gfx1100_esm2_exact_route
from fireprot.backends.amd import esm2_gemm


class ESM2Gfx1100SelectionTest(unittest.TestCase):
    def test_exact_runtime_without_hip_library_is_selected(self):
        self.assertTrue(gfx1100_esm2_exact_route(
            'esm2-3b', {}, 'gfx1100', '2.13.0+rocm10.0.0', '7.15.26333', 4))

    def test_every_capability_dimension_fails_closed(self):
        cases = [
            ('esm3-open', {}, 'gfx1100', '2.13.0+rocm10.0.0', '7.15.26333', 4),
            ('esm2-3b', {'hip_esm': {}}, 'gfx1100', '2.13.0+rocm10.0.0', '7.15.26333', 4),
            ('esm2-3b', {}, 'gfx90a', '2.13.0+rocm10.0.0', '7.15.26333', 4),
            ('esm2-3b', {}, 'gfx1100', '2.12.0+rocm7.2', '7.15.26333', 4),
            ('esm2-3b', {}, 'gfx1100', '2.13.0+rocm10.0.0', '7.2.53211', 4),
            ('esm2-3b', {}, 'gfx1100', '2.13.0+rocm10.0.0', '7.15.26333', 1),
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(gfx1100_esm2_exact_route(*case))


class ESM2Gfx1100GemmTest(unittest.TestCase):
    def test_support_fails_closed_on_shape_dtype_and_bias(self):
        original = torch.is_grad_enabled
        torch.is_grad_enabled = lambda: False
        try:
            good = SimpleNamespace(is_cuda=True, device=SimpleNamespace(index=0),
                dtype=torch.float32, is_contiguous=lambda: True, requires_grad=False,
                is_conj=lambda: False, is_neg=lambda: False, shape=(1002, 4, 2560),
                numel=lambda: 4008 * 2560)
            weight = SimpleNamespace(**good.__dict__)
            weight.shape = (2560, 2560)
            bias = SimpleNamespace(**good.__dict__)
            bias.shape = (2560,)
            with mock.patch.object(torch.cuda, 'get_device_properties',
                    return_value=SimpleNamespace(gcnArchName='gfx1100:sramecc-')):
                self.assertTrue(esm2_gemm.supported(good, weight, bias))
                good.shape = (4008, 2560)
                self.assertFalse(esm2_gemm.supported(good, weight, bias))
                good.shape = (1002, 4, 2560)
                weight.shape = (2559, 2560)
                self.assertFalse(esm2_gemm.supported(good, weight, bias))
                self.assertFalse(esm2_gemm.supported(good, weight, None))
        finally:
            torch.is_grad_enabled = original


if __name__ == '__main__':
    unittest.main()
