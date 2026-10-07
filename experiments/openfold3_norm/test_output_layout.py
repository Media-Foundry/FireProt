"""GPU regression for native LayerNorm's canonical singleton strides.

Set FIREPROT_TEST_NORM_LIBRARY to a qualified norm.so and HIP_VISIBLE_DEVICES
explicitly. This test loads an existing library; it never builds a kernel.
"""
import os
import unittest

import torch
import torch.utils.deterministic

from screen import Kernel


@unittest.skipUnless(os.environ.get('FIREPROT_TEST_NORM_LIBRARY'),
                     'A qualified HIP normalization library is required')
class OutputLayout(unittest.TestCase):
    @torch.inference_mode()
    def test_sliced_singleton_batch_has_native_output_strides(self):
        self.assertIn(os.environ.get('HIP_VISIBLE_DEVICES'), ('6', '7'))
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.manual_seed(101)
        kernel = Kernel(os.environ['FIREPROT_TEST_NORM_LIBRARY'], 32)
        x = torch.randn(1, 12, 5, 128, device='cuda')[:, :2]
        self.assertTrue(x.is_contiguous())
        self.assertNotEqual(x.stride(), (1280, 640, 128, 1))
        weight = torch.randn(128, device='cuda')
        bias = torch.randn(128, device='cuda')
        expected = torch.native_layer_norm(x, (128,), weight, bias, 1e-5)[0]
        actual = kernel(x, weight, bias)
        self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), 0)


if __name__ == '__main__':
    unittest.main()
