"""CPU/meta guards and restoration; numerical GPU gates live in the screen."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import gfx1100_copy as candidate


class FakeRuntime:
    calls = 0
    fallbacks = 0

    def __init__(self, *args):
        pass


class Triangle:
    def _inference_forward(self, z, mask=None, inplace_chunk_size=None, with_add=True,
                           use_triton_triangle_kernels=False):
        return z


class GuardTests(unittest.TestCase):
    def test_tail_and_interior_layouts(self):
        for chunk in (232, 256):
            x = torch.empty((1, chunk, 1000, 128), device='meta')
            self.assertIsNotNone(candidate.Gfx1100Copy.layout(x))
            self.assertIsNotNone(candidate.Gfx1100Copy.layout(x.permute(0, 3, 2, 1)))

    def test_unqualified_length_chunk_and_overlap(self):
        for shape in ((256, 1536, 128), (128, 1000, 128), (2, 256, 1000, 128)):
            self.assertIsNone(candidate.Gfx1100Copy.layout(torch.empty(shape, device='meta')))
        x = torch.empty((128, 1000, 256), device='meta')
        self.assertIsNone(candidate.Gfx1100Copy.layout(x.as_strided(x.shape, (1, 1, 1))))

    def test_wrong_library_hash_fails_before_device_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'unqualified.so'
            path.write_bytes(b'not a library')
            with self.assertRaisesRegex(ValueError, 'digest'):
                candidate.Gfx1100Copy(path, '0'*64)

    def fake_patch(self, runtime):
        def unexpected(*args, **kwargs):
            raise AssertionError('Unsupported CPU input reached candidate arithmetic')
        return Triangle, unexpected

    def test_disabled_cpu_route_restores_original(self):
        original = Triangle._inference_forward
        with tempfile.TemporaryDirectory() as directory, patch.object(candidate, 'Gfx1100Copy', FakeRuntime), patch.object(candidate, 'patched', self.fake_patch):
            with candidate.triangle_copy_scope(None, directory, 'unused', 'unused', False):
                value = torch.ones(2)
                self.assertIs(Triangle()._inference_forward(value), value)
            report = json.loads(Path(directory, 'triangle-copy.json').read_text())
            self.assertTrue(report['completed'])
            self.assertEqual(report['counts'], dict(accepted=0, fallback=1))
        self.assertIs(Triangle._inference_forward, original)

    def test_zero_coverage_rejected_and_restored(self):
        original = Triangle._inference_forward
        with tempfile.TemporaryDirectory() as directory, patch.object(candidate, 'Gfx1100Copy', FakeRuntime), patch.object(candidate, 'patched', self.fake_patch):
            with self.assertRaisesRegex(RuntimeError, 'no HIP coverage'):
                with candidate.triangle_copy_scope(None, directory, 'unused', 'unused', True):
                    Triangle()._inference_forward(torch.ones(2))
            self.assertFalse(json.loads(Path(directory, 'triangle-copy.json').read_text())['completed'])
        self.assertIs(Triangle._inference_forward, original)

    def test_body_exception_restores_original(self):
        original = Triangle._inference_forward
        with tempfile.TemporaryDirectory() as directory, patch.object(candidate, 'Gfx1100Copy', FakeRuntime), patch.object(candidate, 'patched', self.fake_patch):
            with self.assertRaisesRegex(RuntimeError, 'body failure'):
                with candidate.triangle_copy_scope(None, directory, 'unused', 'unused', False):
                    raise RuntimeError('body failure')
        self.assertIs(Triangle._inference_forward, original)


if __name__ == '__main__':
    unittest.main()
