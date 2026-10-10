"""CPU lifecycle tests; actual autocast arithmetic requires the GPU diagnostic."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from experiments.openfold3_precision.scope import pairformer_precision_scope


class Triangle(torch.nn.Module):
    def _inference_forward(self):
        return None


class Linear(torch.nn.Module):
    def forward(self, x):
        return x.to(torch.float16)


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear()
        self.inject = None

    def forward(self, s, z):
        if self.inject is not None:
            self.inject()
        return self.linear(s), self.linear(z)


class Sample(torch.nn.Module):
    def __init__(self, diffusion):
        super().__init__()
        self.diffusion = diffusion

    def forward(self, value):
        return self.diffusion(self.diffusion(value))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pairformer_stack = Stack()
        self.diffusion_module = torch.nn.Identity()
        self.sample_diffusion = Sample(self.diffusion_module)
        self.aux_heads = torch.nn.Identity()
        self.shared = types.SimpleNamespace(num_recycles=1,
            diffusion=types.SimpleNamespace(no_full_rollout_steps=2))


class ScopeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        module = types.ModuleType('openfold3.core.model.layers.triangular_multiplicative_update')
        module.TriangleMultiplicativeUpdate = Triangle
        p = patch.dict(sys.modules, {module.__name__: module})
        p.start(); self.addCleanup(p.stop)
        digests = iter(('1cd02947e24bcea5d30dba98122e6a38e707f88ebe1d4c4f2aa16f4b177f7e0c',
                        'cdb66c9553ef42b93835dec92f3e6dcf711dbf139a5a59aa1f2561216435af9f',
                        '612caf7abd07a86a7fdf4eda7e0a2170c04b407f9121fc286ab864a648db1ba7'))
        p = patch('experiments.openfold3_precision.scope.hashlib.sha256',
                  side_effect=lambda _: types.SimpleNamespace(hexdigest=lambda: next(digests)))
        p.start(); self.addCleanup(p.stop)
        # No CUDA in these tests; the toy layer explicitly supplies FP16 outputs.
        p = patch('torch.amp.autocast', side_effect=lambda *a, **k: nullcontext())
        p.start(); self.addCleanup(p.stop)
        self.model = Model().eval()

    def report(self):
        return json.loads((self.output / 'pairformer-precision.json').read_text())

    @torch.inference_mode()
    def test_fp32_boundaries_and_recycle_counts(self):
        x = torch.ones(2, 3)
        with pairformer_precision_scope(self.model, self.output, 'float16'):
            for _ in range(2):
                s, z = self.model.pairformer_stack(x, x)
                self.assertEqual(s.dtype, torch.float32)
                self.assertEqual(z.dtype, torch.float32)
            self.model.sample_diffusion(s)
            self.model.aux_heads(z)
        report = self.report()
        self.assertTrue(report['completed'])
        self.assertEqual(report['boundary_calls']['diffusion_module'], 2)
        self.assertGreater(report['output_dtypes']['torch.float16'], 0)
        self.assertNotIn('forward', self.model.pairformer_stack.__dict__)

    @torch.inference_mode()
    def test_half_overflow_aborts_and_restores_forward(self):
        original = Triangle._inference_forward
        with self.assertRaises(FloatingPointError):
            with pairformer_precision_scope(self.model, self.output, 'float16'):
                self.model.pairformer_stack(torch.full((2, 3), 70000.), torch.ones(2, 3))
        self.assertFalse(self.report()['completed'])
        self.assertIsNotNone(self.report()['nonfinite'])
        self.assertIs(Triangle._inference_forward, original)
        self.assertNotIn('forward', self.model.pairformer_stack.__dict__)

    @torch.inference_mode()
    def test_diffusion_inside_low_precision_region_is_rejected(self):
        self.model.pairformer_stack.inject = lambda: self.model.diffusion_module(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, 'escaped PairFormer'):
            with pairformer_precision_scope(self.model, self.output, 'float16'):
                self.model.pairformer_stack(torch.ones(1), torch.ones(1))
        self.assertFalse(self.report()['completed'])
        self.assertEqual(len(self.model.diffusion_module._forward_pre_hooks), 0)

    @torch.inference_mode()
    def test_missing_diffusion_work_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Diffusion step coverage'):
            with pairformer_precision_scope(self.model, self.output, 'float16'):
                for _ in range(2):
                    self.model.pairformer_stack(torch.ones(1), torch.ones(1))
        self.assertFalse(self.report()['completed'])


if __name__ == '__main__':
    unittest.main()
