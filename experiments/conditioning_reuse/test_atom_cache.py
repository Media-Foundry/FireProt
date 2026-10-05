"""Lifecycle checks; source/eligibility overrides apply only to CPU fixtures."""
import tempfile
import unittest
from functools import wraps
from unittest.mock import patch

import torch

from experiments.conditioning_reuse import atom_cache as module
from experiments.conditioning_reuse.test_verified_atom_reuse import Sample, strict


class AtomCacheTest(unittest.TestCase):
    def test_hits_native_call_count_and_scope_reset(self):
        sample = Sample().eval()
        encoder = sample.diffusion_module.atom_attn_enc
        native = encoder.get_atom_reps
        calls = []
        @wraps(native)
        def counted(*args, **kwargs):
            calls.append(1)
            return native(*args, **kwargs)
        with strict(), tempfile.TemporaryDirectory() as directory:
            expected = sample._sample_rollout(1.)
            with patch.object(module, 'validate_atom_sources'), patch.object(module, 'eligible', return_value=True):
                with patch.object(encoder, 'get_atom_reps', counted):
                    with module.atom_cache(sample, directory) as report:
                        for _ in range(2):
                            actual = sample._sample_rollout(1.)
                            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
                    self.assertEqual(len(calls), 2)
            self.assertEqual([(s['hits'], s['misses']) for s in report['scopes']], [(2, 1), (2, 1)])
            self.assertNotIn('_sample_rollout', sample.__dict__)
            self.assertNotIn('get_atom_reps', encoder.__dict__)

    def test_changed_dependency_invalidates(self):
        sample = Sample().eval()
        encoder = sample.diffusion_module.atom_attn_enc
        def rollout(base):
            return [sample.diffusion_module({'base': b}, torch.zeros(1, 1, 3, 3))
                    for b in (base, base + 1, base + 1)]
        sample._sample_rollout = rollout
        with strict(), tempfile.TemporaryDirectory() as directory:
            expected = rollout(1.)
            with patch.object(module, 'validate_atom_sources'), patch.object(module, 'eligible', return_value=True):
                with module.atom_cache(sample, directory) as report:
                    actual = sample._sample_rollout(1.)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
            self.assertEqual((report['scopes'][0]['hits'], report['scopes'][0]['misses']), (1, 2))
            self.assertIs(sample._sample_rollout, rollout)

    def test_unsupported_input_falls_back(self):
        sample = Sample().eval()
        with strict(), tempfile.TemporaryDirectory() as directory:
            expected = sample._sample_rollout(1.)
            with patch.object(module, 'validate_atom_sources'):
                with module.atom_cache(sample, directory) as report:
                    actual = sample._sample_rollout(1.)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
            self.assertEqual(report['scopes'][0]['fallbacks'], 3)

    def test_source_guard_and_exception_restore(self):
        sample = Sample(fail=True).eval()
        with strict(), tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
                with module.atom_cache(sample, directory):
                    pass
            with patch.object(module, 'validate_atom_sources'), patch.object(module, 'eligible', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'injected consumer failure'):
                    with module.atom_cache(sample, directory):
                        sample._sample_rollout(1.)
            self.assertNotIn('_sample_rollout', sample.__dict__)
            self.assertNotIn('get_atom_reps', sample.diffusion_module.atom_attn_enc.__dict__)


if __name__ == '__main__':
    unittest.main()
