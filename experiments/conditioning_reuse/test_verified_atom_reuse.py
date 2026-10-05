"""CPU sensitivity checks; these fixtures do not establish model parity."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.conditioning_reuse import verified_atom_reuse as module


class Encoder(torch.nn.Module):
    def __init__(self, change=False, random=False):
        super().__init__()
        self.change = change
        self.random = random
        self.noisy_position_embedder = torch.nn.Module()
        self.noisy_position_embedder.linear_r = torch.nn.Linear(3, 4, bias=False)

    def get_atom_reps(self, batch, rl=None, si_trunk=None, zij_trunk=None):
        if self.random:
            torch.rand(1)
        cl = torch.full((1, 1, 3, 4), batch['base'])
        plm = torch.zeros(1, 1, 3, 3, 2)
        if self.change and rl.flatten()[0].item() > 0:
            plm[..., 0] = -0.0
        ql = cl + self.noisy_position_embedder.linear_r(rl)
        return ql, cl, plm


class Diffusion(torch.nn.Module):
    def __init__(self, mutate=False, fail=False, **kwargs):
        super().__init__()
        self.atom_attn_enc = Encoder(**kwargs)
        self.mutate = mutate
        self.fail = fail

    def forward(self, batch, rl):
        ql, cl, plm = self.atom_attn_enc.get_atom_reps(batch, rl)
        if self.mutate:
            plm.add_(1)
        if self.fail:
            raise RuntimeError('injected consumer failure')
        return ql + cl


class Sample(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.diffusion_module = Diffusion(**kwargs)

    def _sample_rollout(self, base):
        return [self.diffusion_module({'base': base}, torch.full((1, 1, 3, 3), float(i)))
                for i in range(3)]


@contextmanager
def strict():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.inference_mode():
            yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)


class AtomReuseTest(unittest.TestCase):
    def assert_restored(self, sample):
        self.assertNotIn('_sample_rollout', sample.__dict__)
        self.assertNotIn('forward', sample.diffusion_module.__dict__)
        self.assertNotIn('get_atom_reps', sample.diffusion_module.atom_attn_enc.__dict__)

    def test_two_rollouts_reset_and_match_native(self):
        sample = Sample().eval()
        with strict(), tempfile.TemporaryDirectory() as directory:
            expected = [sample._sample_rollout(base) for base in (1., 8.)]
            with patch.object(module, 'validate_atom_sources'), module.verified_atom_reuse(sample, directory) as report:
                actual = [sample._sample_rollout(base) for base in (1., 8.)]
            for a, b in zip(sum(expected, []), sum(actual, [])):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(len(report['scopes']), 2)
            for scope in report['scopes']:
                self.assertTrue(scope['completed'])
                self.assertEqual(len(scope['calls']), 3)
                self.assertTrue(all(c['completed'] and c['consumer_unchanged']
                                    for c in scope['calls']))
            self.assert_restored(sample)

    def test_dependency_rng_mutation_and_exception_fail_closed(self):
        for settings, message in ((dict(change=True), 'differs'),
                                  (dict(random=True), 'differs'),
                                  (dict(mutate=True), 'mutated'),
                                  (dict(fail=True), 'injected')):
            with self.subTest(settings=settings), strict(), tempfile.TemporaryDirectory() as directory:
                sample = Sample(**settings).eval()
                with patch.object(module, 'validate_atom_sources'):
                    with self.assertRaisesRegex(RuntimeError, message):
                        with module.verified_atom_reuse(sample, directory):
                            sample._sample_rollout(1.)
                data = json.loads(Path(directory, 'atom-reuse-check.json').read_text())
                self.assertFalse(data['scopes'][0]['completed'])
                self.assert_restored(sample)

    def test_source_guard_rejects_fixture(self):
        sample = Sample().eval()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
                with module.verified_atom_reuse(sample, directory):
                    pass
        self.assert_restored(sample)

    def test_training_rejected(self):
        with strict(), tempfile.TemporaryDirectory() as directory:
            sample = Sample()
            with patch.object(module, 'validate_atom_sources'):
                with self.assertRaisesRegex(RuntimeError, 'strict deterministic inference'):
                    with module.verified_atom_reuse(sample, directory):
                        sample._sample_rollout(1.)
            self.assert_restored(sample)


if __name__ == '__main__':
    unittest.main()
