"""Storage rejection checks; GPU arithmetic is tested by the strict screen."""
import unittest
from types import SimpleNamespace

import torch

from experiments.hip_softmax.owned_scores import disjoint_full_storage
from experiments.hip_softmax.scope_softmax import confidence_scope


class OwnershipGuards(unittest.TestCase):
    def test_private_full_allocation_and_full_view(self):
        scores = torch.empty(16)
        self.assertTrue(disjoint_full_storage(scores, [torch.empty(4)]))
        self.assertTrue(disjoint_full_storage(scores.view(4, 4), [torch.empty(4)]))

    def test_partial_storage_rejected(self):
        storage = torch.empty(32)
        self.assertFalse(disjoint_full_storage(storage[:16], [torch.empty(4)]))
        self.assertFalse(disjoint_full_storage(storage[16:], [torch.empty(4)]))

    def test_aliased_bias_rejected(self):
        scores = torch.empty(16)
        self.assertFalse(disjoint_full_storage(scores, [scores[-4:]]))
        self.assertFalse(disjoint_full_storage(scores.view(4, 4), [scores.view(4, 4).T]))

    def test_confidence_scope_restores_after_return_and_exception(self):
        state = SimpleNamespace(in_confidence=0)

        class Head:
            def forward(self, fail=False):
                if state.in_confidence != 1:
                    raise AssertionError('Missing confidence scope')
                if fail:
                    raise ValueError('injected')
                return 7

        head = Head()
        model = SimpleNamespace(aux_heads=head)
        with confidence_scope(model, state):
            self.assertEqual(head.forward(), 7)
        self.assertEqual(state.in_confidence, 0)
        self.assertNotIn('forward', head.__dict__)
        with self.assertRaisesRegex(ValueError, 'injected'):
            with confidence_scope(model, state):
                head.forward(True)
        self.assertEqual(state.in_confidence, 0)
        self.assertNotIn('forward', head.__dict__)

    def test_confidence_scope_restores_existing_instance_override(self):
        state = SimpleNamespace(in_confidence=0)
        original = lambda: state.in_confidence
        head = SimpleNamespace(forward=original)
        with confidence_scope(SimpleNamespace(aux_heads=head), state):
            self.assertEqual(head.forward(), 1)
        self.assertIs(head.forward, original)
        self.assertEqual(head.forward(), 0)


if __name__ == "__main__":
    unittest.main()
