"""Storage rejection checks; GPU arithmetic is tested by the strict screen."""
import unittest

import torch

from experiments.hip_softmax.owned_scores import disjoint_full_storage


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


if __name__ == "__main__":
    unittest.main()
