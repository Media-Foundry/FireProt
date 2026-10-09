"""CPU regressions for parent-state restoration and Lightning-style batching."""
from contextlib import ExitStack
from functools import partial
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, SequentialSampler
from lightning_fabric.utilities.seed import pl_worker_init_function
from isolated_scope import isolated_data_scope

class Sample(Dataset):
    def __init__(self, fail=False):
        self.fail = fail
    def __len__(self):
        return 1
    def __getitem__(self, index):
        value = torch.rand(2) + random.random() + float(np.random.rand())
        if self.fail:
            raise ValueError('sample failure')
        return value

class ScopeTests(unittest.TestCase):
    def test_parent_state_restored_for_success_and_failure(self):
        initial_threads = torch.get_num_threads()
        initial_determinism = torch.are_deterministic_algorithms_enabled()
        initial_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True)
            for fail in (False, True):
                dataset = Sample(fail)
                # Lightning supplies a batch sampler, making batch_size None.
                loader = DataLoader(dataset, batch_sampler=BatchSampler(SequentialSampler(dataset), 1, False),
                    num_workers=0, generator=torch.Generator().manual_seed(101),
                    worker_init_fn=partial(pl_worker_init_function, rank=0))
                self.assertIsNone(loader.batch_size)
                cpu = torch.get_rng_state().clone()
                py = random.getstate()
                np_state = np.random.get_state()
                original = DataLoader._get_iterator
                with tempfile.TemporaryDirectory() as root:
                    with patch('torch.cuda.device_count', return_value=0), isolated_data_scope(root):
                        if fail:
                            with self.assertRaisesRegex(ValueError, 'sample failure'):
                                next(iter(loader))
                        else:
                            self.assertEqual(tuple(next(iter(loader)).shape), (1, 2))
                    self.assertTrue(Path(root, 'isolated-data.json').is_file())
                self.assertIs(DataLoader._get_iterator, original)
                self.assertTrue(torch.equal(torch.get_rng_state(), cpu))
                self.assertEqual(random.getstate(), py)
                after = np.random.get_state()
                self.assertEqual(after[0], np_state[0])
                np.testing.assert_array_equal(after[1], np_state[1])
                self.assertEqual(after[2:], np_state[2:])
                self.assertEqual(torch.get_num_threads(), 2)
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.set_num_threads(initial_threads)
            torch.use_deterministic_algorithms(initial_determinism, warn_only=initial_warn)

if __name__ == '__main__':
    unittest.main()
