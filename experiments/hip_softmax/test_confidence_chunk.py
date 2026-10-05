import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.hip_softmax import confidence_chunk as module


def tensors(n, dtype=torch.float32):
    return (torch.empty(1, n, 384, dtype=dtype, device='meta'),
            torch.empty(1, n, n, 128, dtype=dtype, device='meta'))


class ChunkPolicyTest(unittest.TestCase):
    def test_only_measured_length_and_precision_are_bounded(self):
        self.assertEqual(module.chunk_limit(tensors(1536), 1024), 256)
        self.assertEqual(module.chunk_limit(tensors(1536), 128), 128)
        self.assertEqual(module.chunk_limit(tensors(1000), 1024), 1024)
        self.assertEqual(module.chunk_limit(tensors(1536, torch.float16), 1024), 1024)
        self.assertEqual(module.chunk_limit((), 1024), 1024)

    def test_original_tuner_executes_and_scope_restores_on_failure(self):
        class Tuner:
            cached_arg_data = None
            cached_chunk_size = None

            def tune_chunk_size(self, representative_fn, args, max_chunk_size):
                return representative_fn(max_chunk_size)

        tuner = Tuner()
        stack = SimpleNamespace(training=False, chunk_size_tuner=tuner)
        model = SimpleNamespace(aux_heads=SimpleNamespace(
            pairformer_embedding=SimpleNamespace(pairformer_stack=stack)))
        old, warn = (torch.are_deterministic_algorithms_enabled(),
                     torch.is_deterministic_algorithms_warn_only_enabled())
        torch.use_deterministic_algorithms(True)
        try:
            with torch.inference_mode(), patch.object(module, 'validate_sources'):
                report = {}
                with module.confidence_chunk_cap(model, report):
                    self.assertEqual(tuner.tune_chunk_size(lambda n: n // 2, tensors(1536), 1024), 128)
                self.assertEqual(report['calls'][0]['bounded_max'], 256)
                self.assertEqual(report['calls'][0]['selected'], 128)
                self.assertNotIn('tune_chunk_size', tuner.__dict__)
                with self.assertRaisesRegex(ValueError, 'injected'):
                    with module.confidence_chunk_cap(model, {}):
                        raise ValueError('injected')
                self.assertNotIn('tune_chunk_size', tuner.__dict__)
                tuner.cached_chunk_size = 256
                with self.assertRaisesRegex(RuntimeError, 'fresh unused'):
                    with module.confidence_chunk_cap(model, {}):
                        pass
        finally:
            torch.use_deterministic_algorithms(old, warn_only=warn)

    def test_source_mismatch_is_rejected(self):
        class Stack:
            def _prep_blocks(self):
                pass

        class Tuner:
            def tune_chunk_size(self):
                pass

        with self.assertRaisesRegex(RuntimeError, 'Unsupported confidence chunk source'):
            module.validate_sources(Stack(), Tuner())


if __name__ == '__main__':
    unittest.main()
