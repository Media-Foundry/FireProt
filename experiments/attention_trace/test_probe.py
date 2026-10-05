"""CPU checks of observer transparency and scope; no GPU performance claims."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.attention_trace.probe import AttentionTrace, trace_operations


def native(query, key, value, biases, use_high_precision=False):
    scores = torch.einsum('...qc, ...kc->...qk', query, key)
    for bias in biases:
        scores += bias
    return torch.einsum('...qk, ...kc->...qc', scores.softmax(-1), value)


def dispatch(function, *args, **kwargs):
    return function(*args, **kwargs)


class TraceTest(unittest.TestCase):
    def test_kernel_correlation_rejects_duplicate_launches(self):
        events = [
            dict(cat='user_annotation', ph='X', name='fireprot.attention.QK',
                 pid=1, tid=1, ts=0, dur=10),
            dict(cat='cpu_op', name='aten::bmm', pid=1, tid=1, ts=2, dur=3,
                 args={'External id': 9, 'Input Dims': [[2, 3, 4], [2, 4, 3]],
                       'Input Strides': [[12, 4, 1], [12, 3, 1]]}),
            dict(cat='cuda_runtime', name='hipExtModuleLaunchKernel',
                 pid=1, tid=1, ts=3, dur=1, args={'External id': 9, 'correlation': 6}),
            dict(cat='kernel', name='fixture_gemm', ts=100, dur=10,
                 args={'External id': 9, 'correlation': 6, 'device': 0, 'stream': 1}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'trace.json')
            data = {'einsums': [{'label': 'QK'}]}
            path.write_text(json.dumps({'traceEvents': events}))
            trace_operations(data, path)
            self.assertEqual(data['einsums'][0]['operations'][0]['kernels'][0]['name'],
                             'fixture_gemm')
            for duplicate in (events[2], events[3]):
                path.write_text(json.dumps({'traceEvents': events + [duplicate]}))
                with self.assertRaisesRegex(RuntimeError, 'correlation'):
                    trace_operations(data, path)

    @torch.inference_mode()
    def test_layout_correlation_and_exact_result(self):
        generator = torch.Generator().manual_seed(101)
        # Head-interleaved input exercises the einsum packing path.
        q, k, v = [torch.randn(2, 7, 4, 3, generator=generator).transpose(1, 2)
                   for _ in range(3)]
        bias = torch.randn(1, 4, 7, 7, generator=generator)
        expected = native(q, k, v, [bias])
        state = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            probe = AttentionTrace(dispatch, directory, q.shape, occurrence=2, cpu_only=True)
            self.assertTrue(torch.equal(expected, probe(native, q, k, v, [bias])))
            self.assertFalse(probe.attempted)
            actual = probe(native, q, k, v, [bias])
            probe.require_complete()
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            path = Path(directory, 'attention-trace.json')
            first = path.read_bytes()
            data = json.loads(first)
            self.assertTrue(data['completed'])
            self.assertEqual([r['label'] for r in data['einsums']], ['QK', 'PV'])
            for r in data['einsums']:
                self.assertTrue(r['completed'])
                self.assertTrue(any(op['operation'] == 'aten::bmm'
                                    for op in r['operations']))
            self.assertEqual(data['query']['strides'], list(q.stride()))
            events = json.loads(Path(directory, data['trace']).read_text())['traceEvents']
            names = {e.get('name') for e in events}
            self.assertTrue({'fireprot.attention.QK', 'fireprot.attention.PV'} <= names)
            self.assertTrue(torch.equal(expected, probe(native, q, k, v, [bias])))
            self.assertEqual(path.read_bytes(), first)

    def test_failure_is_recorded_without_retry(self):
        q = torch.zeros(1, 1, 2, 2)
        def failure(*args, **kwargs):
            raise RuntimeError('injected attention failure')
        with tempfile.TemporaryDirectory() as directory:
            probe = AttentionTrace(dispatch, directory, q.shape, cpu_only=True)
            with self.assertRaisesRegex(RuntimeError, 'injected attention failure'):
                probe(failure, q, q, q, [])
            data = json.loads(Path(directory, 'attention-trace.json').read_text())
            self.assertFalse(data['completed'])
            self.assertTrue(probe.attempted)
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                probe.require_complete()
            self.assertTrue(torch.equal(probe(native, q, q, q, []), q))

    def test_nonmatching_call_passes_through(self):
        q = torch.zeros(1, 1, 2, 2)
        with tempfile.TemporaryDirectory() as directory:
            probe = AttentionTrace(dispatch, directory, (2, 1, 2, 2), cpu_only=True)
            self.assertTrue(torch.equal(probe(native, q, q, q, []), q))
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaisesRegex(RuntimeError, 'not observed'):
                probe.require_complete()


if __name__ == '__main__':
    unittest.main()
