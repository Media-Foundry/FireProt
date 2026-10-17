"""Screen real N1000 model chunk sizes before experimental model integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('library')
    parser.add_argument('sha256')
    parser.add_argument('output')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    sys.path[:0] = [str(repo / 'python'), str(repo / 'runtime/repos/openfold3')]
    import torch
    import openfold3.core.model.primitives.attention as attention
    from fireprot.adapters.openfold3.attention import attention_scope
    from gfx1100_scope import Gfx1100Dispatcher, Gfx1100Softmax
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(101)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    kernel = Gfx1100Softmax(args.library, args.sha256)
    dispatcher = Gfx1100Dispatcher(kernel)
    original = attention._attention
    records = []
    report = dict(schema_version=1, scope='Synthetic operator check at observed model chunk sizes; not E2E',
                  seed=101, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  records=records)
    def candidate(q, k, v, biases):
        with attention_scope(dispatcher):
            return attention._attention(q, k, v, biases)
    def measure(callback):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        value = callback()
        b.record()
        b.synchronize()
        return a.elapsed_time(b)
    try:
        with torch.inference_mode():
            for batch in (232, 256):
                n = 1000
                q, k, v = [torch.randn(batch, n, 4, 32, device='cuda').transpose(1, 2) * 0.125 for _ in range(3)]
                for ending in (False, True):
                    pair = torch.randn(n, n, 4, device='cuda') * 0.1
                    bias = pair.permute((2, 1, 0) if ending else (2, 0, 1)).unsqueeze(0)
                    for kind in ('zero', 'partial', 'nonzero'):
                        storage = torch.zeros(batch, 1, 1, n * 2, device='cuda')
                        mask = storage[..., ::2]
                        if kind == 'partial':
                            mask[..., -7:] = -float('inf')
                        elif kind == 'nonzero':
                            mask.normal_(0, 0.1)
                        biases = [mask, bias]
                        count = kernel.launches_submitted
                        expected = original(q, k, v, biases)
                        actual = candidate(q, k, v, biases)
                        exact = torch.equal(expected.contiguous().view(torch.int32), actual.contiguous().view(torch.int32))
                        launched = kernel.launches_submitted - count
                        assert exact and launched == 1
                        del expected, actual
                        times = [[], []]
                        callbacks = [lambda: original(q, k, v, biases), lambda: candidate(q, k, v, biases)]
                        for iteration in range(4):
                            for arm in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                                times[arm].append(measure(callbacks[arm]))
                        records.append(dict(batch=batch, n=n, ending=ending, mask=kind,
                            exact=exact, validation_launches=launched,
                            native_ms=times[0], candidate_ms=times[1],
                            ratio=statistics.median(times[0]) / statistics.median(times[1])))
                        del storage, mask, biases
                    del pair, bias
                del q, k, v
        report['status'] = 'passed'
    finally:
        report['runtime'] = dispatcher.manifest()
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], cases=len(records), launches=kernel.launches_submitted)))


if __name__ == '__main__':
    main()
