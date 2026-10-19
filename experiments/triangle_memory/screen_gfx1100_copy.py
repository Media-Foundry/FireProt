"""N1000/tail-chunk bit patterns, canaries, streams and complete copy timings."""
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.utils.deterministic
from gfx1100_copy import Gfx1100Copy
from screen import bits_equal, measure


@torch.inference_mode()
def main(build, destination):
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(101)
    receipt = json.loads(Path(build).read_text())
    runtime = Gfx1100Copy(receipt['library'], receipt['library_sha256'])
    rows = []
    for chunk, offset in ((256, 0), (256, 512), (232, 768)):
        src = torch.randint(-(2**31), 2**31-1, (chunk, 1000, 128), device='cuda', dtype=torch.int32)
        src.flatten()[:4] = torch.tensor([0, -2147483648, 2143289345, 2139095040], device='cuda', dtype=torch.int32)
        src = src.view(torch.float32)
        for mode in ('row', 'column', 'bmm', 'norm'):
            if mode == 'row':
                inp = src.permute(2, 0, 1)
                parent = torch.full((128, 1000, 1000), 17., device='cuda')
                out = parent[:, offset:offset+chunk, :]
            elif mode == 'column':
                inp = src.permute(2, 1, 0)
                parent = torch.full((128, 1000, 1000), 17., device='cuda')
                out = parent[:, :, offset:offset+chunk]
            elif mode == 'bmm':
                inp = src.permute(2, 1, 0)
                parent = out = torch.empty(inp.shape, device='cuda')
            else:
                inp = src.permute(2, 1, 0).contiguous().permute(1, 2, 0)
                parent = out = torch.empty(inp.shape, device='cuda')
            expected = parent.clone()
            ref = (expected[:, offset:offset+chunk, :] if mode == 'row' else
                   expected[:, :, offset:offset+chunk] if mode == 'column' else expected)
            ref.copy_(inp)
            before = runtime.calls
            runtime.copy_(out, inp)
            assert runtime.calls == before+1 and bits_equal(parent, expected)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                runtime.copy_(out, inp)
            torch.cuda.current_stream().wait_stream(stream)
            assert bits_equal(parent, expected)
            samples = dict(native=[], hip=[])
            functions = dict(native=lambda: out.copy_(inp), hip=lambda: runtime.copy_(out, inp))
            for order in [('native', 'hip'), ('hip', 'native'), ('native', 'hip')]:
                for arm in order:
                    samples[arm].append(measure(functions[arm]))
            medians = {arm: statistics.median(v) for arm, v in samples.items()}
            rows.append(dict(mode=mode, chunk=chunk, offset=offset, exact=True,
                             samples_ms=samples, medians_ms=medians,
                             speedup=medians['native']/medians['hip']))
            del parent, out, expected, ref, inp
        del src
    before = runtime.fallbacks
    a = torch.arange(32, device='cuda', dtype=torch.float32)
    runtime.copy_(a, a)
    b = torch.empty_like(a)
    runtime.copy_(b, a)
    assert runtime.fallbacks == before+2 and bits_equal(a, b)
    with Path(destination).open('x') as out:
        json.dump(dict(correctness_passed=True, results=rows, calls=runtime.calls,
            fallbacks=runtime.fallbacks, scope='Copy operator only, not E2E',
            device=torch.cuda.get_device_name(), torch=str(torch.__version__), hip=torch.version.hip,
            strict=True, seed=101), out, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
