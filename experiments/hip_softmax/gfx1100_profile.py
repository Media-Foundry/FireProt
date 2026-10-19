"""Diagnostic stage events and two sampled traces; never an E2E speedup run."""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch


@contextmanager
def profile_scope(model, output):
    output = Path(output)
    saved, events, traces = [], [], []
    counts = defaultdict(int)
    sampled = {
        id(model.pairformer_stack.blocks[1]): ('pairformer_block', 2),
        id(model.sample_diffusion.diffusion_module): ('diffusion_step', 5),
    }
    modules = {id(module): (name, module) for name, module in model.named_children()}
    for name, module in model.named_modules():
        if id(module) in sampled or type(module).__name__.startswith(('TriangleMultiplication', 'TriangleAttention')):
            modules.setdefault(id(module), (name, module))

    def wrap(name, module, original):
        def forward(*args, **kwargs):
            counts[name] += 1
            sample = sampled.get(id(module))
            capture = sample is not None and counts[name] == sample[1]
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            if capture:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=False, with_stack=False,
                ) as profiler:
                    start.record()
                    result = original(*args, **kwargs)
                    end.record()
                path = output / (sample[0] + '.chrome.json')
                profiler.export_chrome_trace(str(path))
                operations = [dict(name=e.key, count=e.count, shapes=e.input_shapes,
                    self_cpu_us=e.self_cpu_time_total, self_device_us=e.self_device_time_total,
                    device_us=e.device_time_total)
                    for e in profiler.key_averages(group_by_input_shape=True)]
                traces.append(dict(module=name, call=counts[name], operations=operations,
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            else:
                start.record()
                result = original(*args, **kwargs)
                end.record()
            events.append((name, counts[name], capture, start, end))
            return result
        return forward

    try:
        for name, module in modules.values():
            saved.append((module, 'forward' in module.__dict__, module.__dict__.get('forward')))
            module.forward = wrap(name, module, module.forward)
        yield
        torch.cuda.synchronize()
        assert len(traces) == 2, 'Both steady-call traces must execute'
        report = dict(schema_version=1, completed=True, traces=traces,
            events=[dict(module=n, call=c, profiled=p, inclusive_ms=a.elapsed_time(b))
                    for n, c, p, a, b in events],
            limits=['Diagnostic instrumentation; timings are not E2E benchmark results.',
                    'Nested module durations overlap and must not be summed.',
                    'Sampled traces do not establish CU saturation or compute/memory bounds.'])
        (output / 'stage-profile.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    finally:
        for module, existed, previous in reversed(saved):
            if existed:
                module.forward = previous
            else:
                delattr(module, 'forward')
