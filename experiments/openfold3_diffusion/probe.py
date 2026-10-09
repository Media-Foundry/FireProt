"""One steady diffusion-call trace; diagnostic only, never an E2E benchmark."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import inspect
import json
import torch

SOURCE_SHA256 = '3b5ed9fb090d53df2b5ad16be517cfdca7f5290a6bb8274e71408a3ff465155d'


def metadata(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), strides=list(value.stride()),
                    dtype=str(value.dtype), device=str(value.device),
                    storage_offset=value.storage_offset(), requires_grad=value.requires_grad)
    if isinstance(value, dict):
        return {str(k): metadata(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [metadata(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return dict(type=type(value).__qualname__)


@contextmanager
def diffusion_probe(model, output, capture_call=5):
    if capture_call != 5:
        raise ValueError('This diagnostic fixes the sampled call to5')
    diffusion = model.sample_diffusion.diffusion_module
    source = Path(inspect.getfile(type(diffusion)))
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Diffusion source differs from the audited implementation')
    if (diffusion.training or not torch.is_inference_mode_enabled()
            or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise RuntimeError('Strict deterministic inference is required')
    output = Path(output)
    original = diffusion.forward
    had_local, local = 'forward' in diffusion.__dict__, diffusion.__dict__.get('forward')
    events, operations, child_calls = [], [], []
    calls = 0
    captured = False
    trace = output / 'diffusion-call5.chrome.json'
    report = dict(scope='Steady diffusion call5; diagnostic instrumentation, not a speedup comparison',
                  source_sha256=SOURCE_SHA256, capture_call=capture_call,
                  input_metadata=None, child_calls=child_calls, torch_operations=operations)

    @contextmanager
    def children():
        saved = []
        def wrap(name, function):
            def call(*args, **kwargs):
                child_calls.append(dict(name=name, inputs=metadata(dict(args=args, kwargs=kwargs))))
                with torch.profiler.record_function('fireprot.diffusion.' + name):
                    return function(*args, **kwargs)
            return call
        try:
            for name, child in diffusion.named_children():
                saved.append((child, 'forward' in child.__dict__, child.__dict__.get('forward')))
                child.forward = wrap(name, child.forward)
            yield
        finally:
            for child, existed, previous in reversed(saved):
                if existed:
                    child.forward = previous
                else:
                    delattr(child, 'forward')

    def forward(*args, **kwargs):
        nonlocal calls, captured
        calls += 1
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if calls != capture_call:
            start.record()
            result = original(*args, **kwargs)
            end.record()
        else:
            report['input_metadata'] = metadata(dict(args=args, kwargs=kwargs))
            with children(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, profile_memory=False, with_stack=False,
            ) as profiler:
                start.record()
                with torch.profiler.record_function('fireprot.diffusion.denoise'):
                    result = original(*args, **kwargs)
                end.record()
            profiler.export_chrome_trace(str(trace))
            for event in profiler.key_averages(group_by_input_shape=True):
                operations.append(dict(name=event.key, count=event.count,
                    input_shapes=event.input_shapes, self_cpu_us=event.self_cpu_time_total,
                    cpu_us=event.cpu_time_total, self_device_us=event.self_device_time_total,
                    device_us=event.device_time_total))
            captured = True
        events.append((calls, start, end))
        return result

    diffusion.forward = forward
    try:
        yield
        torch.cuda.synchronize()
        report.update(completed=True, captured=captured, calls=calls,
            call_events=[dict(call=n, inclusive_ms=a.elapsed_time(b), profiled=n == capture_call)
                         for n, a, b in events],
            trace=dict(path=str(trace), sha256=hashlib.sha256(trace.read_bytes()).hexdigest()) if captured else None,
            limits=['One sampled module call; no complete rollout kernel census.',
                    'Outer augmentation/noise/scheduler work is outside the sampled module.',
                    'Stage/operator times overlap; do not sum nested durations.',
                    'Call5 contains profiler overhead; other calls still include event instrumentation.',
                    'Shapes/strides and trace durations do not establish CU saturation or a memory/computation bound.'])
        if calls != 200 or not captured or not operations:
            raise RuntimeError('Expected exactly200 calls and one nonempty steady-call capture')
        (output / 'diffusion-profile.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    finally:
        if had_local:
            diffusion.forward = local
        else:
            delattr(diffusion, 'forward')
