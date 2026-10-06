"""Triangle multiplication substage events; diagnostic only, never E2E evidence."""
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import torch


@contextmanager
def measure(model, output):
    from fireprot.backends.amd.extension import load_amd_extension
    from fireprot.backends.amd import lt_bmm
    extension = load_amd_extension()
    if extension is None:
        raise RuntimeError('Triangle probe requires the selected AMD extension')
    events = defaultdict(list)
    handles = []
    replacements = []
    pending = defaultdict(list)

    def wrap(owner, name, label):
        original = getattr(owner, name)
        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            events[label].append((start, end))
            return result
        replacements.append((owner, name, original))
        setattr(owner, name, timed)

    try:
        for name in ('dual_projection_tiled', 'packed_norm_vector', 'triangle_residual'):
            wrap(extension, name, name)
        wrap(lt_bmm, 'packed_bmm', 'packed_bmm')
        for name, module in model.named_modules():
            parts = name.split('.')
            if len(parts) < 2 or parts[-2] not in ('tri_mul_out', 'tri_mul_in'):
                continue
            if parts[-1] not in ('layer_norm_in', 'linear_z', 'linear_g'):
                continue
            label = parts[-1]
            def before(module, args, name=name):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                pending[name].append(event)
            def after(module, args, result, name=name, label=label):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events[label].append((pending[name].pop(), event))
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, original in reversed(replacements):
            setattr(owner, name, original)
        torch.cuda.synchronize()
        rows = []
        for name, values in events.items():
            times = [start.elapsed_time(end) for start, end in values]
            rows.append(dict(name=name, calls=len(times), total_ms=sum(times),
                             mean_ms=sum(times)/len(times), values_ms=times))
        Path(output, 'base-triangle-profile.json').write_text(json.dumps(dict(
            scope='Inclusive GPU events with host gaps; extension eagerly loaded; instrumented E2E is not benchmark evidence',
            stages=sorted(rows, key=lambda row: -row['total_ms'])), indent=2)+'\n')
