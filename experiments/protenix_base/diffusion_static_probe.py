"""Inclusive event diagnostics for diffusion after static attention-bias caching."""
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import torch


@contextmanager
def measure(model, output):
    root = model.diffusion_module
    selected = {
        'denoiser': root,
        'conditioning': root.diffusion_conditioning,
        'global_transformer': root.diffusion_transformer,
        'atom_encoder': root.atom_attention_encoder,
        'atom_decoder': root.atom_attention_decoder,
    }
    encoder = root.atom_attention_encoder
    for name in ('atom_transformer', 'small_mlp', 'linear_no_bias_cl',
                 'linear_no_bias_cm', 'linear_no_bias_s', 'linear_no_bias_r',
                 'layernorm_s', 'linear_no_bias_q'):
        selected['encoder/' + name] = getattr(encoder, name)
    events = defaultdict(list)
    pending = defaultdict(list)
    handles = []
    for name, module in selected.items():
        def before(module, args, name=name):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            pending[name].append(event)
        def after(module, args, result, name=name):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events[name].append((pending[name].pop(), event))
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    try:
        yield
        if any(pending.values()) or any(len(events[name]) != 200 for name in selected):
            raise RuntimeError('Incomplete D200 diffusion profile coverage')
        torch.cuda.synchronize()
        rows = []
        for name, pairs in events.items():
            values = [begin.elapsed_time(end) for begin, end in pairs]
            rows.append(dict(name=name, calls=len(values), total_ms=sum(values),
                             values_ms=values))
        Path(output, 'base-diffusion-static-profile.json').write_text(json.dumps(dict(
            scope='Inclusive GPU event intervals including launch gaps; nested rows overlap; not E2E speedup',
            rows=sorted(rows, key=lambda row: -row['total_ms'])), indent=2)+'\n')
    finally:
        for handle in handles:
            handle.remove()
