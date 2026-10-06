"""Inclusive diagnostics of cached global diffusion blocks and their SDPA calls."""
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import torch
import torch.nn.functional as F


@contextmanager
def measure(model, output):
    events = defaultdict(list)
    pending = defaultdict(list)
    handles = []
    original = F.scaled_dot_product_attention
    layouts = {}

    def start(name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        pending[name].append(event)

    def end(name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        events[name].append((pending[name].pop(), event))

    def sdpa(*args, **kwargs):
        q = kwargs.get('query', args[0] if args else None)
        selected = q is not None and tuple(q.shape) == (1, 16, 1000, 48)
        if not selected:
            return original(*args, **kwargs)
        layouts['q'] = dict(shape=list(q.shape), stride=list(q.stride()), dtype=str(q.dtype))
        start('sdpa_core')
        result = original(*args, **kwargs)
        end('sdpa_core')
        return result

    for block in model.diffusion_module.diffusion_transformer.blocks:
        pair = block.attention_pair_bias
        selected = {'attention_total': pair, 'adaptive_norm': pair.layernorm_a,
                    'attention_with_projections': pair.attention,
                    'conditioned_transition': block.conditioned_transition_block}
        for name in ('linear_q', 'linear_k', 'linear_v', 'linear_o', 'linear_g'):
            selected[name] = getattr(pair.attention, name)
        for name, module in selected.items():
            def before(module, args, name=name):
                start(name)
            def after(module, args, result, name=name):
                end(name)
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))
    F.scaled_dot_product_attention = sdpa
    try:
        yield
        required = set(selected) | {'sdpa_core'}
        if any(pending.values()) or any(len(events[name]) != 4800 for name in required):
            raise RuntimeError('Incomplete 24-block D200 global diffusion coverage: ' +
                               str({name: len(events[name]) for name in required}))
        torch.cuda.synchronize()
        rows = []
        for name, pairs in events.items():
            values = [a.elapsed_time(b) for a, b in pairs]
            rows.append(dict(name=name, calls=len(values), total_ms=sum(values)))
        Path(output, 'base-diffusion-global-profile.json').write_text(json.dumps(dict(
            scope='Inclusive event timings with launch gaps; rows overlap; diagnostic only',
            layouts=layouts, rows=sorted(rows, key=lambda r: -r['total_ms'])), indent=2)+'\n')
    finally:
        F.scaled_dot_product_attention = original
        for handle in handles:
            handle.remove()
