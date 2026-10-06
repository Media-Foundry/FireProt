"""Opt-in module event profiling for a frozen Rust-supervised Base worker.

GPU event intervals include host launch gaps and nested intervals overlap.
These diagnostic timings must not be presented as uninstrumented E2E results.
"""
from contextlib import contextmanager
from collections import defaultdict
import json
from pathlib import Path
import torch


@contextmanager
def measure(model, output):
    events = defaultdict(list)
    pending = defaultdict(list)
    handles = []
    inventory = []
    leaf_names = {'tri_mul_out', 'tri_mul_in', 'tri_att_start', 'tri_att_end',
                  'pair_transition', 'single_transition', 'attention_pair_bias'}
    stages = {'pairformer_stack', 'diffusion_module', 'confidence_head',
              'input_embedder', 'msa_module', 'template_embedder'}
    for name, module in model.named_modules():
        tail = name.rsplit('.', 1)[-1]
        if name not in stages and tail not in leaf_names:
            continue
        group = name if name in stages else name.split('.')[0] + '/' + tail
        inventory.append({'name': name, 'class': type(module).__name__, 'group': group})
        def before(module, args, name=name):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            pending[name].append(event)
        def after(module, args, result, name=name, group=group):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events[group].append((pending[name].pop(), event))
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        torch.cuda.synchronize()
        rows = []
        for name, pairs in events.items():
            values = [start.elapsed_time(end) for start, end in pairs]
            rows.append(dict(name=name, calls=len(values), total_ms=sum(values),
                             mean_ms=sum(values)/len(values), values_ms=values))
        Path(output, 'base-stage-profile.json').write_text(json.dumps(dict(
            scope='Inclusive GPU event intervals with launch gaps; nested rows overlap; diagnostic only',
            stages=sorted(rows, key=lambda row: -row['total_ms']), inventory=inventory), indent=2)+'\n')
