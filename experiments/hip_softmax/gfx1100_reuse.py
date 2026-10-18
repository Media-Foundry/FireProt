"""Experimental composition of source-pinned rollout-local reuse on gfx1100."""
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path


@contextmanager
def reuse_scope(model, output):
    import torch
    from .pair_cache import pair_cache
    from .atom_cache import atom_cache
    from .template_reuse import template_reuse_scope
    mode = os.environ['FIREPROT_GFX1100_REUSE_MODE']
    if mode not in ('0', '1', '2', '3'):
        raise ValueError('Experimental reuse mode must be 0, 1, 2 or 3')
    if torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(':')[0] != 'gfx1100':
        raise RuntimeError('This experimental composition requires gfx1100')
    report = dict(schema_version=1, mode=int(mode), completed=False)
    reports = {}
    torch.cuda.reset_peak_memory_stats()
    try:
        with ExitStack() as stack:
            if int(mode) >= 1:
                reports['pair'] = stack.enter_context(pair_cache(model.sample_diffusion, output))
            if int(mode) >= 2:
                reports['atom'] = stack.enter_context(atom_cache(model.sample_diffusion, output))
            if int(mode) >= 3:
                stack.enter_context(template_reuse_scope(model, output,
                    'f75d982e7c94462a0276b22c860c69676baab003556d2c665a5ab9db7bbb9d63'))
            yield
        for name, item in reports.items():
            scopes = item['scopes']
            if len(scopes) != 1 or not scopes[0]['completed']:
                raise RuntimeError(f'{name} cache did not complete exactly one rollout')
            if any(scopes[0][key] != expected for key, expected in
                   [('calls', 200), ('hits', 199), ('misses', 1), ('fallbacks', 0)]):
                raise RuntimeError(f'{name} cache did not execute the qualified 200-call pattern')
        if int(mode) >= 3:
            template = json.loads(Path(output, 'template-reuse.json').read_text())
            if template['counts']['reused_slots'] <= 0:
                raise RuntimeError('Template reuse did not execute')
            report['template'] = template
        report['completed'] = True
    finally:
        report.update(caches=reports, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        Path(output, 'reuse-runtime.json').write_text(json.dumps(report, indent=2) + '\n')
