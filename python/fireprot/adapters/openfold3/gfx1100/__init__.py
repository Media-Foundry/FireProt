"""Source-pinned gfx1100 OpenFold3 composition with explicit native fallbacks."""
from contextlib import ExitStack
import json
from pathlib import Path

import torch

from fireprot.adapters.openfold3.attention import attention_scope
from fireprot.backends.amd.softmax_gfx1100 import Gfx1100Softmax
from .dispatch import Gfx1100Dispatcher
from .pair_cache import pair_cache
from .atom_cache import atom_cache
from .template_reuse import template_reuse_scope


def forward(model, original, args, kwargs, output):
    """Keep upstream arithmetic; scope retained features to this model call."""
    batch = args[0] if args else kwargs['batch']
    length = int(batch['token_mask'].shape[-1])
    shared = model.config.architecture.shared
    reuse = (length in (1000, 1536) and shared.num_recycles == 3
        and shared.diffusion.no_full_rollout_steps == 200
        and shared.diffusion.no_full_rollout_samples == 1)
    dispatcher = Gfx1100Dispatcher(Gfx1100Softmax())
    output = Path(output)
    report = dict(schema_version=1, token_count=length, completed=False,
        reuse_enabled=reuse, fallback_reason=None if reuse else 'Unqualified reuse length/workload; native feature computations retained',
        caches={})
    torch.cuda.reset_peak_memory_stats()
    try:
        with ExitStack() as stack:
            stack.enter_context(attention_scope(dispatcher))
            if reuse:
                report['caches']['pair'] = stack.enter_context(pair_cache(model.sample_diffusion, output))
                report['caches']['atom'] = stack.enter_context(atom_cache(model.sample_diffusion, output))
                stack.enter_context(template_reuse_scope(model, output,
                    'f75d982e7c94462a0276b22c860c69676baab003556d2c665a5ab9db7bbb9d63'))
            result = original(model, *args, **kwargs)
        if reuse:
            report['template'] = json.loads((output / 'template-reuse.json').read_text())
        report['completed'] = True
        return result
    finally:
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (output / 'gfx1100-composition.json').write_text(json.dumps(report, indent=2) + '\n')
        (output / 'operator-runtime.json').write_text(json.dumps(dispatcher.manifest(), indent=2) + '\n')
