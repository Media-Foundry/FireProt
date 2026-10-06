"""Reuse qualified HIP elementwise kernels in Base diffusion transitions."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import torch


@contextmanager
def install(model, output):
    mode = os.environ.get('FIREPROT_BASE_DIFFUSION_TRANSITION', 'off')
    if mode not in ('off', 'on'):
        raise ValueError('Unknown diffusion transition mode')
    if mode == 'off':
        yield
        return
    if (model.training or not torch.version.hip
            or not torch.are_deterministic_algorithms_enabled()
            or not torch.utils.deterministic.fill_uninitialized_memory):
        raise RuntimeError('Strict deterministic HIP inference required')
    from fireprot.backends.amd.extension import load_amd_extension
    ext = load_amd_extension()
    if ext is None:
        raise RuntimeError('Qualified AMD extension unavailable')
    diffusion = model.diffusion_module
    groups = [('global', list(diffusion.diffusion_transformer.blocks)),
              ('local', list(diffusion.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks)
               + list(diffusion.atom_attention_decoder.atom_transformer.diffusion_transformer.blocks))]
    if [len(blocks) for _, blocks in groups] != [24, 6]:
        raise RuntimeError('Expected Base diffusion block geometry')
    previous = []
    counts = {}
    records = dict(mode=mode, calls=counts, error=None,
                   kernels=['silu_mul_f32', 'sigmoid_mul_f32'])
    try:
        for group, blocks in groups:
            for index, block in enumerate(blocks):
                module = block.conditioned_transition_block
                name = f'{group}/{index}'
                counts[name] = 0
                width, tokens = (768, 1000) if group == 'global' else (128, 7979)
                if module.c_a != width or module.n != 2 or module.training:
                    raise RuntimeError('Unsupported transition configuration')
                previous.append((module, module.__dict__.get('forward')))

                def forward(a, s, module=module, name=name, width=width, tokens=tokens):
                    if (torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
                            or torch.cuda.is_current_stream_capturing()
                            or not a.is_cuda or a.device != s.device
                            or a.dtype != torch.float32 or s.dtype != torch.float32
                            or tuple(a.shape) != (1, tokens, width)):
                        raise RuntimeError('Transition input outside qualified contract')
                    normalized = module.adaln(a, s)
                    activation = module.linear_nobias_a1(normalized)
                    gate = module.linear_nobias_a2(normalized)
                    hidden = ext.silu_mul_f32(activation, gate)
                    output_gate = module.linear_s(s)
                    projected = module.linear_nobias_b(hidden)
                    result = ext.sigmoid_mul_f32(output_gate, projected)
                    counts[name] += 1
                    return result

                module.forward = forward
        yield
        if any(value != 200 for value in counts.values()):
            raise RuntimeError('Incomplete D200 transition fusion coverage')
    except BaseException as error:
        records['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        for module, value in previous:
            if value is None:
                delattr(module, 'forward')
            else:
                module.forward = value
        Path(output, 'base-diffusion-transition.json').write_text(json.dumps(records, indent=2)+'\n')
