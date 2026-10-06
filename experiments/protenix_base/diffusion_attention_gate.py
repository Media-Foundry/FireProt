"""Experimental global diffusion output gating with fused layout conversion."""
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import torch
from fireprot.backends.amd._published import select_c


@contextmanager
def install(model, output):
    mode = os.environ.get('FIREPROT_BASE_DIFFUSION_GATE_MODE', 'off')
    if mode not in ('off', 'on'):
        raise ValueError('Unknown diffusion gate mode')
    if mode == 'off':
        yield
        return
    if (model.training or not torch.version.hip
            or not torch.are_deterministic_algorithms_enabled()
            or not torch.utils.deterministic.fill_uninitialized_memory
            or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx90a'):
        raise RuntimeError('Strict gfx90a inference required')
    selected = select_c('fireprot_base_diffusion_gate', 'FIREPROT_BASE_DIFFUSION_GATE',
                        'base-diffusion-gate-v1')
    if selected is None:
        raise RuntimeError('Published diffusion gate kernel required')
    lib, receipt = selected
    entry = lib.base_diffusion_gate
    entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                      ctypes.c_int, ctypes.c_void_p]
    entry.restype = ctypes.c_int
    blocks = list(model.diffusion_module.diffusion_transformer.blocks)
    if len(blocks) != 24:
        raise RuntimeError('Expected 24 global diffusion blocks')
    previous = []
    counts, layouts = {}, {}
    records = dict(mode=mode, calls=counts, layouts=layouts, published=receipt, error=None)
    try:
        for index, block in enumerate(blocks):
            module = block.attention_pair_bias.attention
            if module.num_heads != 16 or module.c_hidden != 48 or module.linear_g is None:
                raise RuntimeError('Unsupported diffusion attention geometry')
            previous.append((module, module.__dict__.get('_wrap_up')))
            counts[str(index)] = 0

            def wrap(o, q_x, module=module, index=index):
                if (torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
                        or torch.cuda.is_current_stream_capturing()
                        or not o.is_cuda or o.device != q_x.device
                        or o.dtype != torch.float32 or q_x.dtype != torch.float32
                        or tuple(o.shape) != (1, 1000, 16, 48)
                        or tuple(q_x.shape) != (1, 1000, 768)
                        or o.stride(3) != 1
                        or (o.stride(1), o.stride(2)) not in ((768, 48), (48, 48000))):
                    raise RuntimeError('Attention gate input outside qualified contract')
                gate = module.linear_g(q_x)
                if not gate.is_contiguous() or tuple(gate.shape) != (1, 1000, 768):
                    raise RuntimeError('Expected fresh contiguous gate projection')
                with torch.cuda.device(o.device):
                    status = entry(o.data_ptr(), gate.data_ptr(), o.stride(1), o.stride(2),
                                   torch.cuda.current_stream(o.device).cuda_stream)
                if status:
                    raise RuntimeError(f'Diffusion gate HIP status {status}')
                counts[str(index)] += 1
                layout = str(tuple(o.stride()))
                layouts[layout] = layouts.get(layout, 0) + 1
                return module.linear_o(gate)

            module._wrap_up = wrap
        yield
        if any(value != 200 for value in counts.values()):
            raise RuntimeError('Incomplete global gate fusion coverage')
    except BaseException as error:
        records['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        for module, value in previous:
            if value is None:
                delattr(module, '_wrap_up')
            else:
                module._wrap_up = value
        Path(output, 'base-diffusion-gate.json').write_text(json.dumps(records, indent=2)+'\n')
