"""Tensor-only checkpoint placement for the qualified standalone ESMC route."""
from contextlib import ExitStack


def _gpu_streamed_esmc_state(root, mapping, published_to_native):
    """Convert one block at a time with the native, exact row-concatenation map."""
    import torch
    from safetensors import safe_open
    with ExitStack() as stack:
        tensors = {}
        for shard in sorted({v for k,v in mapping.items() if k.startswith('esmc.')}):
            handle = stack.enter_context(safe_open(root/shard, framework='pt', device='cuda:0'))
            for key in handle.keys():
                if key.startswith('esmc.'):
                    tensors[key] = handle
        groups = {}
        for key in tensors:
            pieces = key.split('.')
            group = '.'.join(pieces[:3]) if len(pieces)>3 and pieces[1]=='layers' else 'root'
            groups.setdefault(group,[]).append(key)
        state = {}
        for keys in groups.values():
            raw = {}
            for key in keys:
                tensor = tensors[key].get_tensor(key)
                raw[key[5:]] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            converted = published_to_native(raw)
            if state.keys() & converted.keys():
                raise ValueError('ESMC block conversion produced overlapping state keys')
            state.update(converted)
            del raw, converted, tensor
    return state


def streamed_esmc_state(root, mapping, published_to_native):
    import os
    mode = os.environ.get('FIREPROT_ESMC_CPU_LOADING', '0')
    if mode == '0':
        return _gpu_streamed_esmc_state(root, mapping, published_to_native)
    if mode != '1':
        raise ValueError('FIREPROT_ESMC_CPU_LOADING must be 0 or 1')
    from .esmc_cpu_loading import reused_pinned_esmc_state
    return reused_pinned_esmc_state(root, mapping, published_to_native)
