"""Experimental per-block CPU BF16 conversion and bounded asynchronous H2D."""
from collections import deque
from contextlib import ExitStack


def cpu_streamed_esmc_state(root, mapping, published_to_native, *, pinned=False):
    import torch
    from safetensors import safe_open
    pending = deque()
    with ExitStack() as stack:
        tensors = {}
        for shard in sorted({v for k,v in mapping.items() if k.startswith('esmc.')}):
            handle = stack.enter_context(safe_open(root/shard, framework='pt', device='cpu'))
            for key in handle.keys():
                if key.startswith('esmc.'):
                    tensors[key] = handle
        groups = {}
        for key in tensors:
            pieces = key.split('.')
            group = '.'.join(pieces[:3]) if len(pieces)>3 and pieces[1]=='layers' else 'root'
            groups.setdefault(group,[]).append(key)
        state = {}
        try:
            for keys in groups.values():
                if len(pending) >= 2:
                    event, held = pending.popleft()
                    event.synchronize()
                    del held
                raw = {}; staging = []
                for key in keys:
                    tensor = tensors[key].get_tensor(key)
                    host = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                    if pinned:
                        host = host.pin_memory()
                        staging.append(host)
                    raw[key[5:]] = host.to(device='cuda:0', non_blocking=pinned)
                converted = published_to_native(raw)
                if state.keys() & converted.keys():
                    raise ValueError('ESMC block conversion produced overlapping state keys')
                state.update(converted)
                if pinned:
                    event = torch.cuda.Event()
                    event.record()
                    pending.append((event, staging))
                del raw, converted, tensor, host, staging
        finally:
            if pinned:
                torch.cuda.current_stream().synchronize()
                pending.clear()
    return state


def reused_pinned_esmc_state(root, mapping, published_to_native):
    """Cast directly into two reusable pinned block buffers before async H2D."""
    import torch
    from safetensors import safe_open
    with ExitStack() as stack:
        tensors = {}
        for shard in sorted({v for k,v in mapping.items() if k.startswith('esmc.')}):
            handle = stack.enter_context(safe_open(root/shard, framework='pt', device='cpu'))
            for key in handle.keys():
                if key.startswith('esmc.'):
                    tensors[key] = handle
        groups = {}
        for key in tensors:
            pieces = key.split('.')
            group = '.'.join(pieces[:3]) if len(pieces)>3 and pieces[1]=='layers' else 'root'
            groups.setdefault(group,[]).append(key)
        slots = [[], []]
        events = [None, None]
        state = {}
        try:
            for index, keys in enumerate(groups.values()):
                slot = index % 2
                if events[slot] is not None:
                    events[slot].synchronize()
                buffers = slots[slot]
                raw = {}
                for position, key in enumerate(keys):
                    tensor = tensors[key].get_tensor(key)
                    dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
                    if position == len(buffers):
                        buffers.append(torch.empty(tensor.shape, dtype=dtype, pin_memory=True))
                    elif buffers[position].shape != tensor.shape or buffers[position].dtype != dtype:
                        buffers[position] = torch.empty(tensor.shape, dtype=dtype, pin_memory=True)
                    buffers[position].copy_(tensor)
                    raw[key[5:]] = buffers[position].to(device='cuda:0', non_blocking=True)
                del buffers[len(keys):]
                converted = published_to_native(raw)
                if state.keys() & converted.keys():
                    raise ValueError('ESMC block conversion produced overlapping state keys')
                state.update(converted)
                events[slot] = torch.cuda.Event()
                events[slot].record()
                del raw, converted, tensor
        finally:
            torch.cuda.current_stream().synchronize()
    return state
