"""Diagnostic fresh-process loader/head trial; not packaged Rust E2E evidence.

Run with the model's pinned source and worker directories on PYTHONPATH.
The candidate is injected into an isolated function namespace, never upstream.
"""
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import fireprot_workers.plm as worker
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.transport import save_arrays

request = json.loads(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2]); output.mkdir(exist_ok=False)
candidate = sys.argv[3] != 'native'
packed = sys.argv[3] == 'packed'
reference = Path(sys.argv[4])
# These screens compare transfers within the local loader. The independent
# official-factory E2E experiment uses prepare_esm3_loader_pairs.py instead.
os.environ['FIREPROT_ESM3_REFERENCE_FACTORY'] = '0'
torch.set_num_threads(8)
torch.manual_seed(request['seed'])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
_, sequence = read_sequence(request['input']['path'])
loader = worker.load_model

def packed_transfer(state):
    """Pack each native dtype on the host and transfer one contiguous slab."""
    groups = {}
    result = {}
    for key, value in state.items():
        groups.setdefault(value.dtype, []).append((key, value))
    for dtype, entries in groups.items():
        total = sum(value.numel() for _, value in entries)
        host = torch.empty(total, dtype=dtype, pin_memory=True)
        offset = 0
        for _, value in entries:
            host[offset:offset+value.numel()].copy_(value.reshape(-1))
            offset += value.numel()
        device = host.to('cuda', non_blocking=True)
        offset = 0
        for key, value in entries:
            result[key] = device[offset:offset+value.numel()].view(value.shape)
            offset += value.numel()
        torch.cuda.synchronize()
    return result

if candidate:
    source = inspect.getsource(loader)
    old = "model = load_atlas(checkpoint, device='cuda').eval()"
    new = """from atlaslm.model import AtlasLM
        with torch.device('meta'):
            model = AtlasLM(2304, 36, 48).to(torch.bfloat16)
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        state = {k: v.to(device='cuda', dtype=torch.bfloat16, non_blocking=True)
                 for k, v in weights.items()}
        model.load_state_dict(state, strict=True, assign=True)
        for module in model.modules():
            if hasattr(module, 'init_buffers'):
                module.init_buffers(device=torch.device('cuda'))
        model.eval()"""
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = "raw[key[5:]] = handle.get_tensor(key)"
    assert source.count(old) == 1
    if packed:
        source = source.replace("model.load_state_dict(published_to_native(raw), strict=True, assign=True)",
                                "model.load_state_dict(published_to_native(packed_transfer(raw)), strict=True, assign=True)")
        source = source.replace("state = {k: v.to(device='cuda', dtype=torch.bfloat16, non_blocking=True)\n                 for k, v in weights.items()}",
                                "state = packed_transfer(weights)")
        source = source.replace("model.load_state_dict(weights, strict=True, assign=True)",
                                "model.load_state_dict(packed_transfer(weights), strict=True, assign=True)")
    else:
        source = source.replace(old, old + ".to(device='cuda', dtype=torch.bfloat16, non_blocking=True)")
    source = source.replace("model.to(device='cuda', dtype=torch.bfloat16)",
                            "model.to(device='cuda', dtype=torch.bfloat16, non_blocking=True)")
    namespace = dict(vars(worker))
    namespace['packed_transfer'] = packed_transfer
    exec(compile(source, '<loading-screen>', 'exec'), namespace)
    loader = namespace['load_model']
started = time.perf_counter()
model, tokens, forward = loader(request, sequence)
torch.cuda.synchronize()
loaded = time.perf_counter() - started
if candidate and request['workload']['model_name'] == 'esm3-open':
    def sequence_head(x, embed, attentions=None):
        return SimpleNamespace(sequence_logits=model.output_heads.sequence_head(x),
                               embeddings=embed)
    model.output_heads.forward = sequence_head
with torch.inference_mode():
    started = time.perf_counter()
    arrays = forward(); torch.cuda.synchronize()
    first = time.perf_counter() - started
    save_arrays(output/'sample.npz', arrays)
    times = []
    for _ in range(5):
        started = time.perf_counter(); forward(); torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
with np.load(reference, allow_pickle=False) as native, np.load(output/'sample.npz', allow_pickle=False) as actual:
    exact = {key: native[key].dtype == actual[key].dtype and
             native[key].shape == actual[key].shape and
             native[key].tobytes() == actual[key].tobytes() for key in native.files}
    assert set(native.files) == set(actual.files)
report = dict(model=request['workload']['model_name'], candidate=candidate, packed=packed,
              load_seconds=loaded, first_forward_seconds=first,
              warm_forward_seconds=times, exact=exact)
(output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
if not all(exact.values()):
    raise RuntimeError('Loader/head trial changed a selected public output')
