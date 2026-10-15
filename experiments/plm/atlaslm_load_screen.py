"""AtlasLM eager checkpoint deserialization screen with exact output checks.

The upstream constructor and strict load_state_dict remain unchanged. Only the
checkpoint's mmap/map_location options differ in this diagnostic. Each invocation
must use a fresh output directory and the pinned AtlasLM environment/source.
"""
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from fireprot_workers.plm import load_model
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.transport import save_arrays

request = json.loads(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2]); output.mkdir(exist_ok=False)
mode = sys.argv[3]
reference = Path(sys.argv[4])
assert request['workload']['model_name'] == 'atlaslm-3b'
assert mode in ('native', 'eager_cpu', 'eager_gpu')
request['backend'] = 'native'
torch.set_num_threads(8); torch.manual_seed(request['seed'])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
_, sequence = read_sequence(request['input']['path'])
checkpoint = Path(request['checkpoints']['model']['path']).resolve()
original_load = torch.load
loads = []

def load(path, *args, **kwargs):
    assert Path(path).resolve() == checkpoint
    if mode != 'native':
        kwargs['mmap'] = False
        kwargs['map_location'] = 'cuda' if mode == 'eager_gpu' else 'cpu'
    start = time.perf_counter()
    state = original_load(path, *args, **kwargs)
    loads.append(dict(seconds=time.perf_counter()-start,
                      mmap=kwargs.get('mmap'), map_location=kwargs.get('map_location')))
    return state

start = time.perf_counter()
with patch.object(torch, 'load', load):
    model, tokens, forward = load_model(request, sequence)
torch.cuda.synchronize()
loading = time.perf_counter() - start
loading_peak = torch.cuda.max_memory_allocated()
loading_reserved = torch.cuda.max_memory_reserved()
with torch.inference_mode():
    start = time.perf_counter(); arrays = forward(); torch.cuda.synchronize()
    first = time.perf_counter()-start
    save_arrays(output/'sample.npz', arrays)
    warm = []
    for _ in range(5):
        start = time.perf_counter(); forward(); torch.cuda.synchronize()
        warm.append(time.perf_counter()-start)
    if mode == 'native':
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as prof:
            forward(); torch.cuda.synchronize()
        prof.export_chrome_trace(str(output/'forward-trace.json'))
        (output/'profile.txt').write_text(prof.key_averages().table(
            sort_by='self_cuda_time_total', row_limit=25))
with np.load(reference, allow_pickle=False) as expected, np.load(output/'sample.npz', allow_pickle=False) as actual:
    assert set(expected.files) == set(actual.files)
    exact = {key: actual[key].dtype == expected[key].dtype and
             actual[key].shape == expected[key].shape and
             actual[key].tobytes() == expected[key].tobytes() for key in expected.files}
report = dict(mode=mode, load_seconds=loading, checkpoint_loads=loads,
              loading_peak_bytes=loading_peak, loading_reserved_bytes=loading_reserved,
              first_forward_seconds=first, warm_forward_seconds=warm,
              exact=exact, hip_visible_devices=os.environ['HIP_VISIBLE_DEVICES'])
(output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
if not all(exact.values()):
    raise RuntimeError('AtlasLM checkpoint loading changed the native outputs')
