"""Experimental worker-only CPU affinity; no system or memory policy changes."""
import json
import os
from pathlib import Path
import time


def expand(text):
    result = set()
    for part in text.strip().split(','):
        bounds = part.split('-')
        low = int(bounds[0])
        high = int(bounds[-1])
        if len(bounds) > 2 or low < 0 or high < low:
            raise ValueError('Invalid sysfs CPU range')
        result.update(range(low, high+1))
    return result


def begin(request):
    flag = os.environ.get('FIREPROT_ESMC_CPU_LOCAL', '0')
    if flag not in ('0', '1'):
        raise ValueError('FIREPROT_ESMC_CPU_LOCAL must be 0 or 1')
    if request['workload']['model'] != 'plm' or request['workload']['model_name'] != 'esmc-fold2-6b' or request['hip_device'] != 6:
        raise ValueError('This affinity experiment requires standalone ESMC on HIP6')
    if len(list(Path('/proc/self/task').iterdir())) != 1:
        raise RuntimeError('Apply affinity before importing threaded runtimes')
    pci = Path('/sys/bus/pci/devices/0000:8e:00.0')
    node = int((pci/'numa_node').read_text())
    local = expand((pci/'local_cpulist').read_text())
    original = os.sched_getaffinity(0)
    if node != 1 or local != set(range(64, 128)) | set(range(192, 256)) or not local <= original:
        raise RuntimeError('CPU topology or allowed CPU mask differs from the experiment')
    if flag == '1':
        os.sched_setaffinity(0, local)
    effective = os.sched_getaffinity(0)
    if effective != (local if flag == '1' else original):
        raise RuntimeError('Requested CPU affinity was not applied')
    return dict(schema_version=1, mode=flag, gpu_pci='0000:8e:00.0', gpu_numa_node=node,
                original_cpus=sorted(original), effective_cpus=sorted(effective),
                gpu_local_cpus=sorted(local), memory_policy='unchanged', placement_seconds=None)


def instrument_loading(diagnostic):
    from . import plm_loading
    original = plm_loading.streamed_esmc_state
    def timed(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        diagnostic['placement_seconds'] = time.perf_counter()-start
        return result
    plm_loading.streamed_esmc_state = timed


def finish(diagnostic):
    effective = sorted(os.sched_getaffinity(0))
    if effective != diagnostic['effective_cpus']:
        raise RuntimeError('Main worker affinity changed during inference')
    masks = {}
    for task in Path('/proc/self/task').iterdir():
        try:
            mask = ','.join(map(str, sorted(os.sched_getaffinity(int(task.name)))))
        except ProcessLookupError:
            continue
        masks[mask] = masks.get(mask, 0)+1
    diagnostic['thread_masks_at_finish'] = masks
    Path(os.environ['FIREPROT_OUTPUT_DIR']).joinpath('cpu-placement.json').write_text(
        json.dumps(diagnostic, indent=2)+'\n')
