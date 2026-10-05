"""Diagnostic Torch boundary; execution, isolation and acceptance remain in Rust.

Reports inclusive stage events and Torch profiler observations. Instrumented
latencies are not performance comparisons. Does not change scientific settings.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time


@contextmanager
def model_probe(model, model_name, output):
    import torch
    originals, events, observations = [], [], []
    selected = 'get_pairformer_output' if model_name == 'protenix' else 'run_trunk'
    methods = ([selected, 'sample_diffusion', 'run_confidence_head']
               if model_name == 'protenix' else [selected, '_rollout'])
    profiling = False

    def wrap(obj, name, label, capture=False):
        original = getattr(obj, name)
        had_local = name in obj.__dict__
        local = obj.__dict__.get(name)
        def call(*args, **kwargs):
            nonlocal profiling
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                if capture and not profiling:
                    profiling = True
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True, profile_memory=False, with_stack=False,
                    ) as profile:
                        result = original(*args, **kwargs)
                    # Export Torch's observed statistics without ranking or promoting kernels.
                    for event in profile.key_averages(group_by_input_shape=True):
                        observations.append(dict(name=event.key, count=event.count,
                            input_shapes=event.input_shapes,
                            self_cpu_us=event.self_cpu_time_total,
                            cpu_us=event.cpu_time_total,
                            self_device_us=event.self_device_time_total,
                            device_us=event.device_time_total))
                    return result
                return original(*args, **kwargs)
            finally:
                end.record()
                events.append((label, start, end))
        setattr(obj, name, call)
        originals.append((obj, name, had_local, local))

    try:
        for name in methods:
            wrap(model, name, name, name == selected)
        for name in ('input_embedder', 'template_embedder', 'msa_module', 'pairformer_stack'):
            if hasattr(model, name):
                wrap(getattr(model, name), 'forward', name)
        if model_name == 'openfold3':
            for name in ('sample_diffusion', 'aux_heads'):
                wrap(getattr(model, name), 'forward', name)
        started = time.perf_counter()
        yield
        torch.cuda.synchronize()
        data = dict(schema_version=1, model=model_name, profiler_scope=selected,
            instrumented_wall_seconds=time.perf_counter()-started,
            stage_events=[dict(stage=n, inclusive_ms=a.elapsed_time(b)) for n,a,b in events],
            torch_operations=observations,
            limits=['Stage timings are inclusive and overlap; do not sum nested stages.',
                    'Profiler times are diagnostic and include instrumentation overhead.',
                    'Torch input shapes do not establish strides or backend eligibility.',
                    'Profiler covers trunk; diffusion/confidence currently have stage timings only.'])
        Path(output, 'operator-profile.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    finally:
        for obj, name, had_local, local in reversed(originals):
            if had_local:
                setattr(obj, name, local)
            else:
                delattr(obj, name)
