"""Bounded Torch diagnostic; no operator replacement or tensor value capture.

Rust continues to own invocation, artifacts and numerical acceptance. This
observer labels the existing QK/PV einsums and records their actual ATen layouts.
Profiler output is diagnostic, never an E2E performance measurement.
"""
import json
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode


def tensor_metadata(value):
    if not isinstance(value, torch.Tensor):
        return None
    return dict(shape=list(value.shape), strides=list(value.stride()),
                dtype=str(value.dtype), device=str(value.device),
                storage_offset=value.storage_offset())


def trace_operations(data, path):
    """Extract CPU ancestry and explicit CPU-launch/GPU correlation from Kineto.

    Nested TorchDispatchMode does not observe the audited inference_mode path.
    Never change inference mode to make an observer work.
    """
    events = json.loads(Path(path).read_text())['traceEvents']
    selected = {'aten::bmm', 'aten::mm', 'aten::clone', 'aten::copy_',
                'aten::_unsafe_view', 'aten::view'}
    for record in data['einsums']:
        ranges = [e for e in events if e.get('cat') == 'user_annotation'
                  and e.get('name') == 'fireprot.attention.' + record['label']
                  and e.get('ph') == 'X']
        if len(ranges) != 1:
            raise RuntimeError('Ambiguous attention CPU range')
        region = ranges[0]
        operations = []
        for event in events:
            if (event.get('cat') != 'cpu_op' or event.get('name') not in selected
                    or (event.get('pid'), event.get('tid')) != (region['pid'], region['tid'])
                    or event['ts'] < region['ts']
                    or event['ts'] + event.get('dur', 0) > region['ts'] + region['dur'] + 0.001):
                continue
            args = event.get('args', {})
            dims, strides = args.get('Input Dims', []), args.get('Input Strides', [])
            inputs = [dict(shape=shape, strides=strides[i] if i < len(strides) else None)
                      for i, shape in enumerate(dims)]
            external_id = args.get('External id')
            kernels = []
            seen = set()
            if external_id is not None:
                for kernel in events:
                    ka = kernel.get('args', {})
                    if kernel.get('cat') != 'kernel' or ka.get('External id') != external_id:
                        continue
                    correlation = ka.get('correlation')
                    launches = [e for e in events if e.get('cat') == 'cuda_runtime'
                                and e.get('args', {}).get('correlation') == correlation
                                and e.get('args', {}).get('External id') == external_id]
                    if correlation is None or len(launches) != 1:
                        raise RuntimeError('Ambiguous GPU launch correlation')
                    launch = launches[0]
                    identity = (ka.get('device'), ka.get('stream'), correlation)
                    if (identity in seen
                            or (launch.get('pid'), launch.get('tid')) != (event['pid'], event['tid'])
                            or launch['ts'] < event['ts']
                            or launch['ts'] + launch.get('dur', 0) > event['ts'] + event['dur'] + 0.001):
                        raise RuntimeError('Duplicate or unrelated GPU launch correlation')
                    seen.add(identity)
                    kernels.append(dict(name=kernel['name'], correlation=correlation,
                                        device=ka.get('device'), stream=ka.get('stream'),
                                        launch=launches[0]['name']))
            operations.append(dict(operation=event['name'], inputs=inputs,
                                   external_id=external_id, kernels=kernels, completed=True,
                                   metadata_source='kineto_trace'))
        record['operations'] = operations


class _EinsumObserver(TorchFunctionMode):
    def __init__(self, records):
        super().__init__()
        self.records = records

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        labels = {'...qc, ...kc->...qk': 'QK', '...qk, ...kc->...qc': 'PV'}
        label = labels.get(args[0]) if func is torch.einsum and args and isinstance(args[0], str) else None
        if label is None:
            return func(*args, **kwargs)
        record = dict(label=label, equation=args[0],
                      inputs=[tensor_metadata(v) for v in args[1:]],
                      operations=[], completed=False)
        self.records.append(record)
        with torch.profiler.record_function('fireprot.attention.' + label):
            result = func(*args, **kwargs)
        record.update(output=tensor_metadata(result), completed=True)
        return result


class AttentionTrace:
    """Wrap one dispatcher inside its existing source-pinned attention_scope.

    Exactly one matching call is attempted, including when that call raises.
    `occurrence` allows a later call to be selected explicitly; it does not prove
    that tuning/warmup has finished. Output files must not already exist.
    CPU-only mode exists for observer tests, not AMD performance measurements.
    """
    def __init__(self, dispatcher, output, shape, *, occurrence=1, cpu_only=False):
        if len(shape) != 4 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError('Expected an exact positive four-dimensional query shape')
        if type(occurrence) is not int or occurrence <= 0:
            raise ValueError('occurrence must be a positive integer')
        self.dispatcher = dispatcher
        self.output = Path(output)
        self.shape = tuple(shape)
        self.occurrence = occurrence
        self.cpu_only = cpu_only
        self.matches = 0
        self.attempted = False

    def require_complete(self):
        if not self.attempted:
            raise RuntimeError('Requested attention call was not observed')
        data = json.loads((self.output / 'attention-trace.json').read_text())
        if (not data['completed'] or not (self.output / data.get('trace', '')).is_file()
                or [r['label'] for r in data['einsums']] != ['QK', 'PV']
                or any(not r['completed'] for r in data['einsums'])
                or any(not any(op['operation'] == 'aten::bmm' and op['completed']
                               and len(op['inputs']) == 2
                               and all(v['shape'] and v['strides'] is not None for v in op['inputs'])
                               and (self.cpu_only or op['kernels'])
                               for op in r['operations']) for r in data['einsums'])):
            raise RuntimeError('Incomplete QK/PV attention observation')

    def __call__(self, native, query, key, value, biases, use_high_precision=False):
        def invoke():
            return self.dispatcher(native, query, key, value, biases,
                                   use_high_precision=use_high_precision)
        if self.attempted or tuple(query.shape) != self.shape:
            return invoke()
        self.matches += 1
        if self.matches != self.occurrence:
            return invoke()
        if query.dtype != torch.float32 or (not self.cpu_only and not query.is_cuda):
            raise RuntimeError('Trace requires FP32 on the selected device')
        self.output.mkdir(parents=True, exist_ok=True)
        metadata = self.output / 'attention-trace.json'
        trace = self.output / 'attention-trace.chrome.json'
        if metadata.exists() or trace.exists():
            raise FileExistsError('Refusing to overwrite an attention trace')
        self.attempted = True
        data = dict(schema_version=1, completed=False, occurrence=self.matches,
                    cpu_only=self.cpu_only,
                    query=tensor_metadata(query), key=tensor_metadata(key),
                    value=tensor_metadata(value),
                    biases=[tensor_metadata(v) for v in biases], einsums=[],
                    limits=['One selected call; may belong to chunk tuning.',
                            'Instrumentation changes timing; no E2E claim.',
                            'Kernel correlation requires the raw trace, not aggregate names.',
                            'Model output parity must be checked separately.'])
        activities = [torch.profiler.ProfilerActivity.CPU]
        if not self.cpu_only:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile = None
        try:
            with torch.profiler.profile(activities=activities, record_shapes=True,
                                        profile_memory=False, with_stack=False) as profile:
                with _EinsumObserver(data['einsums']):
                    result = invoke()
            data['completed'] = True
            return result
        finally:
            try:
                if profile is not None:
                    profile.export_chrome_trace(str(trace))
                    data['trace'] = trace.name
                    trace_operations(data, trace)
            finally:
                metadata.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
