"""Bounded Torch diagnostic; no operator replacement or tensor value capture.

Rust continues to own invocation, artifacts and numerical acceptance. This
observer labels the existing QK/PV einsums and records their actual ATen layouts.
Profiler output is diagnostic, never an E2E performance measurement.
"""
import json
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode


def tensor_metadata(value):
    if not isinstance(value, torch.Tensor):
        return None
    return dict(shape=list(value.shape), strides=list(value.stride()),
                dtype=str(value.dtype), device=str(value.device),
                storage_offset=value.storage_offset())


class _LayoutObserver(TorchDispatchMode):
    def __init__(self, records, label):
        super().__init__()
        self.records = records
        self.label = label

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        selected = name in {
            'aten.bmm.default', 'aten.mm.default', 'aten.clone.default',
            'aten.copy_.default', 'aten._unsafe_view.default', 'aten.view.default',
        }
        record = None
        if selected:
            record = dict(label=self.label, operation=name,
                          inputs=[tensor_metadata(v) for v in args], completed=False)
            self.records.append(record)
        result = func(*args, **(kwargs or {}))
        if record is not None:
            record.update(output=tensor_metadata(result), completed=True)
        return result


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
            with _LayoutObserver(record['operations'], label):
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
                or any(not any(op['operation'] == 'aten.bmm.default' and op['completed']
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
            finally:
                metadata.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
