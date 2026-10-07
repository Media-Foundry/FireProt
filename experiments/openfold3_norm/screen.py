"""Exact-array and complete-operator screen; no model dispatch changes.

Only gfx90a is supported. Run after all active model timing jobs have finished.
The output records failures as well as successes; only exact variants are timed.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F
import torch.utils.deterministic


class Kernel:
    def __init__(self, library, width):
        self.library = ctypes.CDLL(str(library))
        self.entry = self.library.fireprot_norm128
        self.entry.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64,
            ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        self.entry.restype = ctypes.c_int
        self.width = width

    def __call__(self, x, weight, bias, *, stats=False):
        if (x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous()
                or x.shape[-1] != 128 or x.data_ptr() % 16):
            raise ValueError('Expected aligned contiguous HIP FP32 C128 input')
        for parameter in (weight, bias):
            if parameter is not None and (
                parameter.shape != (128,) or parameter.dtype != x.dtype
                or parameter.device != x.device or not parameter.is_contiguous()
                or parameter.data_ptr() % 16):
                raise ValueError('Expected aligned contiguous FP32 affine parameters')
        # Native LayerNorm returns canonical contiguous strides, including
        # singleton dimensions. empty_like can preserve a sliced batch stride;
        # equal values then take a different downstream Linear execution path.
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        rows = x.numel() // 128
        mean = torch.empty(rows, device=x.device, dtype=x.dtype) if stats else None
        inverse = torch.empty_like(mean) if stats else None
        ptr = lambda t: None if t is None else t.data_ptr()
        status = self.entry(ptr(x), ptr(weight), ptr(bias), ptr(out), ptr(mean),
            ptr(inverse), rows, 1e-5, self.width,
            torch.cuda.current_stream(x.device).cuda_stream)
        if status:
            raise RuntimeError(f'HIP normalization status {status}')
        return (out, mean, inverse) if stats else out


def compare(actual, expected):
    return dict(bitwise_equal=torch.equal(actual.reshape(-1).view(torch.int32),
                                         expected.reshape(-1).view(torch.int32)),
                max_abs_delta=float((actual.reshape(-1) - expected.reshape(-1)).abs().max()))


def elapsed(function):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    value = function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def verified_captures(directory, manifest=None):
    manifest = Path(manifest) if manifest else Path(__file__).with_name('captures.json')
    pins = json.loads(manifest.read_text())
    paths = sorted(Path(directory).glob('trunk_pairformer-tri_att_*.pt'))
    if len(pins) != 2 or {p.name for p in paths} != set(pins):
        raise ValueError('Both pinned trained attention captures are required')
    for path in paths:
        expected = pins[path.name]
        if path.stat().st_size != expected['size']:
            raise ValueError(f'Trained capture size differs: {path.name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected['sha256']:
            raise ValueError(f'Trained capture hash differs: {path.name}')
    return paths, pins


@torch.inference_mode()
def main(build, destination, captures=None):
    capture_paths, capture_pins = verified_captures(captures) if captures else ([], {})
    if 'gfx90a' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('This experiment only supports gfx90a')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(101)
    out = Path(destination)
    out.mkdir(exist_ok=False)
    receipt = json.loads(Path(build).read_text())
    library = Path(receipt['library'])
    assert hashlib.sha256(library.read_bytes()).hexdigest() == receipt['library_sha256']
    kernels = {w: Kernel(library, w) for w in (64, 32)}
    checks, timing = [], []

    def check(name, x, weight, bias):
        expected = torch.native_layer_norm(x, (128,), weight, bias, 1e-5)
        for width, kernel in kernels.items():
            actual = kernel(x, weight, bias, stats=True)
            report = {key: compare(a, b) for key, a, b in zip(
                ('output', 'mean', 'rstd'), actual, expected)}
            checks.append(dict(name=name, width=width, shape=list(x.shape),
                               arrays=report, passed=all(v['bitwise_equal'] for v in report.values())))

    for rows in (17, 1024, 65536):
        x = torch.randn(rows, 128, device='cuda')
        weight = torch.randn(128, device='cuda')
        bias = torch.randn(128, device='cuda')
        check(f'random-{rows}', x, weight, bias)
        if rows == 1024:
            check('plain', x, None, None)
            check('scale-only', x, weight, None)
            check('bias-only', x, None, bias)
            check('constant', torch.full_like(x, 0.25), weight, bias)
            check('low-variance', x * 1e-4 + 1, weight, bias)
            check('large', x * 1e10, weight, bias)
            x.zero_()
            x[:, ::2] = -0.0
            check('signed-zero', x, weight, bias)

    if captures:
        for capture in capture_paths:
            snap = torch.load(capture, map_location='cpu', weights_only=True, mmap=True)
            x = snap['z'].cuda().contiguous()
            weight = snap['state_dict']['layer_norm.weight'].cuda()
            bias = snap['state_dict']['layer_norm.bias'].cuda()
            check(capture.name, x, weight, bias)
            del x, weight, bias, snap

    passed = [w for w in kernels if all(v['passed'] for v in checks if v['width'] == w)]
    for rows in (393216, 2359296):
        if not passed:
            break
        x = torch.randn(rows, 128, device='cuda')
        weight = torch.randn(128, device='cuda')
        bias = torch.randn(128, device='cuda')
        reference = F.layer_norm(x, (128,), weight, bias, 1e-5)
        functions = {'torch': lambda: F.layer_norm(x, (128,), weight, bias, 1e-5)}
        for width in passed:
            assert compare(kernels[width](x, weight, bias), reference)['bitwise_equal']
            functions[str(width)] = lambda w=width: kernels[w](x, weight, bias)
        for f in functions.values():
            f()
        samples = {name: [] for name in functions}
        order = list(functions)
        for i in range(8):
            for name in order if i % 2 == 0 else reversed(order):
                samples[name].append(elapsed(functions[name]))
        timing.append(dict(rows=rows, samples_ms=samples,
            medians_ms={n: statistics.median(v) for n, v in samples.items()}))
        del x, weight, bias, reference
    result = dict(schema_version=1, passed_widths=passed, checks=checks, timing=timing,
        trained_capture_pins=capture_pins, trained_capture_count=len(capture_paths),
        library_sha256=receipt['library_sha256'], torch=str(torch.__version__),
        hip=torch.version.hip, seed=101, strict=True,
        scope='Operator output, mean and rstd bits; complete allocation-inclusive operator timing')
    (out / 'screen.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed_widths=passed, timing=timing), indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
