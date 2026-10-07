"""Qualification experiment: direct strided reads versus copy plus exact C128 HIP.

Run only after the active paired model benchmark is terminal. No model dispatch.
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
if __package__:
    from .screen import Kernel, compare, elapsed, verified_captures
else:
    from screen import Kernel, compare, elapsed, verified_captures


class StridedKernel:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        self.entry = self.library.fireprot_norm128_strided
        self.entry.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 4 + [
            ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        self.entry.restype = ctypes.c_int

    def __call__(self, x, weight, bias, *, stats=False):
        if x.dtype != torch.float32 or not x.is_cuda or x.shape[-1] != 128 or x.stride(-1) != 1 or x.data_ptr() % 16:
            raise ValueError('Expected aligned HIP FP32 input with contiguous C128 axis')
        if x.is_conj() or x.is_neg():
            raise ValueError('Lazy view bits unsupported')
        shape = x.shape
        reduced = x
        while reduced.ndim > 3 and reduced.shape[0] == 1:
            reduced = reduced.squeeze(0)
        if reduced.is_contiguous():
            inner_rows = reduced.numel() // 128
            outer_stride, inner_stride = 0, 128
        elif reduced.ndim == 3:
            inner_rows = reduced.shape[1]
            outer_stride, inner_stride = reduced.stride()[:2]
        else:
            raise ValueError('Only two row axes or dense tensors are supported')
        for parameter in (weight, bias):
            if parameter is not None and (parameter.shape != (128,) or parameter.dtype != x.dtype or parameter.device != x.device or not parameter.is_contiguous() or parameter.data_ptr() % 16):
                raise ValueError('Unsupported affine parameter')
        rows = x.numel() // 128
        out = torch.empty(shape, device=x.device, dtype=x.dtype)
        mean = torch.empty(rows, device=x.device, dtype=x.dtype) if stats else None
        inverse = torch.empty_like(mean) if stats else None
        ptr = lambda t: None if t is None else t.data_ptr()
        status = self.entry(ptr(x), ptr(weight), ptr(bias), ptr(out), ptr(mean), ptr(inverse), rows, inner_rows, outer_stride, inner_stride, 1e-5, 32, torch.cuda.current_stream(x.device).cuda_stream)
        if status:
            raise RuntimeError(f'Strided HIP status {status}')
        return (out, mean, inverse) if stats else out


@torch.inference_mode()
def main(build, baseline_library, baseline_sha, captures, destination):
    captures, capture_pins = verified_captures(captures)
    assert 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(101)
    report = Path(destination)
    report.mkdir(exist_ok=False)
    receipt = json.loads(Path(build).read_text())
    assert hashlib.sha256(Path(receipt['library']).read_bytes()).hexdigest() == receipt['library_sha256']
    assert hashlib.sha256(Path(baseline_library).read_bytes()).hexdigest() == baseline_sha
    candidate, baseline = StridedKernel(receipt['library']), Kernel(baseline_library, 32)
    checks, timings = [], []

    def check(name, x, w, b, timed=False):
        expected = torch.native_layer_norm(x, (128,), w, b, 1e-5)
        actual = candidate(x, w, b, stats=True)
        arrays = {n: compare(a, e) for n, a, e in zip(('output', 'mean', 'rstd'), actual, expected)}
        checks.append(dict(name=name, shape=list(x.shape), stride=list(x.stride()), arrays=arrays, passed=all(v['bitwise_equal'] for v in arrays.values())))
        if not checks[-1]['passed'] or not timed:
            return
        assert compare(baseline(x.contiguous(), w, b), expected[0])['bitwise_equal']
        functions = {'copy_plus_hip': lambda: baseline(x.contiguous(), w, b), 'strided_hip': lambda: candidate(x, w, b)}
        for fn in functions.values(): fn()
        samples = {n: [] for n in functions}
        names = list(functions)
        for i in range(8):
            for n in names if i % 2 == 0 else reversed(names):
                samples[n].append(elapsed(functions[n]))
        timings.append(dict(name=name, samples_ms=samples, medians_ms={n: statistics.median(v) for n,v in samples.items()}))

    # Tails, nonzero storage offsets, four affine branches and signed zeros.
    small = torch.randn(19, 23, 128, device='cuda')
    w, b = torch.randn(128, device='cuda'), torch.randn(128, device='cuda')
    for name, x in [('tail',small), ('slice',small[1:17, 2:19]), ('transpose',small.transpose(0,1))]:
        for affine, pw, pb in [('both',w,b), ('scale',w,None), ('bias',None,b), ('none',None,None)]:
            check(name+'-'+affine,x,pw,pb)
    small.zero_(); small[..., ::2] = -0.0
    check('signed-zero-transpose',small.transpose(0,1),w,b)
    # Real trained inputs, sliced/transposed to the observed model layouts.
    for path in captures:
        snap = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        z = snap['z'].cuda().contiguous().reshape(1536,1536,128)
        w = snap['state_dict']['layer_norm.weight'].cuda()
        b = snap['state_dict']['layer_norm.bias'].cuda()
        for name, x in [('dense',z), ('row-chunk',z[256:512]), ('col-chunk',z[:,256:512]), ('transpose',z.transpose(0,1)), ('transposed-chunk',z.transpose(0,1)[256:512])]:
            check(path.stem+'-'+name,x,w,b,timed=True)
        # Observed transposed stride 98304 originates from a dense 1536x768 slice.
        narrow = z[:,:768].contiguous().transpose(0,1)[256:512]
        check(path.stem+'-transposed-narrow',narrow,w,b,timed=True)
        del narrow, z, w, b, snap, x
    payload = dict(checks=checks, timing=timings, correctness_passed=all(v['passed'] for v in checks), candidate=receipt, baseline_sha256=baseline_sha, capture_pins=capture_pins, scope='Allocation-inclusive operator timing; strict FP32 native output/mean/rstd bits; no E2E claim')
    (report/'screen.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(correctness_passed=payload['correctness_passed'], timing=timings),indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
