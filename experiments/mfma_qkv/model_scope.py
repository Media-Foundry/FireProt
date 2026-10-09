"""Opt-in index32 fusion trial inside the qualified stream-owned workspace.

The existing Rust lease and retained scores remain allocated. This trial does
not expose probabilities or change resource policy; only the PV result escapes.
"""
from contextlib import contextmanager
from pathlib import Path
import ctypes
import hashlib
import json
import torch
from experiments.mfma_qk import workspace_attention as native
from experiments.mfma_qk.probe_fusion import validate_inputs

@contextmanager
def qkv_scope(output, library, expected_sha256):
    if hashlib.sha256(Path(library).read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('QKV library digest mismatch')
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != '47d2efa63c4402367af9282f05411b2caa123dc9f8826b5314f82afae8cc966d':
        raise ValueError('Unqualified workspace source')
    original = native.WorkspaceFusion.__call__
    packer = original.__globals__.get('_key_packer')
    if packer is None:
        raise RuntimeError('Qualified HIP key scope must be entered first')
    lib = ctypes.CDLL(library)
    entry = lib.fireprot_qkv_softmax_f32
    entry.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] + [ctypes.c_int64] * 5 + [ctypes.c_void_p]
    entry.restype = ctypes.c_int
    counts = {}
    fallback = {}

    def invoke(self, query, key, value, mask, bias):
        if self.batch not in (32, 128):
            fallback[str(self.batch)] = fallback.get(str(self.batch), 0) + 1
            return original(self, query, key, value, mask, bias)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent workspace use is forbidden')
        submitted = False
        try:
            self._check_inputs(query, key, value, mask, bias)
            stream = torch.cuda.current_stream(query.device).cuda_stream
            if self._scores is None or self._device != query.device or self._stream != stream:
                raise RuntimeError('Fusion requires the existing allocated stream-owned workspace')
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Graph capture is unsupported')
            packed_q = query.reshape(self.batch * 4, 1536, 32)
            packed_k = packer(key)
            packed_v = value.reshape(self.batch * 4, 1536, 32)
            if self.contiguous_bias:
                bias = bias.contiguous()
            if validate_inputs(packed_q, packed_k, mask, bias) != self.batch or not packed_v.is_contiguous():
                raise ValueError('Unqualified packed operands')
            result = torch.empty((self.batch, 4, 1536, 32), device=query.device, dtype=query.dtype)
            submitted = True
            status = entry(packed_q.data_ptr(), packed_k.data_ptr(), packed_v.data_ptr(),
                mask.data_ptr(), bias.data_ptr(), result.data_ptr(), None, self.batch,
                mask.stride(0) if mask.shape[0] != 1 else 0, mask.stride(-1),
                bias.stride(1), bias.stride(2), bias.stride(3), stream)
            if status:
                raise RuntimeError(f'QKV fusion launch failed: {status}')
            self.calls += 1
            counts[str(self.batch)] = counts.get(str(self.batch), 0) + 1
            return result
        except BaseException:
            if submitted:
                self._closed = True
            raise
        finally:
            self._lock.release()

    native.WorkspaceFusion.__call__ = invoke
    try:
        yield
    finally:
        native.WorkspaceFusion.__call__ = original
        Path(output, 'qkv-fusion.json').write_text(json.dumps(dict(
            library=library, library_sha256=expected_sha256, calls_by_batch=counts,
            fallback_by_batch=fallback, retained_scores_written=False,
            scope='Experimental PV-result-only fusion; existing workspace allocation/lease retained; parent QK receipt describes fallback, not fused calls'
        ), indent=2) + '\n')
