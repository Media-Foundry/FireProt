"""Scoped FP32 attention dispatch; not yet enabled in model workers.

QK/PV and fallback arithmetic remain upstream. Rust owns capability selection,
cache decisions and counters. Python owns tensor inspection and invocation scope.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from pathlib import Path
import threading

import torch

from fireprot.ops.registry import OperationRegistry, signature_from_tensor

SOURCE_SHA256 = "30fe5433982ee73e3d7ce5341fa6b510b1ea332dbde072d56129632cc53f0746"
_current = ContextVar("fireprot_openfold3_attention", default=None)
_lock = threading.RLock()
_installed = False


@contextmanager
def _scoped_hook(module, dispatcher):
    """One installation; other execution contexts retain the captured native call."""
    global _installed
    with _lock:
        if _installed:
            raise RuntimeError("An OpenFold3 attention scope is already installed")
        original = module._attention
        live = True
        def call(*args, **kwargs):
            active = _current.get()
            if not live or active is None:
                return original(*args, **kwargs)
            return active(original, *args, **kwargs)
        module._attention = call
        _installed = True
        token = _current.set(dispatcher)
    try:
        yield
    finally:
        with _lock:
            live = False
            _current.reset(token)
            _installed = False
            if module._attention is not call:
                raise RuntimeError("OpenFold3 attention hook changed during its scope")
            module._attention = original


@contextmanager
def attention_scope(dispatcher):
    import openfold3.core.model.primitives.attention as module
    source = Path(module.__file__).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Unsupported OpenFold3 attention source")
    function = module._attention
    code = getattr(function, "__code__", None)
    if (code is None or code.co_name != "_attention"
            or Path(code.co_filename).resolve() != source):
        raise RuntimeError("OpenFold3 attention was already replaced")
    with _scoped_hook(module, dispatcher):
        yield


class AttentionDispatcher:
    """Supply a verified SoftmaxFP32 instance, or None for native-only dispatch."""
    def __init__(self, kernel):
        self.kernel = kernel
        self.registry = OperationRegistry()
        self.registry.register(
            operation="attention", name="hip_mask_bias_softmax_fp32",
            function=self._fused, priority=10, version="softmax-fp32-v1",
            supports=lambda signature: (
                dict(signature.attributes).get("eligible") == "True",
                "FP32 GPU tensor, mask, layout or published-kernel contract mismatch"))

    def _eligible(self, query, key, value, biases):
        if self.kernel is None or torch.is_grad_enabled():
            return False
        if not isinstance(biases, (tuple, list)) or len(biases) != 2:
            return False
        tensors = (query, key, value, *biases)
        if any(type(t) is not torch.Tensor for t in tensors):
            return False
        if any(t.dtype != torch.float32 or not t.is_cuda or t.device != query.device
               or t.requires_grad or t.is_neg() or t.is_conj() for t in tensors):
            return False
        if (query.device.index != self.kernel.device or query.ndim != 4
                or query.shape != key.shape or query.shape != value.shape
                or not 0 < query.shape[0] <= 1536 or query.shape[1] != 4
                or query.shape[2] not in (1000, 1536) or query.shape[3] != 32):
            return False
        mask, bias = biases
        n = query.shape[2]
        return (mask.ndim == bias.ndim == 4
                and mask.shape[0] in (1, query.shape[0])
                and tuple(mask.shape[1:]) == (1, 1, n)
                and tuple(bias.shape) == (1, 4, n, n)
                and all(s >= 0 for t in tensors for s in t.stride()))

    def __call__(self, native, query, key, value, biases, use_high_precision=False):
        if type(query) is not torch.Tensor:
            return native(query, key, value, biases, use_high_precision=use_high_precision)
        eligible = self._eligible(query, key, value, biases)
        signature = signature_from_tensor(
            "attention", query, architecture="gfx90a" if self.kernel is not None else "unknown",
            mode="training" if torch.is_grad_enabled() else "inference",
            mask="pair_and_token", inplace=False,
            attributes={"eligible": eligible, "high_precision": use_high_precision})
        return self.registry.dispatch(signature, native, query, key, value, biases,
                                      use_high_precision=use_high_precision)

    def _fused(self, query, key, value, biases, use_high_precision=False):
        # The published kernel only accepts FP32, so both precision modes use
        # this same autocast context in the pinned upstream implementation.
        with torch.amp.autocast("cuda", dtype=torch.float32):
            scores = torch.einsum("...qc, ...kc->...qk", query, key)
            if self.kernel.supports(scores, biases):
                scores = self.kernel(scores, biases)
            else:
                # QK has already executed; preserve it for unmatched output layout.
                for bias in biases:
                    scores += bias
                scores = torch.nn.functional.softmax(scores, dim=-1)
            result = torch.einsum("...qk, ...kc->...qc", scores, value)
        return result.to(dtype=query.dtype)

    def manifest(self):
        return dict(registry=self.registry.manifest(),
                    kernel=None if self.kernel is None else self.kernel.receipt,
                    limits=["Registry selection counts precede QK and do not prove a HIP launch."])
