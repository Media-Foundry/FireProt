"""Validate fused QKV preparation, exact layouts, stream ordering and fallback."""
import os
from types import SimpleNamespace

import pytest
import torch

from fireprot.backends.amd import qkv_projection as qkv
from fireprot.backends.amd.extension import load_amd_extension

HIP = pytest.mark.skipif(os.environ.get("FIREPROT_RUN_HIP_TESTS") != "1" or not torch.version.hip,
                        reason="explicit HIP opt-in required")


def test_disabled_and_unavailable_fallback(monkeypatch):
    monkeypatch.delenv("FIREPROT_ENABLE_QKV_PROJECTION", raising=False)
    sentinel = object()
    args = dict(module=SimpleNamespace(), q_x=None, kv_x=None, apply_scale=True, native=lambda: sentinel)
    assert qkv.prepare(**args) is sentinel
    monkeypatch.setattr(qkv, "supported", lambda *args: True)
    monkeypatch.setattr(qkv, "load_amd_extension", lambda: None)
    assert qkv.prepare(**args) is sentinel


@HIP
@pytest.mark.parametrize("batch,n", [(1000, 1000), (512, 1536)])
def test_qkv_stream_numerics_and_guards(batch, n, monkeypatch):
    os.environ["LAYERNORM_TYPE"] = "torch"
    from protenix.model.triangular.layers import Attention
    monkeypatch.setenv("FIREPROT_ENABLE_QKV_PROJECTION", "1")
    monkeypatch.setenv("FIREPROT_ENABLE_LONG1536", "1")
    ext = load_amd_extension()
    assert ext is not None
    torch.manual_seed(1021)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        module = Attention(128, 128, 128, 32, 4).cuda().eval()
        for p in module.parameters():
            p.normal_(0, .1)
        x = torch.randn(batch, n, 128, device="cuda")
        x[0, 0] = 0
        x[0, 1] = 100
        x[0, 2] = -100
        weights = [layer.weight for layer in (module.linear_q, module.linear_k, module.linear_v)]
        saved = [t.clone() for t in (x, *weights)]
        reference = module._prep_qkv(x, x)
        before = qkv.manifest()["counts"][f"n{n}"]
        for _ in range(3):
            actual = qkv.prepare(module=module, q_x=x, kv_x=x.view_as(x), apply_scale=True,
                                 native=lambda: pytest.fail("unexpected fallback"))
            for a, b in zip(actual, reference):
                assert a.shape == (batch, 4, n, 32)
                assert a.stride() == (n * 128, 32, 128, 1)
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        assert qkv.manifest()["counts"][f"n{n}"] == before + 3
        for t, original in zip((x, *weights), saved):
            torch.testing.assert_close(t, original, atol=0, rtol=0)
        assert not qkv.supported(module, x, x, False)
        assert qkv.supported(module, x, x.view_as(x), True)
        assert not qkv.supported(module, x, x.clone(), True)
        assert not qkv.supported(module.train(), x, x, True)
        module.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert not qkv.supported(module, x, x, True)
        module.linear_q.precision = torch.bfloat16
        assert not qkv.supported(module, x, x, True)
        module.linear_q.precision = None
        handle = module.linear_q.register_forward_hook(lambda *args: None)
        assert not qkv.supported(module, x, x, True)
        handle.remove()
        module.linear_q.bias = torch.nn.Parameter(torch.zeros(128, device="cuda"))
        assert not qkv.supported(module, x, x, True)
        module.linear_q.bias = None
        for bad, message in ((weights[0].T, "contiguous"), (weights[0][:2], "weights"),
                             (weights[0].half(), "FP32"),
                             (torch.empty(16385, device="cuda")[1:].view(128, 128), "alignment")):
            with pytest.raises(RuntimeError, match=message):
                ext.qkv_projection(x, bad, weights[1], weights[2])
        with pytest.raises(RuntimeError, match="shape"):
            ext.qkv_projection(x[:2], *weights)
        if n == 1536:
            monkeypatch.setenv("FIREPROT_ENABLE_LONG1536", "0")
            assert not qkv.supported(module, x, x, True)
    torch.cuda.synchronize()
    assert not qkv.supported(module, x, x, True)
    with pytest.raises(RuntimeError, match="inference"):
        ext.qkv_projection(x, *weights)
