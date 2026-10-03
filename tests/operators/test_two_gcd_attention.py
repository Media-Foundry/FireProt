"""Two-GCD lifetime, device gating and full-output checks."""
import os
from types import SimpleNamespace
import pytest
import torch
from fireprot.backends.amd import two_gcd_attention as dual


def test_disabled_and_visibility_gate(monkeypatch):
    monkeypatch.delenv("FIREPROT_ENABLE_TWO_GCD_ATTENTION", raising=False)
    assert dual.rejection_reason(None, None, None, None, []) == "disabled"
    monkeypatch.setenv("FIREPROT_ENABLE_TWO_GCD_ATTENTION", "1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7")
    assert "physical GPUs" in dual.rejection_reason(None, None, None, None, [])


@pytest.mark.skipif(os.environ.get("FIREPROT_RUN_TWO_GCD_TESTS") != "1" or not torch.version.hip,
                    reason="explicit two-GCD test opt-in required")
def test_full_output_stream_lifetime_and_rejection(monkeypatch):
    from fireprot.backends.amd.triangle_attention import _direct_sdpa

    assert os.environ.get("HIP_VISIBLE_DEVICES") in {"6,7", "7,6"}
    monkeypatch.setenv("FIREPROT_ENABLE_TWO_GCD_ATTENTION", "1")
    torch.manual_seed(743)
    module = SimpleNamespace(training=False)
    caller = torch.cuda.Stream(device=0)
    with torch.inference_mode(), torch.cuda.stream(caller):
        q, k, v = [torch.randn(512, 1536, 4, 32, device="cuda:0").transpose(1, 2) for _ in range(3)]
        q.mul_(32**-0.5)
        bias = torch.randn(1536, 1536, 4, device="cuda:0").permute(2, 0, 1).unsqueeze(0)
        bias[:, :, 0, :] = float("-inf")
        bias[:, :, 1, 0] = 80
        kwargs = dict(module=module, q=q, k=k, v=v, biases=[bias])
        with monkeypatch.context() as patch:
            patch.setenv("FIREPROT_ENABLE_TWO_GCD_ATTENTION", "0")
            reference = _direct_sdpa(**kwargs, native=lambda: None)
        before = [t.clone() for t in (q,k,v,bias)]
        for _ in range(3):
            actual = _direct_sdpa(**kwargs, native=lambda: None)
            assert actual.is_contiguous()
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
            del actual
            # Exercise allocator reuse after asynchronous peer temporaries die.
            temporary = torch.empty((512,1536,128), device="cuda:0"); temporary.fill_(13); del temporary
        for t, saved in zip((q,k,v,bias), before):
            torch.testing.assert_close(t,saved,atol=0,rtol=0)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            assert "autocast" in dual.rejection_reason(**kwargs)
        module.training=True
        assert dual.rejection_reason(**kwargs)=="requires inference"
        module.training=False
        assert "strides" in dual.rejection_reason(module,q.contiguous(),k,v,[bias])
        assert torch.cuda.current_device()==0
    caller.synchronize()
    assert dual.manifest()['counts']['selected'] >= 3
