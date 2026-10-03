"""Validate the padded N1000 vector-bias specialization and ABI guards."""
import os
from types import SimpleNamespace
import pytest
import torch
from fireprot.backends.amd import ck_attention as ck
from fireprot.backends.amd import triangle_attention as ta


@pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS') != '1' or not torch.version.hip,
                    reason='explicit HIP test opt-in required')
def test_vector1000_alignment_tail_stream_and_queue(monkeypatch):
    monkeypatch.setattr(ck, '_vector1000_enabled', True)
    assert ck.load_extension() is not None, ck.extension_status()
    entry = ck._entry_vector
    assert entry(None, None, None, None, None, 1000, 999, None) == 2
    assert entry(None, None, None, None, None, 1000, 1000, None) == 3
    assert entry(4, 16, 16, 16, 16, 1000, 1000, None) == 3
    torch.manual_seed(919)
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        q, k, v = [torch.randn(1000, 1000, 4, 32, device='cuda').transpose(1, 2)
                   for _ in range(3)]
        q.div_(32 ** 0.5)
        # contiguous() preserves this offset: the wrapper must repair alignment.
        backing = torch.randn(4 * 1000 * 1000 + 1, device='cuda')
        bias = backing[1:].view(1, 4, 1000, 1000)
        assert bias.is_contiguous() and bias.data_ptr() % 16 == 4
        bias[:, :, -1, :] = -float('inf')
        bias[:, :, -2, -1] = 80
        saved = [t.clone() for t in (q, k, v, bias)]
        kwargs = dict(module=SimpleNamespace(training=False), q=q, k=k, v=v,
                      biases=[bias], native=lambda: pytest.fail('native selected'))
        reference = ta._direct_sdpa(**kwargs)
        outputs = [ck.attention(**kwargs) for _ in range(3)]
        for output in outputs:
            assert output.is_contiguous()
            torch.testing.assert_close(output, reference, atol=2e-5, rtol=2e-5)
        for tensor, before in zip((q, k, v, bias), saved):
            torch.testing.assert_close(tensor, before, atol=0, rtol=0)
        storage = torch.full((q.numel() + 8192,), 123.5, device='cuda')
        output = storage[4096:-4096].view(1000, 1000, 4, 32).transpose(1, 2)
        packed = bias.clone()
        assert entry(q.data_ptr(), k.data_ptr(), v.data_ptr(), packed.data_ptr(),
                     output.data_ptr(), 1000, 1000, torch.cuda.current_stream().cuda_stream) == 0
        torch.testing.assert_close(output.transpose(1, 2), reference, atol=2e-5, rtol=2e-5)
        assert bool((storage[:4096] == 123.5).all())
        assert bool((storage[-4096:] == 123.5).all())
    torch.cuda.synchronize()


@pytest.mark.parametrize("flag,expected", [(None, True), ("0", False), ("1", True)])
def test_vector1000_environment_default(flag, expected):
    import subprocess
    import sys
    environment = os.environ.copy()
    environment.pop("FIREPROT_ENABLE_CK_VECTOR_BIAS1000", None)
    if flag is not None:
        environment["FIREPROT_ENABLE_CK_VECTOR_BIAS1000"] = flag
    code = "from fireprot.backends.amd.ck_attention import _vector1000_enabled; print(_vector1000_enabled)"
    assert subprocess.check_output([sys.executable, "-c", code], env=environment, text=True).strip() == str(expected)
