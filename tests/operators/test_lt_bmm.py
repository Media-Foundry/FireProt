"""Exact-gate fallback and non-default-stream coverage for experimental Lt BMM."""
import os
import pytest
import torch
from fireprot.backends.amd import lt_bmm


def test_disabled_keeps_native_autograd(monkeypatch):
    monkeypatch.delenv('FIREPROT_ENABLE_LT_BMM_1536', raising=False)
    monkeypatch.setattr(lt_bmm, 'load_extension', lambda: pytest.fail('unexpected build'))
    a=torch.randn(2,3,3,requires_grad=True)
    b=torch.randn(2,3,3,requires_grad=True)
    result=lt_bmm.packed_bmm(a,b,True)
    torch.testing.assert_close(result,torch.bmm(a,b.transpose(-1,-2)))
    result.sum().backward()
    assert a.grad is not None and b.grad is not None


def test_unavailable_extension_falls_back(monkeypatch):
    monkeypatch.setattr(lt_bmm,'supported',lambda a,b: True)
    monkeypatch.setattr(lt_bmm,'load_extension',lambda: None)
    a,b=torch.randn(2,3,3),torch.randn(2,3,3)
    torch.testing.assert_close(lt_bmm.packed_bmm(a,b,False),torch.bmm(a.transpose(-1,-2),b))


@pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,
                    reason='requires HIP GPU tests')
def test_exact_gate_and_stream_workspace(monkeypatch):
    assert os.environ.get('HIP_VISIBLE_DEVICES') in {'6','7'}
    monkeypatch.setenv('FIREPROT_ENABLE_LT_BMM_1536','1')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(443)
    with torch.inference_mode():
        a,b=[torch.randn(128,1536,1536,device='cuda').div_(1536**0.5) for _ in range(2)]
        assert lt_bmm.supported(a,b)
        assert not lt_bmm.supported(a,b.transpose(-1,-2))
        assert not lt_bmm.supported(a[:1],b[:1])
        assert not lt_bmm.supported(torch.zeros(1),b)
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        outputs=[]
        before=lt_bmm.extension_status()['counts']
        for outgoing,stream in zip((True,False),streams):
            with torch.cuda.stream(stream):
                outputs.append(lt_bmm.packed_bmm(a,b,outgoing))
                # Reuse allocator blocks after workspace lifetime ends.
                scratch=torch.empty(32*1024*1024,device='cuda',dtype=torch.uint8)
                scratch.fill_(17)
                del scratch
        torch.cuda.synchronize()
        assert lt_bmm.extension_status()['loaded'],lt_bmm.extension_status()
        for outgoing,result in zip((True,False),outputs):
            expected=torch.bmm(a,b.transpose(-1,-2)) if outgoing else torch.bmm(a.transpose(-1,-2),b)
            torch.testing.assert_close(result,expected,atol=2e-5,rtol=2e-5)
        after=lt_bmm.extension_status()['counts']
        assert after['outgoing']==before['outgoing']+1
        assert after['incoming']==before['incoming']+1
