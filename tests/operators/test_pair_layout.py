"""Pair-layout view eligibility and residual update semantics."""
import os
import pytest
import torch
from fireprot.backends.amd import pair_layout


def test_cpu_and_disabled_keep_contiguous(monkeypatch):
    x=torch.arange(2*2*128).view(2,2,128).float()
    monkeypatch.setenv('FIREPROT_ENABLE_PAIR_LAYOUT','1')
    monkeypatch.setenv('FIREPROT_ENABLE_TRANSPOSED_LAYERNORM','1')
    y=pair_layout.transpose_pair(x,training=False)
    assert y.is_contiguous() and y.data_ptr()!=x.data_ptr()
    assert torch.equal(y,x.transpose(0,1))


@pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,
                    reason='explicit HIP opt-in required')
@pytest.mark.parametrize('n',[1000,1536])
def test_view_residual_roundtrip(n,monkeypatch):
    monkeypatch.setenv('FIREPROT_ENABLE_PAIR_LAYOUT','1')
    monkeypatch.setenv('FIREPROT_ENABLE_TRANSPOSED_LAYERNORM','1')
    monkeypatch.setenv('FIREPROT_ENABLE_LONG1536','1')
    with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
        x=torch.randn(n,n,128,device='cuda');update=torch.randn_like(x)
        reference=x.transpose(0,1).contiguous();reference+=update
        reference=reference.transpose(0,1).contiguous()
        y=pair_layout.transpose_pair(x,training=False)
        assert y.data_ptr()==x.data_ptr() and y.stride()==(128,n*128,1)
        y+=update
        result=pair_layout.transpose_pair(y,training=False)
        assert result.is_contiguous() and result.data_ptr()==x.data_ptr()
        assert torch.equal(result,reference)
        monkeypatch.setenv('FIREPROT_ENABLE_PAIR_LAYOUT','0')
        assert pair_layout.transpose_pair(x,training=False).is_contiguous()
        monkeypatch.setenv('FIREPROT_ENABLE_PAIR_LAYOUT','1')
        assert pair_layout.transpose_pair(x,training=True).is_contiguous()
        monkeypatch.setattr(pair_layout,'load_amd_extension',lambda:None)
        assert pair_layout.transpose_pair(x,training=False).is_contiguous()
