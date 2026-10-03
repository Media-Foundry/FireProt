"""Exact sequence specialization must preserve complete attention semantics."""
import dataclasses
import os
from types import SimpleNamespace
import pytest
import torch
from fireprot.backends.amd import ck_attention as ck
from fireprot.backends.amd import triangle_attention as ta
from fireprot.ops.registry import OperationSignature,registry


def test_exact1536_capability(monkeypatch):
    signature=OperationSignature.create(operation=ta.OPERATION,device_type='cuda',architecture='gfx90a',dtype='float32',shape=(512,4,1536,32),layout='strided',mode='inference',mask='triangle-bias',attributes={'bias_shapes':repr(((1,4,1536,1536),)),'qkv_strides':repr(((196608,32,128,1),)*3),'ck_tile_compatible':True})
    monkeypatch.setattr(ck,'_enabled',True);monkeypatch.setattr(ck,'_long1536_enabled',True)
    monkeypatch.setattr(ta,'_long1536_enabled',True);monkeypatch.setattr(ta,'_long_sequence_enabled',True);monkeypatch.setattr(ta,'_long_direct_enabled',True)
    monkeypatch.setattr(ck,'load_extension',lambda:object())
    assert ck._supports_long1536(signature) is True
    for changes in (dict(shape=(512,4,1535,32)),dict(dtype='bfloat16'),dict(mode='training'),dict(architecture='gfx942')):
        assert ck._supports_long1536(dataclasses.replace(signature,**changes))[0] is False


@pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,reason='explicit HIP opt-in required')
def test_exact1536_full_outputs_and_stream(monkeypatch):
    monkeypatch.setattr(ck,'_enabled',True);monkeypatch.setattr(ck,'_exact1536_enabled',True);monkeypatch.setattr(ck,'_long1536_enabled',True)
    monkeypatch.setattr(ta,'_long1536_enabled',True)
    registry.clear_runtime_state()
    assert ck.load_extension() is not None,ck.extension_status()
    # Invalid dimensions are rejected before any pointer is dereferenced.
    assert ck._entry_exact(None,None,None,None,None,512,1535,None)==2
    torch.manual_seed(853)
    try:
        with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
            q,k,v=[torch.randn(512,1536,4,32,device='cuda').transpose(1,2) for _ in range(3)];q.div_(32**0.5)
            bias=torch.randn(1536,1536,4,device='cuda').permute(2,0,1).unsqueeze(0)
            saved=[t.clone() for t in (q,k,v,bias)]
            kwargs=dict(module=SimpleNamespace(training=False),q=q,k=k,v=v,biases=[bias],native=lambda:pytest.fail('native selected'))
            reference=ta._direct_sdpa(**kwargs)
            queued=[ta.dispatch_triangle_attention(**kwargs) for _ in range(3)]
            for output in queued:
                assert output.is_contiguous()
                torch.testing.assert_close(output,reference,atol=2e-5,rtol=2e-5)
            del queued,reference
            for t,before in zip((q,k,v,bias),saved):torch.testing.assert_close(t,before,atol=0,rtol=0)
            bias[:,:,0,:]=-float('inf');bias[:,:,1,0]=80
            torch.testing.assert_close(ta.dispatch_triangle_attention(**kwargs),ta._direct_sdpa(**kwargs),atol=2e-5,rtol=2e-5)
        torch.cuda.synchronize()
        assert any(x['implementation']==ck.LONG1536_IMPLEMENTATION and x['count']==4 for x in registry.manifest()['selection_counts'])
    finally:registry.clear_runtime_state()
