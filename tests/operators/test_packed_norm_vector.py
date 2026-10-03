"""Packed LayerNorm must preserve the low-variance behavior of the old tree."""
import os
import pytest
import torch
from fireprot.backends.amd.extension import load_amd_extension
from fireprot.backends.amd import triangle_multiplication as tm


@pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,
                    reason='explicit HIP opt-in required')
@pytest.mark.parametrize('n',[1000,1536])
def test_packed_norm_vector_low_variance_queue_and_stream(n):
    ext=load_amd_extension();assert ext is not None
    torch.manual_seed(971)
    with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
        x=torch.randn(128,n,n,device='cuda');w=torch.randn(128,device='cuda');b=torch.randn(128,device='cuda')
        x[:,-1,-1]=1.+torch.arange(128,device='cuda')*1e-6
        x[:,-1,-2]=1000.;x[:,-1,-3]=-1000.
        saved=[t.clone() for t in (x,w,b)]
        for eps in (1e-5,1e-3):
            reference=ext.packed_layer_norm_f32_128(x,w,b,eps)
            outputs=[ext.packed_norm_vector(x,w,b,eps) for _ in range(3)]
            for actual in outputs:
                assert actual.is_contiguous() and actual.shape==(n,n,128)
                torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
            del outputs,actual,reference
        for tensor,before in zip((x,w,b),saved):torch.testing.assert_close(tensor,before,atol=0,rtol=0)
        with pytest.raises(RuntimeError,match='exact'):ext.packed_norm_vector(x[:1],w,b,1e-5)
        with pytest.raises(RuntimeError,match='positive'):ext.packed_norm_vector(x,w,b,0.)
        unaligned=torch.empty(129,device='cuda')[1:]
        with pytest.raises(RuntimeError,match='alignment'):ext.packed_norm_vector(x,unaligned,b,1e-5)
    torch.cuda.synchronize()


@pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,
                    reason='explicit HIP opt-in required')
@pytest.mark.parametrize('n',[1000,1536])
def test_packed_norm_complete_triangle(n,monkeypatch):
    os.environ['LAYERNORM_TYPE']='torch'
    from fireprot.adapters.protenix.runtime import install
    from protenix.model.triangular.triangular import TriangleMultiplicativeUpdate
    install('hipblas');assert load_amd_extension() is not None
    torch.manual_seed(977);torch.backends.cuda.matmul.allow_tf32=False
    monkeypatch.setattr(tm,'_vector_packed_enabled',True)
    with torch.inference_mode():
        for outgoing in (True,False):
            module=TriangleMultiplicativeUpdate(128,128,outgoing).cuda().eval()
            for p in module.parameters():p.normal_(0,0.1)
            source=torch.randn(n,n,128,device='cuda');z=torch.empty_like(source)
            for add in (True,False):
                kwargs=dict(module=module,z=z,mask=None,inplace_safe=True,add_with_inplace=add,
                            inplace_chunk_size=256,native=lambda:pytest.fail('unexpected fallback'))
                monkeypatch.setattr(tm,'_vector_norm_enabled',False);z.copy_(source)
                reference=tm._long_sequence_packed(**kwargs).clone()
                monkeypatch.setattr(tm,'_vector_norm_enabled',True);z.copy_(source)
                output=tm._long_sequence_packed(**kwargs)
                assert output.data_ptr()==z.data_ptr()
                torch.testing.assert_close(output,reference,atol=2e-5,rtol=2e-5)
            del module,source,z,reference,output
    torch.cuda.synchronize()


@pytest.mark.parametrize('flag,expected',[(None,True),('0',False),('1',True)])
def test_packed_norm_environment_default(flag,expected):
    import subprocess
    import sys
    env=os.environ.copy();env.pop('FIREPROT_ENABLE_VECTOR_PACKED_NORM',None)
    if flag is not None:env['FIREPROT_ENABLE_VECTOR_PACKED_NORM']=flag
    code='from fireprot.backends.amd.triangle_multiplication import vector_manifest; print(vector_manifest()["packed_norm_enabled"])'
    assert subprocess.check_output([sys.executable,'-c',code],env=env,text=True).strip()==str(expected)
