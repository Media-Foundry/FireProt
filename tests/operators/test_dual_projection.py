"""Validate the opt-in fused FP32 dual projection and its dispatch guards."""
import os
import subprocess
import sys
import pytest
import torch
from fireprot.backends.amd.extension import load_amd_extension
from fireprot.backends.amd import triangle_multiplication as tm

HIP = pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,
                        reason='explicit HIP opt-in required')

@pytest.mark.parametrize('flag,expected',[(None,False),('0',False),('1',True)])
def test_dual_projection_environment(flag,expected):
    env=os.environ.copy();env.pop('FIREPROT_ENABLE_DUAL_PROJECTION',None)
    if flag is not None:env['FIREPROT_ENABLE_DUAL_PROJECTION']=flag
    code='from fireprot.backends.amd.triangle_multiplication import vector_manifest; print(vector_manifest()["dual_projection_enabled"])'
    assert subprocess.check_output([sys.executable,'-c',code],env=env,text=True).strip()==str(expected)

@HIP
@pytest.mark.parametrize('n',[1000,1536])
def test_dual_projection_queue_stream_and_guards(n):
    ext=load_amd_extension();assert ext is not None
    torch.manual_seed(991);torch.backends.cuda.matmul.allow_tf32=False
    with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
        x=torch.randn(n,n,128,device='cuda')
        x[0,0]=0.;x[0,1]=100.;x[0,2]=-100.
        wg=torch.randn(128,128,device='cuda')*.1;wp=torch.randn_like(wg)*.1
        saved=[t.clone() for t in (x,wg,wp)]
        reference=ext.triangle_vector_pack(torch.nn.functional.linear(x,wg),torch.nn.functional.linear(x,wp))
        outputs=[ext.dual_projection(x,wg,wp) for _ in range(3)]
        for actual in outputs:
            assert actual.is_contiguous() and actual.shape==(128,n,n)
            torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
        for t,before in zip((x,wg,wp),saved):torch.testing.assert_close(t,before,atol=0,rtol=0)
        with pytest.raises(RuntimeError,match='exact'):ext.dual_projection(x[:2],wg,wp)
        with pytest.raises(RuntimeError,match='weights'):ext.dual_projection(x,wg[:2],wp)
        with pytest.raises(RuntimeError,match='contiguous'):ext.dual_projection(x,wg.T,wp)
        bad=torch.empty(128*128+1,device='cuda')[1:].view(128,128)
        with pytest.raises(RuntimeError,match='alignment'):ext.dual_projection(x,bad,wp)
        with pytest.raises(RuntimeError,match='FP32'):ext.dual_projection(x,wg.half(),wp)
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError,match='inference'):ext.dual_projection(x,wg,wp)

@HIP
@pytest.mark.parametrize('n',[1000,1536])
def test_dual_projection_complete_triangle_and_bias_fallback(n,monkeypatch):
    os.environ['LAYERNORM_TYPE']='torch'
    from fireprot.adapters.protenix.runtime import install
    from protenix.model.triangular.triangular import TriangleMultiplicativeUpdate
    install('hipblas');assert load_amd_extension() is not None
    torch.manual_seed(997);torch.backends.cuda.matmul.allow_tf32=False
    with torch.inference_mode():
        for outgoing in (True,False):
            module=TriangleMultiplicativeUpdate(128,128,outgoing).cuda().eval()
            for p in module.parameters():p.normal_(0,.1)
            source=torch.randn(n,n,128,device='cuda');z=torch.empty_like(source)
            for add in (True,False):
                kwargs=dict(module=module,z=z,mask=None,inplace_safe=True,add_with_inplace=add,
                            inplace_chunk_size=256,native=lambda:pytest.fail('unexpected native fallback'))
                monkeypatch.setattr(tm,'_dual_projection_enabled',False);z.copy_(source)
                reference=tm._long_sequence_packed(**kwargs).clone()
                monkeypatch.setattr(tm,'_dual_projection_enabled',True);z.copy_(source)
                before=tm.vector_manifest()['dual_projection_counts'][f'n{n}']
                actual=tm._long_sequence_packed(**kwargs)
                assert actual.data_ptr()==z.data_ptr()
                assert tm.vector_manifest()['dual_projection_counts'][f'n{n}']==before+1
                torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
            module.linear_a_g.bias=torch.nn.Parameter(torch.ones(128,device='cuda'))
            assert not tm._supports_dual_projection(module,source)
            monkeypatch.setattr(tm,'_dual_projection_enabled',False);z.copy_(source)
            reference=tm._long_sequence_packed(**kwargs).clone()
            monkeypatch.setattr(tm,'_dual_projection_enabled',True);z.copy_(source)
            actual=tm._long_sequence_packed(**kwargs)
            torch.testing.assert_close(actual,reference,atol=0,rtol=0)
            del module,source,z,reference,actual
    torch.cuda.synchronize()
