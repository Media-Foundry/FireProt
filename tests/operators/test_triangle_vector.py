"""Integrated vector packing and gated residual contracts for long triangles."""
import os
import pytest
import torch
from fireprot.backends.amd import triangle_multiplication as tm
from fireprot.backends.amd.extension import load_amd_extension


@pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,
                    reason='explicit HIP opt-in required')
@pytest.mark.parametrize('n',[1000,1536])
def test_vector_pack_residual_contracts(n):
    ext=load_amd_extension();assert ext is not None
    torch.manual_seed(947)
    with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
        gate=torch.randn(n,n,128,device='cuda');value=torch.randn_like(gate)
        gate.view(-1)[:4]=torch.tensor([-80.,80.,-float('inf'),float('inf')],device='cuda')
        saved_gate=gate.clone();saved_value=value.clone()
        reference=ext.sigmoid_pack_f32_128(gate,value)
        torch.testing.assert_close(ext.triangle_vector_pack(gate,value),reference,atol=0,rtol=0)
        del reference
        expected=torch.sigmoid(gate)*value
        z=torch.randn_like(gate);initial=z.clone()
        for _ in range(3):
            assert ext.triangle_residual(gate,value,z,True).data_ptr()==z.data_ptr()
        torch.testing.assert_close(z,((initial+expected)+expected)+expected,atol=2e-6,rtol=2e-6)
        ext.triangle_residual(gate,value,z,False)
        torch.testing.assert_close(z,expected,atol=2e-6,rtol=2e-6)
        for tensor,saved in [(gate,saved_gate),(value,saved_value)]:
            torch.testing.assert_close(tensor,saved,atol=0,rtol=0)
        with pytest.raises(RuntimeError,match='alias'):
            ext.triangle_residual(gate,value,value,True)
        with pytest.raises(RuntimeError,match='exact'):
            ext.triangle_vector_pack(gate[:2],value[:2])
        unaligned=torch.empty(gate.numel()+1,device='cuda')[1:].view_as(gate)
        with pytest.raises(RuntimeError,match='alignment'):
            ext.triangle_vector_pack(unaligned,value)
    torch.cuda.synchronize()


@pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,
                    reason='explicit HIP opt-in required')
@pytest.mark.parametrize('n',[1000,1536])
def test_full_triangle_vector_add_and_overwrite(n,monkeypatch):
    os.environ['LAYERNORM_TYPE']='torch'
    from fireprot.adapters.protenix.runtime import install
    from protenix.model.triangular.triangular import TriangleMultiplicativeUpdate
    install('hipblas');assert load_amd_extension() is not None
    torch.manual_seed(953)
    with torch.inference_mode():
        for outgoing in (True,False):
            module=TriangleMultiplicativeUpdate(128,128,outgoing).cuda().eval()
            for p in module.parameters():p.normal_(0,0.1)
            source=torch.randn(n,n,128,device='cuda');target=torch.empty_like(source)
            for add in (True,False):
                kwargs=dict(module=module,z=target,mask=None,inplace_safe=True,
                            add_with_inplace=add,inplace_chunk_size=256,native=lambda:pytest.fail('fallback'))
                monkeypatch.setattr(tm,'_vector_packed_enabled',False)
                target.copy_(source);reference=tm._long_sequence_packed(**kwargs).clone()
                monkeypatch.setattr(tm,'_vector_packed_enabled',True)
                target.copy_(source);actual=tm._long_sequence_packed(**kwargs)
                assert actual.data_ptr()==target.data_ptr()
                torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
            del module,source,target,reference,actual
    torch.cuda.synchronize()


@pytest.mark.parametrize('flag,expected',[(None,True),('0',False),('1',True)])
def test_vector_environment_default(flag,expected):
    import subprocess
    import sys
    env=os.environ.copy();env.pop('FIREPROT_ENABLE_TRIANGLE_VECTOR',None)
    if flag is not None:env['FIREPROT_ENABLE_TRIANGLE_VECTOR']=flag
    code='from fireprot.backends.amd.triangle_multiplication import vector_manifest; print(vector_manifest()["enabled"])'
    assert subprocess.check_output([sys.executable,'-c',code],env=env,text=True).strip()==str(expected)
