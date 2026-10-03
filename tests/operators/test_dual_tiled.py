"""Check tiled dual-projection selection, strided BMM integration and fallback."""
import os
import subprocess
import sys
import pytest
import torch
from fireprot.backends.amd import triangle_multiplication as tm
from fireprot.backends.amd.extension import load_amd_extension

@pytest.mark.parametrize('flag,expected',[(None,True),('0',False),('1',True)])
def test_tiled_dual_environment(flag,expected):
    env=os.environ.copy();env.pop('FIREPROT_ENABLE_TILED_DUAL_PROJECTION',None)
    if flag is not None:env['FIREPROT_ENABLE_TILED_DUAL_PROJECTION']=flag
    code='from fireprot.backends.amd.triangle_multiplication import vector_manifest; print(vector_manifest()["tiled_dual_projection_enabled"])'
    assert subprocess.check_output([sys.executable,'-c',code],env=env,text=True).strip()==str(expected)

@pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,
                   reason='explicit HIP opt-in required')
@pytest.mark.parametrize('n',[1000,1536])
def test_tiled_dual_complete_triangle_and_disable(n,monkeypatch):
    os.environ['LAYERNORM_TYPE']='torch'
    from fireprot.adapters.protenix.runtime import install
    from protenix.model.triangular.triangular import TriangleMultiplicativeUpdate
    install('hipblas');assert load_amd_extension() is not None
    torch.manual_seed(1013);torch.backends.cuda.matmul.allow_tf32=False
    monkeypatch.setattr(tm,'_dual_projection_enabled',True)
    with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
        for outgoing in (True,False):
            module=TriangleMultiplicativeUpdate(128,128,outgoing).cuda().eval()
            for p in module.parameters():p.normal_(0,.1)
            source=torch.randn(n,n,128,device='cuda');z=torch.empty_like(source)
            for add in (True,False):
                kwargs=dict(module=module,z=z,mask=None,inplace_safe=True,add_with_inplace=add,
                            inplace_chunk_size=256,native=lambda:pytest.fail('unexpected native fallback'))
                monkeypatch.setattr(tm,'_dual_tiled_enabled',False);z.copy_(source)
                reference=tm._long_sequence_packed(**kwargs).clone()
                monkeypatch.setattr(tm,'_dual_tiled_enabled',True)
                before=tm.vector_manifest()['tiled_dual_projection_counts'][f'n{n}']
                for _ in range(3):
                    z.copy_(source);actual=tm._long_sequence_packed(**kwargs)
                    assert actual.data_ptr()==z.data_ptr()
                    torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
                assert tm.vector_manifest()['tiled_dual_projection_counts'][f'n{n}']==before+3
            before=tm.vector_manifest()['tiled_dual_projection_counts'][f'n{n}']
            monkeypatch.setattr(tm,'_dual_projection_enabled',False)
            z.copy_(source);tm._long_sequence_packed(**kwargs)
            assert tm.vector_manifest()['tiled_dual_projection_counts'][f'n{n}']==before
            monkeypatch.setattr(tm,'_dual_projection_enabled',True)
            del module,source,z,reference,actual
    torch.cuda.synchronize()
