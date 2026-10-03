"""Exact pair-layout LayerNorm dispatch, current-stream behavior and C++ guards."""
import os
import pytest
import torch
from fireprot.backends.amd import recycle
from fireprot.backends.amd.extension import load_amd_extension

HIP=pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,
                       reason='explicit HIP opt-in required')


def test_cpu_falls_back(monkeypatch):
    monkeypatch.setenv('FIREPROT_ENABLE_TRANSPOSED_LAYERNORM','1')
    module=torch.nn.LayerNorm(128).eval();x=torch.empty(2,2,128)
    sentinel=object()
    with torch.inference_mode():
        assert recycle.dispatch_layer_norm(module=module,x=x,native=lambda:sentinel) is sentinel


@HIP
@pytest.mark.parametrize('n',[1000,1536])
@pytest.mark.parametrize('subgroup',[False,True,None])
def test_stream_numerics_and_dispatch(n,subgroup,monkeypatch):
    monkeypatch.setenv('FIREPROT_ENABLE_TRANSPOSED_LAYERNORM','1')
    if subgroup is None:
        monkeypatch.delenv('FIREPROT_ENABLE_TRANSPOSED_NORM_SUBGROUP',raising=False)
    else:
        monkeypatch.setenv('FIREPROT_ENABLE_TRANSPOSED_NORM_SUBGROUP','1' if subgroup else '0')
    monkeypatch.setattr(recycle,'_long1536_epilogues_enabled',True)
    ext=load_amd_extension();assert ext is not None
    torch.manual_seed(1047)
    key=('layer_norm','amd-gfx90a-transposed-layernorm'+('-subgroup' if subgroup is not False else ''))
    with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
        module=torch.nn.LayerNorm(128).cuda().eval()
        module.weight.normal_();module.bias.normal_()
        x=torch.randn(n,n,128,device='cuda').transpose(0,1)
        x[0,0]=1000.;x[0,1]=1.+torch.arange(128,device='cuda')*1e-6
        x[0,2]=1000.+torch.arange(128,device='cuda')*.001
        saved=x.clone();expected=module(x)
        before=recycle._fast_path_counts[key]
        actual=recycle.dispatch_layer_norm(module=module,x=x,native=lambda:None)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
        assert actual.is_contiguous() and actual.data_ptr()!=x.data_ptr()
        assert recycle._fast_path_counts[key]==before+1
        assert torch.equal(x,saved)
        sentinel=object()
        monkeypatch.setenv('FIREPROT_ENABLE_TRANSPOSED_LAYERNORM','0')
        assert recycle.dispatch_layer_norm(module=module,x=x,native=lambda:sentinel) is sentinel
        monkeypatch.setenv('FIREPROT_ENABLE_TRANSPOSED_LAYERNORM','1')
        module.train()
        assert recycle.dispatch_layer_norm(module=module,x=x,native=lambda:sentinel) is sentinel
        module.eval()
        with pytest.raises(RuntimeError,match='strides'):
            ext.transposed_layer_norm(x.contiguous(),module.weight,module.bias,1e-5)
        with pytest.raises(RuntimeError,match='epsilon'):
            ext.transposed_layer_norm(x,module.weight,module.bias,1e-3)
        monkeypatch.setattr(recycle,'load_amd_extension',lambda:None)
        assert recycle.dispatch_layer_norm(module=module,x=x,native=lambda:sentinel) is sentinel
    with torch.enable_grad(),pytest.raises(RuntimeError,match='inference'):
        ext.transposed_layer_norm(x,module.weight,module.bias,1e-5)
