"""Transition fusion precision, dispatch, fallback and current-stream contracts."""
import os
from types import SimpleNamespace
import pytest
import torch
from fireprot.backends.amd import recycle,transition_projection
from fireprot.backends.amd.extension import load_amd_extension
HIP=pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,reason='explicit HIP opt-in required')


def test_cpu_and_disabled_reject(monkeypatch):
    module=SimpleNamespace(training=False)
    with torch.inference_mode():
        monkeypatch.delenv('FIREPROT_ENABLE_TRANSITION_PROJECTION',raising=False)
        assert not transition_projection.supported(module,torch.empty(2,128))
        monkeypatch.setenv('FIREPROT_ENABLE_TRANSITION_PROJECTION','1')
        assert not transition_projection.supported(module,torch.empty(2,128))


@HIP
@pytest.mark.parametrize('n',[1000,1536])
def test_projection_stream_dispatch_and_guards(n,monkeypatch):
    monkeypatch.setenv('FIREPROT_ENABLE_LONG1536','1')
    extension=load_amd_extension();assert extension is not None
    torch.manual_seed(1061)
    with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
        module=SimpleNamespace(training=False,n=4,c_in=128,
            layernorm1=torch.nn.LayerNorm(128).cuda().eval(),
            linear_no_bias_a=torch.nn.Linear(128,512,bias=False).cuda().eval(),
            linear_no_bias_b=torch.nn.Linear(128,512,bias=False).cuda().eval(),
            linear_no_bias=torch.nn.Linear(512,128,bias=False).cuda().eval())
        for linear in (module.linear_no_bias_a,module.linear_no_bias_b,module.linear_no_bias):linear.weight.normal_(0,.1)
        x=torch.randn(n,n,128,device='cuda');x[0,0]=0;x[0,1]=100;x[0,2]=-100
        monkeypatch.setenv('FIREPROT_ENABLE_TRANSITION_PROJECTION','0')
        expected=recycle._hip_transition(module=module,x=x,native=lambda:None)
        monkeypatch.delenv('FIREPROT_ENABLE_TRANSITION_PROJECTION',raising=False)
        key=(recycle.TRANSITION_OPERATION,'amd-gfx90a-transition-projection');before=recycle._fast_path_counts[key]
        actual=recycle._hip_transition(module=module,x=x,native=lambda:None)
        assert recycle._fast_path_counts[key]==before+1
        for start in range(0,n,16):torch.testing.assert_close(actual[start:start+16],expected[start:start+16],atol=2e-5,rtol=2e-5)
        assert actual.shape==x.shape and actual.is_contiguous() and actual.data_ptr()!=x.data_ptr()
        matrix=x.reshape(-1,128);assert transition_projection.supported(module,matrix)
        monkeypatch.setenv('FIREPROT_ENABLE_TRANSITION_PROJECTION','1')
        assert transition_projection.supported(module,matrix)
        with torch.autocast('cuda',dtype=torch.bfloat16):assert not transition_projection.supported(module,matrix)
        with torch.autocast('cuda',dtype=torch.float32):assert transition_projection.supported(module,matrix)
        hook=module.linear_no_bias_a.register_forward_hook(lambda *args:None)
        assert not transition_projection.supported(module,matrix);hook.remove()
        module.linear_no_bias_a.precision=torch.float64
        assert not transition_projection.supported(module,matrix);module.linear_no_bias_a.precision=None
        module.training=True;assert not transition_projection.supported(module,matrix);module.training=False
        if n==1536:
            monkeypatch.setenv('FIREPROT_ENABLE_LONG1536','0');assert not transition_projection.supported(module,matrix)
        with pytest.raises(RuntimeError,match='exact long'):
            extension.transition_projection(matrix[:131],module.linear_no_bias_a.weight,module.linear_no_bias_b.weight)
        with pytest.raises(RuntimeError,match='contiguous'):
            extension.transition_projection(matrix,module.linear_no_bias_a.weight.t(),module.linear_no_bias_b.weight)
        sentinel=object();monkeypatch.setattr(recycle,'load_amd_extension',lambda:None)
        assert recycle._hip_transition(module=module,x=x,native=lambda:sentinel) is sentinel
    with torch.enable_grad(),pytest.raises(RuntimeError,match='inference'):
        extension.transition_projection(matrix,module.linear_no_bias_a.weight,module.linear_no_bias_b.weight)
