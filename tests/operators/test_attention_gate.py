"""Out-of-place attention gate semantics and exact-bucket capability checks."""
import os
import pytest
import torch
from fireprot.backends.amd import attention_gate
from fireprot.backends.amd.extension import load_amd_extension


def test_disabled_uses_native(monkeypatch):
    monkeypatch.delenv('FIREPROT_ENABLE_ATTENTION_GATE_1536',raising=False)
    monkeypatch.setattr(attention_gate,'load_amd_extension',lambda: pytest.fail('unexpected build'))
    sentinel=object()
    assert attention_gate.wrap_up(module=None,o=None,x=None,native=lambda:sentinel) is sentinel


def test_unavailable_extension_uses_native(monkeypatch):
    monkeypatch.setattr(attention_gate,'supported',lambda *a:True)
    monkeypatch.setattr(attention_gate,'load_amd_extension',lambda:None)
    sentinel=object()
    assert attention_gate.wrap_up(module=None,o=None,x=None,native=lambda:sentinel) is sentinel


hip_only=pytest.mark.skipif(os.environ.get('FIREPROT_RUN_HIP_TESTS')!='1' or not torch.version.hip,
                            reason='requires explicit HIP tests')


@hip_only
def test_gate_kernel_preserves_inputs_and_extremes():
    assert os.environ.get('HIP_VISIBLE_DEVICES') in {'6','7'}
    extension=load_amd_extension();assert extension is not None
    with torch.inference_mode():
        gate=torch.linspace(-1000,1000,4096,device='cuda').reshape(32,128)
        value=torch.randn_like(gate);saved_gate=gate.clone();saved_value=value.clone()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): result=extension.sigmoid_mul_out_f32(gate,value)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(result,torch.sigmoid(gate)*value,atol=2e-6,rtol=2e-6)
        assert torch.equal(gate,saved_gate) and torch.equal(value,saved_value)
        assert result.data_ptr()!=value.data_ptr()
        with pytest.raises(RuntimeError): extension.sigmoid_mul_out_f32(gate[:,1:],value[:,1:])


@hip_only
def test_full_wrap_and_exact_gate(monkeypatch):
    monkeypatch.setenv('LAYERNORM_TYPE','torch')
    monkeypatch.setenv('FIREPROT_ENABLE_ATTENTION_GATE_1536','1')
    from protenix.model.triangular.layers import Attention
    torch.manual_seed(719)
    with torch.inference_mode():
        module=Attention(128,128,128,32,4).cuda().eval()
        for weight in module.parameters():weight.normal_(std=0.02)
        x=torch.randn(512,1536,128,device='cuda');o=torch.randn(512,1536,4,32,device='cuda')
        assert attention_gate.supported(module,o,x)
        assert not attention_gate.supported(module,o[:1],x[:1])
        module.train();assert not attention_gate.supported(module,o,x);module.eval()
        saved=o.clone();expected=module._wrap_up(o,x)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            assert not attention_gate.supported(module,o,x)
        with torch.autocast('cuda',dtype=torch.float32):
            assert attention_gate.supported(module,o,x)
            result=attention_gate.wrap_up(module=module,o=o,x=x,native=lambda:pytest.fail('unexpected fallback'))
        torch.testing.assert_close(result,expected,atol=2e-5,rtol=2e-5)
        assert torch.equal(o,saved)
