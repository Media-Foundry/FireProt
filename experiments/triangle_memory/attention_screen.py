"""Build and screen exact head-last K packing separately from model timing."""
import json
from pathlib import Path
import sys
import torch
import torch.utils.deterministic
from fireprot.backends.amd._native_hip import load
from attention_runtime import KeyPacker
from screen import bits_equal, measure


@torch.inference_mode()
def main(destination):
    out=Path(destination);out.mkdir(exist_ok=False)
    _,receipt=load('fireprot_triangle_attention_copy_v1',
        Path(__file__).with_name('attention_copy.cpp'),Path('/opt/rocm'),
        extra_files=[__file__],versions={'c_abi':'triangle-key-copy-v1'})
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    pack=KeyPacker(receipt['library'])
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.manual_seed(101)
    base=torch.randint(-(2**31),2**31-1,(128,1536,4,32),device='cuda',dtype=torch.int32)
    base.flatten()[:4]=torch.tensor([0,-2147483648,2143289345,2139095040],device='cuda',dtype=torch.int32)
    key=base.view(torch.float32).permute(0,2,1,3)
    expected=key.transpose(-1,-2).reshape(512,32,1536)
    actual=pack(key)
    assert bits_equal(actual,expected) and pack.calls==1
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):other=pack(key)
    torch.cuda.current_stream().wait_stream(stream)
    assert bits_equal(other,expected) and actual.data_ptr()!=other.data_ptr()
    samples=[]
    for reverse in (False,True,False):
        fns=[lambda:key.transpose(-1,-2).reshape(512,32,1536),lambda:pack(key)]
        values={i:measure(fns[i]) for i in ([1,0] if reverse else [0,1])}
        samples.append(dict(native_ms=values[0],hip_ms=values[1]))
    # Already head-major input must take the native fallback and remain exact.
    n=pack.calls;fallback=pack(key.contiguous());assert pack.calls==n and bits_equal(fallback,expected)
    (out/'screen.json').write_text(json.dumps(dict(correctness_passed=True,samples=samples,
        calls=pack.calls,shape=list(key.shape),strides=list(key.stride()),
        torch=str(torch.__version__),hip=torch.version.hip,seed=101),indent=2)+'\n')


if __name__=='__main__':main(sys.argv[1])
