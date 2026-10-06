"""Replay captured triangle attention with only key packing replaced."""
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import textwrap

import torch
import torch.utils.deterministic
from attention_runtime import KeyPacker
from screen import bits_equal
from experiments.mfma_qk import workspace_attention as workspace_module
from experiments.mfma_qk.workspace_dispatch import load_workspace, WorkspaceDispatcher
from fireprot.adapters.openfold3.attention import AttentionDispatcher, attention_scope
from fireprot.backends.amd.softmax_fp32 import SoftmaxFP32
from openfold3.core.model.layers.triangular_attention import TriangleAttention


@torch.inference_mode()
def main(build, captures, controls, output):
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.preferred_blas_library('cublas')
    torch.manual_seed(101)
    pack=KeyPacker(json.loads(Path(build).read_text())['library'])
    original=workspace_module.WorkspaceFusion.__call__
    assert hashlib.sha256(Path(workspace_module.__file__).read_bytes()).hexdigest()=='47d2efa63c4402367af9282f05411b2caa123dc9f8826b5314f82afae8cc966d'
    source=textwrap.dedent(inspect.getsource(original))
    old='key.transpose(-1, -2).reshape(self.batch * 4, 32, 1536)'
    assert source.count(old)==1
    seen={}
    def packed(key):
        signature=str((tuple(key.shape),tuple(key.stride())))
        seen[signature]=seen.get(signature,0)+1
        return pack(key)
    scope=dict(vars(workspace_module),_key_packer=packed)
    exec(compile(source.replace(old,'_key_packer(key)'),'<key-copy-experiment>','exec'),scope)
    candidate=scope['__call__']
    control_sha=hashlib.sha256(Path(controls).read_bytes()).hexdigest()
    results=[]
    try:
        with load_workspace(controls,control_sha) as workspace:
            dispatcher=WorkspaceDispatcher(AttentionDispatcher(SoftmaxFP32()),workspace)
            with attention_scope(dispatcher):
                for path in sorted(Path(captures).glob('trunk_pairformer-tri_att_*.pt')):
                    snap=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                    module=TriangleAttention(128,32,4,starting=snap['starting']).eval().cuda()
                    module.load_state_dict(snap['state_dict'],strict=True)
                    x=snap['z'].cuda();mask=snap['mask'].cuda()
                    samples={'native':[],'copy':[]};reference=None
                    for mode in ['native','copy']+['native','copy','copy','native']*3:
                        workspace_module.WorkspaceFusion.__call__=candidate if mode=='copy' else original
                        inp=x.clone()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();value=module(inp,mask=mask,**snap['flags']);end.record();end.synchronize()
                        if reference is None:reference=value.clone()
                        else:assert bits_equal(reference,value),(path.name,mode)
                        samples[mode].append(start.elapsed_time(end))
                        del inp,value
                    a=statistics.median(samples['native'][1:]);b=statistics.median(samples['copy'][1:])
                    results.append(dict(capture=str(path),native_ms=a,copy_ms=b,speedup=a/b,
                        exact=True,samples_ms=samples,flags=snap['flags']))
                    del module,x,mask,reference,snap
            assert len(results)==2 and pack.calls>0
    finally:
        workspace_module.WorkspaceFusion.__call__=original
    Path(output).write_text(json.dumps(dict(correctness_passed=True,results=results,
        calls=pack.calls,key_signatures=seen,control_sha256=control_sha,
        torch=str(torch.__version__),hip=torch.version.hip,seed=101),indent=2)+'\n')


if __name__=='__main__':main(*sys.argv[1:])
