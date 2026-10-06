"""Balanced replay of the double-LDS candidate against accepted fused attention."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.utils.deterministic
from attention_runtime import KeyPacker
from split_workspace import make_call, native
from experiments.mfma_qk.benchmark_attention import PackedFusion
from experiments.mfma_qk.workspace_dispatch import load_workspace, WorkspaceDispatcher
from fireprot.adapters.openfold3.attention import AttentionDispatcher, attention_scope
from fireprot.backends.amd.softmax_fp32 import SoftmaxFP32
from openfold3.core.model.layers.triangular_attention import TriangleAttention


@torch.inference_mode()
def main(key_build, captures, controls, candidate_control, output):
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.preferred_blas_library('cublas')
    torch.manual_seed(101)
    pack=KeyPacker(json.loads(Path(key_build).read_text())['library'])
    method=make_call('fusion',pack,None)
    candidate_kernel=PackedFusion(json.loads(Path(candidate_control).read_text()),128)
    original=native.WorkspaceFusion.__call__
    sha=hashlib.sha256(Path(controls).read_bytes()).hexdigest()
    results=[]
    try:
        with load_workspace(controls,sha) as workspace:
            fused_entry=workspace._pool.entry
            dispatcher=WorkspaceDispatcher(AttentionDispatcher(SoftmaxFP32()),workspace)
            with attention_scope(dispatcher):
                for path in sorted(Path(captures).glob('trunk_pairformer-tri_att_*.pt')):
                    snap=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                    module=TriangleAttention(128,32,4,starting=snap['starting']).eval().cuda()
                    module.load_state_dict(snap['state_dict'],strict=True)
                    x=snap['z'].cuda();mask=snap['mask'].cuda()
                    samples={'fusion':[],'double':[]};reference=None
                    for mode in ['fusion','double']+['fusion','double','double','fusion']*3:
                        native.WorkspaceFusion.__call__=method
                        workspace._pool.entry=fused_entry if mode=='fusion' else candidate_kernel.entry
                        inp=x.clone()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();value=module(inp,mask=mask,**snap['flags']);end.record();end.synchronize()
                        if reference is None:reference=value.clone()
                        else:assert torch.equal(reference.contiguous().view(torch.int32),value.contiguous().view(torch.int32)),(path.name,mode)
                        samples[mode].append(start.elapsed_time(end))
                        del inp,value
                    a=statistics.median(samples['fusion'][1:]);b=statistics.median(samples['double'][1:])
                    results.append(dict(capture=str(path),fusion_ms=a,double_ms=b,speedup=a/b,
                        exact=True,samples_ms=samples,flags=snap['flags']))
                    del module,x,mask,reference,snap
            assert len(results)==2 and pack.calls>0
    finally:
        native.WorkspaceFusion.__call__=original
    Path(output).write_text(json.dumps(dict(correctness_passed=True,results=results,
        key_pack_calls=pack.calls,candidate_kernel=candidate_kernel.receipt,
        modes=['combined-v12 fused QK/softmax + HIP key packing',
               'double-LDS fused QK/softmax + same HIP key packing'],
        control_sha256=sha,torch=str(torch.__version__),hip=torch.version.hip,seed=101,
        cache_state='Same process, one retained pool; first call per arm excluded; six measured calls per arm in balanced order; exact checks outside timing'),indent=2)+'\n')


if __name__=='__main__':main(*sys.argv[1:])
