"""Check native bmm(out=), private in-place probabilities, and final PV bits."""
import json
from pathlib import Path
import sys

import torch
import torch.utils.deterministic
from experiments.hip_softmax.owned_scores import OwnedScoreSoftmax
from experiments.mfma_qk.probe_fusion import operands, bitwise_mismatch_count


@torch.inference_mode()
def main(output):
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library('cublas')
    torch.manual_seed(101)
    batch, n = 128, 1536
    softmax = OwnedScoreSoftmax()
    retained = torch.empty(batch,4,n,n,device='cuda')
    records=[]
    for case in ('random','cancellation','subnormal','signed_zero','nonfinite'):
        q,k=operands(case,batch)
        scores=torch.bmm(q,k).view(batch,4,n,n)
        actual=torch.bmm(q,k,out=retained.view(batch*4,n,n))
        assert actual.data_ptr()==retained.data_ptr()
        differences=bitwise_mismatch_count(scores,retained)
        records.append(dict(case=case,mode='raw_bmm_out',differences=differences))
        assert differences==0
        variants=('zero','partial_outer','additive','all_masked') if case=='random' else ('zero',)
        for variant in variants:
            mask=torch.zeros(batch if variant=='partial_outer' else 1,1,1,n,device='cuda')
            if variant=='partial_outer':
                for i in range(batch):mask[i,...,n//2+i:]=-float('inf')
            elif variant=='additive':mask.normal_(0,.25)
            elif variant=='all_masked':mask.fill_(-float('inf'))
            bias=torch.randn(1,4,n,n,device='cuda')*.125
            if case!='random':bias.zero_()
            for transpose in (False,True):
                b=bias.transpose(-1,-2) if transpose else bias
                expected=scores.clone();expected.add_(mask);expected.add_(b)
                expected=torch.softmax(expected,-1)
                torch.bmm(q,k,out=retained.view(batch*4,n,n))
                result=softmax(retained,(mask,b))
                assert result.data_ptr()==retained.data_ptr()
                differences=bitwise_mismatch_count(expected,result)
                records.append(dict(case=case,variant=variant,transposed_bias=transpose,
                    mode='probabilities',differences=differences))
                assert differences==0
                value=torch.randn(batch,4,n,32,device='cuda')*.125
                pv=torch.einsum('...qk,...kc->...qc',expected,value)
                actual_pv=torch.einsum('...qk,...kc->...qc',result,value)
                assert bitwise_mismatch_count(pv,actual_pv)==0
                del expected,result,value,pv,actual_pv
        del q,k,scores,actual,mask,bias,b
    assert len(records)==21 and softmax.storage_fallbacks==0
    Path(output).write_text(json.dumps(dict(correctness_passed=True,records=records,
        batch=batch,retained_bytes=retained.numel()*4,softmax=softmax.experiment_manifest(),
        torch=str(torch.__version__),hip=torch.version.hip,strict=True,seed=101),indent=2)+'\n')


if __name__=='__main__':main(sys.argv[1])
