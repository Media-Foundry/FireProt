"""GPU event measurements of an already numerically screened tensor operation."""
import json
import os
from pathlib import Path
import sys
import torch
from boundary import FusedSoftmax


def main(destination):
    torch.manual_seed(101)
    rows=[]
    kernels={(w,r):FusedSoftmax(w,r) for w in (2,4) for r in (0,1)}
    packed={w:FusedSoftmax(w,1,pack=True) for w in (2,4)}
    with torch.no_grad():
        for n,batch,ending in [(1000,256,False),(1000,256,True),(1536,128,False),(1536,128,True)]:
            scores=torch.randn(batch,4,n,n,device='cuda')*0.5
            bias=torch.randn(n,n,4,device='cuda').permute(2,1,0) if ending else torch.randn(n,n,4,device='cuda').permute(2,0,1)
            bias=bias.unsqueeze(0)*0.1
            mask=torch.zeros(batch,1,1,n,device='cuda')
            work=torch.empty_like(scores)
            # Native uses in-place bias additions to an already materialized score.
            def native():
                work.add_(mask).add_(bias)
                return torch.softmax(work,dim=-1)
            operations={'native':native,**{f'hip-w{w}-reuse{r}':
                (lambda k=k:k(scores,[mask,bias])) for (w,r),k in kernels.items()}}
            operations.update({f'packed-w{w}':(lambda k=k:k(scores,[mask,bias]))
                               for w,k in packed.items()})
            work.copy_(scores)
            reference=native()
            for name,fn in operations.items():
                if name=='native':work.copy_(scores)
                actual=fn()
                assert torch.equal(reference.view(torch.int32),actual.view(torch.int32)),name
                del actual
            del reference
            observations={name:[] for name in operations}
            # Reverse the order between repetitions; score reset is outside events.
            for repeat in range(7):
                names=list(operations)
                if repeat%2:names.reverse()
                for name in names:
                    if name=='native':work.copy_(scores)
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();output=operations[name]();end.record();end.synchronize()
                    if repeat>=2:observations[name].append(start.elapsed_time(end))
                    del output
            rows.append(dict(n=n,batch=batch,ending=ending,bias_strides=list(bias.stride()),observations_ms=observations))
            print(json.dumps(rows[-1]),flush=True)
            del scores,bias,mask,work
    Path(destination).write_text(json.dumps(dict(schema_version=1,rows=rows,seed=101,warmups=2,repetitions=5,
        environment=dict(visible_device=os.environ.get('HIP_VISIBLE_DEVICES'),torch=str(torch.__version__),hip=str(torch.version.hip),device=torch.cuda.get_device_name()),
        scope='GPU events: bias additions plus softmax; excludes QK/PV and score reset; not E2E'),indent=2)+'\n')


if __name__=='__main__':main(sys.argv[1])
