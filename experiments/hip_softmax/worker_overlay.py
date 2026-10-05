"""Experimental upstream attention seam; native QK/PV and all work retained."""
from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def fused_attention_probe(output):
    import torch
    import openfold3.core.model.primitives.attention as attention
    from .boundary import FusedSoftmax, compatible
    kernels={n:FusedSoftmax(2,1,pack=True) for n in (1000,1536)}
    original=attention._attention
    enabled=os.environ.get('FIREPROT_FUSED_SOFTMAX')=='1'
    report=dict(enabled=enabled,calls=0,selected=0,fallback=0,
                launch_config={str(n):{'waves':2,'reuse':1,'pack_bias':True}
                               for n in (1000,1536)},
                kernel=kernels[1000].receipt)

    def call(query,key,value,biases,use_high_precision=False):
        report['calls']+=1
        eligible=(enabled and not torch.is_grad_enabled() and query.is_cuda
            and query.dtype==key.dtype==value.dtype==torch.float32
            and query.ndim==4 and query.shape==key.shape==value.shape
            and query.shape[1]==4 and query.shape[-1]==32
            and key.device==value.device==query.device
            and compatible((*query.shape[:-1],key.shape[-2]),query.device,biases))
        if not eligible:
            report['fallback']+=1
            return original(query,key,value,biases,use_high_precision=use_high_precision)
        # This is the same FP32 autocast context and einsum order as upstream.
        with torch.amp.autocast('cuda',dtype=torch.float32):
            scores=torch.einsum('...qc, ...kc->...qk',query,key)
            if not scores.is_contiguous():
                # Preserve already computed QK; never silently copy an unmatched layout.
                report['fallback']+=1
                for bias in biases:scores+=bias
                scores=attention.softmax_no_cast(scores,dim=-1)
            else:
                scores=kernels[query.shape[-2]](scores,biases)
                report['selected']+=1
            result=torch.einsum('...qk, ...kc->...qc',scores,value)
        return result.to(dtype=query.dtype)

    attention._attention=call
    try:
        yield
    finally:
        attention._attention=original
        Path(output,'hip-softmax-audit.json').write_text(json.dumps(report,indent=2)+'\n')
