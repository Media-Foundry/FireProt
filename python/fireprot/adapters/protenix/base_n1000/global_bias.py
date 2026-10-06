"""Audit/cache static global diffusion biases for the Base N1000 S1 route."""
from contextlib import contextmanager
import json
from pathlib import Path
import torch


@contextmanager
def install(model, output, mode="on"):
    if mode not in ('off','audit','on'):raise ValueError('Unknown bias cache mode')
    if mode=='off':
        yield
        return
    diffusion=model.diffusion_module
    blocks=list(diffusion.diffusion_transformer.blocks)
    if model.training or len(blocks)!=24 or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('Strict 24-block Base inference required')
    handles=[]
    methods=[]
    caches={}
    weights={}
    source=None
    source_key=None
    records=dict(mode=mode,conditioning_calls=0,attention_calls=0,created=0,hits=0,
                 audited_equal=0,cache_bytes=0,error=None)

    def key(tensor):
        return (tensor.data_ptr(),tensor._version,tuple(tensor.shape),tuple(tensor.stride()),str(tensor.dtype),str(tensor.device))

    def before_conditioning(module,args,kwargs):
        nonlocal source,source_key
        pair=kwargs.get('pair_z')
        if (torch.is_grad_enabled() or pair is None or not pair.is_cuda
                or pair.dtype!=torch.float32 or tuple(pair.shape)!=(1000,1000,128)
                or torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32
                or not kwargs.get('use_conditioning',True)):
            raise RuntimeError('Diffusion pair cache contract mismatch')
        if source is None:source,source_key=pair,key(pair)
        if pair is not source or key(pair)!=source_key:
            raise RuntimeError('Diffusion pair source changed during sampling')
        records['conditioning_calls']+=1

    try:
        handles.append(diffusion.diffusion_conditioning.register_forward_pre_hook(before_conditioning,with_kwargs=True))
        for index,block in enumerate(blocks):
            module=block.attention_pair_bias
            if module.n_heads!=16 or module.create_offset_ln_z or module.training:
                raise RuntimeError('Unsupported global attention configuration')
            weights[index]=(key(module.linear_nobias_z.weight),key(module.layernorm_z.weight))
            def check_weights(index=index,module=module):
                if (key(module.linear_nobias_z.weight),key(module.layernorm_z.weight))!=weights[index]:
                    raise RuntimeError('Bias projection parameters changed')
            def observe(attention,args,kwargs,index=index,check_weights=check_weights):
                check_weights()
                bias=kwargs.get('attn_bias')
                if source is None or bias is None or bias.dtype!=torch.float32 or tuple(bias.shape)!=(1,16,1000,1000):
                    raise RuntimeError('Unexpected global bias tensor')
                records['attention_calls']+=1
                if index not in caches:
                    if not torch.isfinite(bias).all().item():raise RuntimeError('Nonfinite first bias')
                    caches[index]=(bias,key(bias))
                    records['created']+=1
                    records['cache_bytes']+=bias.numel()*bias.element_size()
                else:
                    cached,stamp=caches[index]
                    if key(cached)!=stamp:raise RuntimeError('Retained bias was modified')
                    if mode=='audit':
                        if not torch.equal(bias.view(torch.int32),cached.view(torch.int32)):
                            raise RuntimeError('Diffusion bias differs across sampling steps')
                        records['audited_equal']+=1
                    elif bias is not cached:
                        raise RuntimeError('Cached attention did not use retained bias')
            handles.append(module.attention.register_forward_pre_hook(observe,with_kwargs=True))
            original=module.standard_multihead_attention
            previous=module.__dict__.get('standard_multihead_attention')
            methods.append((module,previous))
            def standard(q,kv,z,inplace_safe=False,enable_efficient_fusion=False,
                         index=index,module=module,original=original,check_weights=check_weights):
                if torch.is_grad_enabled() or not enable_efficient_fusion:
                    raise RuntimeError('Qualified fused-bias inference required')
                check_weights()
                if mode=='on' and index in caches:
                    records['hits']+=1
                    return module.attention(q_x=q,kv_x=kv,attn_bias=caches[index][0],inplace_safe=inplace_safe)
                return original(q,kv,z,inplace_safe=inplace_safe,enable_efficient_fusion=enable_efficient_fusion)
            module.standard_multihead_attention=standard
        yield
        if records['conditioning_calls']!=200 or records['attention_calls']!=4800 or records['created']!=24:
            raise RuntimeError('Incomplete Base diffusion cache coverage')
        if mode=='audit' and records['audited_equal']!=4776:raise RuntimeError('Incomplete bias audit')
        if mode=='on' and records['hits']!=4776:raise RuntimeError('Incomplete cache reuse')
    except BaseException as error:
        records['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        for handle in handles:handle.remove()
        for module,previous in methods:
            if previous is None:delattr(module,'standard_multihead_attention')
            else:module.standard_multihead_attention=previous
        Path(output,'base-diffusion-bias-cache.json').write_text(json.dumps(records,indent=2)+'\n')
