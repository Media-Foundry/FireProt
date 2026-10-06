"""Separate audit/cache experiment for six local atom-attention pair biases."""
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import torch


@contextmanager
def install(model, output, cache_contract=None):
    mode=os.environ.get('FIREPROT_BASE_LOCAL_BIAS_CACHE','off')
    if mode not in ('off','audit','on'):raise ValueError('Unknown local bias mode')
    if cache_contract is not None and mode!='on':
        raise RuntimeError('Combined graph trial requires cache mode on')
    if mode=='off':
        yield
        return
    diffusion=model.diffusion_module
    encoder=diffusion.atom_attention_encoder
    decoder=diffusion.atom_attention_decoder
    blocks=list(encoder.atom_transformer.diffusion_transformer.blocks)+list(decoder.atom_transformer.diffusion_transformer.blocks)
    if model.training or len(blocks)!=6 or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('Strict six-block atom inference required')
    signature=inspect.signature(diffusion.forward)
    sources=None
    source_stamps=None
    handles=[]
    methods=[]
    caches={}
    records=dict(mode=mode,denoiser_calls=0,attention_calls=0,created=0,hits=0,
                 audited_equal=0,cache_bytes=0,error=None)

    def key(t):
        return (t.data_ptr(),t._version,tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device))

    # The encoder's static pair update also depends on its fixed trunk-conditioning
    # projections/MLP, beyond each attention layer's own bias weights.
    parameters=list(encoder.parameters())+list(decoder.parameters())
    parameter_stamps=[key(p) for p in parameters]
    owner=None
    def before_denoise(module,args,kwargs):
        nonlocal sources,source_stamps,owner
        bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        data=bound.arguments
        current=[data.get(name) for name in ('p_lm','c_l','s_trunk')]
        if (torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
                or torch.backends.cuda.matmul.allow_tf32 or not data.get('use_conditioning',True)
                or any(t is None or not t.is_cuda or t.dtype!=torch.float32 for t in current)):
            raise RuntimeError('Unsupported static atom-pair conditioning')
        stream=torch.cuda.current_stream(current[0].device).cuda_stream
        if sources is None:
            sources,source_stamps,owner=current,[key(t) for t in current],stream
        if any(a is not b for a,b in zip(current,sources)) or [key(t) for t in current]!=source_stamps or stream!=owner:
            raise RuntimeError('Static atom-pair source or stream changed')
        if [key(p) for p in parameters]!=parameter_stamps:
            raise RuntimeError('Atom-conditioning parameters changed')
        records['denoiser_calls']+=1

    def check_retained():
        for cached,stamp in caches.values():
            if key(cached)!=stamp:
                raise RuntimeError('Graph-retained bias was modified')

    if cache_contract is not None:
        cache_contract.guards.append(check_retained)

    try:
        handles.append(diffusion.register_forward_pre_hook(before_denoise,with_kwargs=True))
        for index,block in enumerate(blocks):
            module=block.attention_pair_bias
            if module.n_heads!=4 or module.training:raise RuntimeError('Unsupported local attention')
            def observe(attention,args,kwargs,index=index):
                bias=kwargs.get('trunked_attn_bias')
                if sources is None or bias is None or bias.dtype!=torch.float32 or bias.ndim!=5 or tuple(bias.shape[:2])!=(1,4) or tuple(bias.shape[-2:])!=(32,128):
                    raise RuntimeError('Unexpected local bias shape')
                records['attention_calls']+=1
                if index not in caches:
                    if not torch.isfinite(bias).all().item():raise RuntimeError('Nonfinite local bias')
                    caches[index]=(bias,key(bias))
                    records['created']+=1
                    records['cache_bytes']+=bias.numel()*bias.element_size()
                else:
                    cached,stamp=caches[index]
                    if key(cached)!=stamp:raise RuntimeError('Retained local bias modified')
                    if mode=='audit':
                        if not torch.equal(bias.contiguous().view(torch.int32),cached.contiguous().view(torch.int32)):
                            raise RuntimeError('Local bias differs between denoising steps')
                        records['audited_equal']+=1
                    elif bias is not cached:raise RuntimeError('Retained local bias not consumed')
            handles.append(module.attention.register_forward_pre_hook(observe,with_kwargs=True))
            original=module.local_multihead_attention
            previous=module.__dict__.get('local_multihead_attention')
            methods.append((module,previous))
            def local(q,kv,z,n_queries=32,n_keys=128,inplace_safe=False,chunk_size=None,
                      index=index,module=module,original=original):
                if torch.is_grad_enabled() or n_queries!=32 or n_keys!=128:
                    raise RuntimeError('Unsupported local attention geometry')
                if mode=='on' and index in caches:
                    records['hits']+=1
                    return module.attention(q_x=q,kv_x=kv,trunked_attn_bias=caches[index][0],
                        n_queries=n_queries,n_keys=n_keys,inplace_safe=inplace_safe,chunk_size=chunk_size)
                return original(q,kv,z,n_queries=n_queries,n_keys=n_keys,inplace_safe=inplace_safe,chunk_size=chunk_size)
            module.local_multihead_attention=local
        yield
        executions=200
        if cache_contract is not None:
            cache_contract.validate_completed()
            executions=cache_contract.python_executions
        if records['denoiser_calls']!=200 or records['attention_calls']!=6*executions or records['created']!=6:
            raise RuntimeError('Incomplete atom bias coverage')
        if mode=='audit' and records['audited_equal']!=1194:raise RuntimeError('Incomplete local audit')
        if mode=='on' and records['hits']!=6*(executions-1):raise RuntimeError('Incomplete local reuse')
    except BaseException as error:
        records['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        for handle in handles:handle.remove()
        for module,previous in methods:
            if previous is None:delattr(module,'local_multihead_attention')
            else:module.local_multihead_attention=previous
        Path(output,'base-local-bias-cache.json').write_text(json.dumps(records,indent=2)+'\n')
