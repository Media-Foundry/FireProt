"""Scoped ESM candidate screens. No changes to upstream source or default routes."""
from contextlib import contextmanager
import ctypes
import torch


@contextmanager
def candidate_scope(model, name, mode, library=None):
    changes=[]
    state=dict(mode=mode, rope_calls=0, swiglu_calls=0, geometry_calls=0)
    state['gelu_calls']=0
    lib=ctypes.CDLL(library) if library else None
    old_gelu=None
    try:
        if name=='esm3-open' and 'geometry' in mode:
            for block in model.transformer.blocks:
                if not block.use_geom_attn: continue
                module=block.geom_attn
                native=module.forward
                def geometry(s, affine, affine_mask, sequence_id, chain_id, _m=module, _native=native):
                    if (_m.mask_and_zero_frameless and not _m.training and not torch.is_grad_enabled()
                            and affine_mask.dtype==torch.bool and not bool(affine_mask.any())):
                        state['geometry_calls']+=1
                        # Native masked_fill writes positive zero before this projection.
                        # Retain out_proj (and its bias, if present) and autocast semantics.
                        zero=torch.zeros((*s.shape[:-1],_m.out_proj.in_features),device=s.device,dtype=s.dtype)
                        return _m.out_proj(zero)
                    return _native(s,affine,affine_mask,sequence_id,chain_id)
                changes.append((module,native));module.forward=geometry
        if name=='esm2-3b' and lib and 'hip' in mode:
            import esm.modules
            old_gelu=esm.modules.gelu
            entry=lib.fireprot_esm2_gelu
            entry.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
            entry.restype=ctypes.c_int
            def gelu(x):
                if not x.is_cuda or x.dtype!=torch.float32 or not x.is_contiguous():return old_gelu(x)
                out=torch.empty_like(x)
                code=entry(x.data_ptr(),out.data_ptr(),x.numel(),torch.cuda.current_stream().cuda_stream)
                if code:raise RuntimeError(f'ESM2 GELU HIP {code}')
                state['gelu_calls']+=1
                return out
            esm.modules.gelu=gelu
        if lib and 'hip' in mode and name!='esm2-3b':
            rope=lib.fireprot_esm_rope
            rope.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*3+[ctypes.c_void_p]
            rope.restype=ctypes.c_int
            swiglu=lib.fireprot_esm_swiglu
            swiglu.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p]
            swiglu.restype=ctypes.c_int
            for block in model.transformer.blocks:
                module=block.attn.rotary; native=module.forward
                def rotate(q,k,*args,_m=module,_native=native,**kwargs):
                    if args or kwargs or q.dtype!=torch.bfloat16 or not q.is_contiguous() or not k.is_contiguous():
                        return _native(q,k,*args,**kwargs)
                    _m._update_cos_sin_cache(q.shape[1],device=q.device,dtype=q.dtype)
                    a,b=torch.empty_like(q),torch.empty_like(k)
                    code=rope(q.data_ptr(),k.data_ptr(),_m._cos_cached.data_ptr(),_m._sin_cached.data_ptr(),
                              a.data_ptr(),b.data_ptr(),q.shape[1],q.shape[0],q.shape[2],torch.cuda.current_stream().cuda_stream)
                    if code: raise RuntimeError(f'ESM RoPE HIP {code}')
                    state['rope_calls']+=1
                    return a,b
                changes.append((module,native));module.forward=rotate
                module=block.ffn if name=='esmc-fold2-6b' else block.ffn[2]
                native=module.forward
                def activation(x,_native=native,_m=module):
                    if x.dtype!=torch.bfloat16 or not x.is_contiguous(): return _native(x)
                    if name=='esmc-fold2-6b':
                        x=torch.nn.functional.layer_norm(x,(_m.hidden_size,),_m.layer_norm_weight,_m.layer_norm_bias,_m.eps)
                        x=torch.nn.functional.linear(x,_m.fc1_weight)
                    width=x.shape[-1]//2
                    out=torch.empty((*x.shape[:-1],width),device=x.device,dtype=x.dtype)
                    code=swiglu(x.data_ptr(),out.data_ptr(),x.numel()//(2*width),width,torch.cuda.current_stream().cuda_stream)
                    if code: raise RuntimeError(f'ESM SwiGLU HIP {code}')
                    state['swiglu_calls']+=1
                    if name=='esmc-fold2-6b': return torch.nn.functional.linear(out,_m.fc2_weight)
                    return out
                changes.append((module,native));module.forward=activation
        yield state
    finally:
        if old_gelu is not None:
            import esm.modules
            esm.modules.gelu=old_gelu
        for module,native in reversed(changes): module.forward=native
