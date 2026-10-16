"""Qualified N1000 ESM operators; preserve native arithmetic and model heads."""
from contextlib import contextmanager
import ctypes
from pathlib import Path

import torch
from fireprot.ops.registry import OperationRegistry, signature_from_tensor


def frameless_output(module, native, s, affine, affine_mask, sequence_id, chain_id):
    """Retain the native projection of masked positive zero, including any bias."""
    if (module.mask_and_zero_frameless and not module.training and not torch.is_grad_enabled()
            and affine_mask.dtype == torch.bool and not bool(affine_mask.any())):
        zero = torch.zeros((*s.shape[:-1], module.out_proj.in_features), device=s.device, dtype=s.dtype)
        return module.out_proj(zero), True
    return native(s, affine, affine_mask, sequence_id, chain_id), False


@contextmanager
def esm3_frameless_scope(model, batch_size, architecture):
    """Skip geometry whose all-false frame mask forces its input to zero."""
    registry = OperationRegistry()
    state = dict(implementation='esm3_frameless_exact_v1', batch_size=batch_size,
                 architecture=architecture, geometry_calls=0)
    changes = []

    def invoke(callback):
        return callback()

    registry.register(operation='plm_esm_geometry', name='esm3_frameless_exact_v1',
        function=invoke, priority=10, version='n1000-v1',
        supports=lambda signature: dict(signature.attributes).get('eligible') == 'True')
    try:
        for block in model.transformer.blocks:
            if not block.use_geom_attn:
                continue
            module = block.geom_attn
            native = module.forward
            def geometry(s, affine, affine_mask, sequence_id, chain_id,
                         _module=module, _native=native):
                eligible = (tuple(s.shape) == (batch_size, 1002, 1536)
                    and s.dtype == torch.bfloat16 and s.is_cuda and s.device.index == 0
                    and s.is_contiguous() and not model.training and not torch.is_grad_enabled()
                    and not s.requires_grad and not s.is_conj() and not s.is_neg()
                    and affine_mask.dtype == torch.bool and tuple(affine_mask.shape) in ((1, 1002), (batch_size, 1002))
                    and not bool(affine_mask.any()))
                signature = signature_from_tensor('plm_esm_geometry', s,
                    architecture=architecture, mode='inference', mask='all_false', inplace=False,
                    attributes=dict(eligible=eligible, model='esm3-open'))
                selected = registry.resolve(signature)
                if selected.implementation is None:
                    return _native(s, affine, affine_mask, sequence_id, chain_id)
                def candidate():
                    output, used = frameless_output(_module, _native, s, affine, affine_mask,
                                                    sequence_id, chain_id)
                    if not used:
                        raise RuntimeError('ESM3 frameless capability changed after selection')
                    state['geometry_calls'] += 1
                    return output
                return selected.implementation.function(candidate)
            changes.append((module, native))
            module.forward = geometry
        yield state
        if state['geometry_calls'] != 1:
            raise RuntimeError(f'ESM3 frameless coverage mismatch: {state["geometry_calls"]} != 1')
    finally:
        for module, native in reversed(changes):
            module.forward = native
        state['registry'] = registry.manifest()


@contextmanager
def esm_scope(model, request, batch_size):
    """Called only after the public scope checks hardware, runtime and N1000."""
    name = request['workload']['model_name']
    if name not in ('esmc-fold2-6b', 'esm2-3b', 'esm3-open'):
        raise ValueError('hip_esm requires an ESM model')
    resource = request['resources']['hip_esm']
    lib = ctypes.CDLL(str(Path(resource['path']).resolve()))
    registry = OperationRegistry()
    state = dict(implementation='hip_esm_exact_v1', library=resource, batch_size=batch_size,
                 rope_calls=0, swiglu_calls=0, geometry_calls=0, gelu_calls=0, calls=0)
    changes = []
    old_gelu = None

    def invoke(callback):
        return callback()

    registry.register(operation='plm_esm', name='hip_esm_exact_v1', function=invoke,
        priority=10, version='n1000-v1', supports=lambda s: dict(s.attributes).get('eligible') == 'True')

    def eligible(x, shape, dtype):
        return (tuple(x.shape) == shape and x.dtype == dtype and x.is_cuda and x.device.index == 0
                and x.is_contiguous() and not model.training and not torch.is_grad_enabled()
                and not x.requires_grad and not x.is_conj() and not x.is_neg())

    def dispatch(x, matched, kind, native, candidate):
        sig = signature_from_tensor('plm_esm', x, architecture='gfx90a', mode='inference',
            mask='native', inplace=False, attributes=dict(eligible=matched, model=name, kind=kind))
        selected = registry.resolve(sig)
        if selected.implementation is None:
            return native()
        state[kind+'_calls'] += 1
        state['calls'] += 1
        return selected.implementation.function(candidate)

    try:
        if name == 'esm2-3b':
            import esm.modules
            old_gelu = esm.modules.gelu
            gelu = lib.fireprot_esm2_gelu
            gelu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            gelu.restype = ctypes.c_int

            def activation(x):
                matched = (eligible(x, (1002,batch_size,10240), torch.float32)
                           or eligible(x, (batch_size,1002,2560), torch.float32))
                def candidate():
                    out = torch.empty_like(x)
                    code = gelu(x.data_ptr(), out.data_ptr(), x.numel(), torch.cuda.current_stream(x.device).cuda_stream)
                    if code: raise RuntimeError(f'ESM2 HIP GELU failed: {code}')
                    return out
                return dispatch(x, matched, 'gelu', lambda: old_gelu(x), candidate)
            esm.modules.gelu = activation
            # Preserve the already-qualified omission of unused attention means.
            for layer in model.layers:
                module = layer.self_attn; native = module.forward
                def attention(*args, _native=native, **kwargs):
                    tensor = kwargs.get('query', args[0] if args else None)
                    if (tensor is not None and tuple(tensor.shape) == (1002,batch_size,2560)
                            and tensor.dtype == torch.float32 and tensor.is_cuda and not model.training
                            and not torch.is_grad_enabled() and not kwargs.get('need_head_weights', False)):
                        kwargs['need_weights'] = False
                    return _native(*args, **kwargs)
                changes.append((module,native)); module.forward = attention
        else:
            heads = 40 if name == 'esmc-fold2-6b' else 24
            width = 6912 if name == 'esmc-fold2-6b' else 4096
            rope = lib.fireprot_esm_rope
            rope.argtypes = [ctypes.c_void_p]*6+[ctypes.c_int]*3+[ctypes.c_void_p]
            rope.restype = ctypes.c_int
            swiglu = lib.fireprot_esm_swiglu
            swiglu.argtypes = [ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p]
            swiglu.restype = ctypes.c_int

            def swiglu_tensor(x):
                out = torch.empty((*x.shape[:-1],width),device=x.device,dtype=x.dtype)
                code = swiglu(x.data_ptr(),out.data_ptr(),batch_size*1002,width,
                              torch.cuda.current_stream(x.device).cuda_stream)
                if code: raise RuntimeError(f'ESM HIP SwiGLU failed: {code}')
                return out

            for block in model.transformer.blocks:
                module = block.attn.rotary; native = module.forward
                if module.interleaved or module.scale_base is not None:
                    raise ValueError('ESM HIP RoPE requires noninterleaved, unscaled native rotation')
                def rotate(q,k,*args,_m=module,_native=native,**kwargs):
                    matched = (not args and not kwargs
                        and eligible(q,(batch_size,1002,heads,64),torch.bfloat16)
                        and eligible(k,(batch_size,1002,heads,64),torch.bfloat16))
                    def candidate():
                        _m._update_cos_sin_cache(1002,device=q.device,dtype=q.dtype)
                        c,s = _m._cos_cached,_m._sin_cached
                        if tuple(c.shape) != (1002,32) or not c.is_contiguous() or not s.is_contiguous():
                            raise ValueError('ESM RoPE cache layout changed')
                        a,b = torch.empty_like(q),torch.empty_like(k)
                        code = rope(q.data_ptr(),k.data_ptr(),c.data_ptr(),s.data_ptr(),a.data_ptr(),b.data_ptr(),
                                    1002,batch_size,heads,torch.cuda.current_stream(q.device).cuda_stream)
                        if code: raise RuntimeError(f'ESM HIP RoPE failed: {code}')
                        return a,b
                    return dispatch(q,matched,'rope',lambda:_native(q,k,*args,**kwargs),candidate)
                changes.append((module,native));module.forward=rotate
                module = block.ffn if name == 'esmc-fold2-6b' else block.ffn[2]
                native = module.forward
                def activation(x,_m=module,_native=native):
                    shape = (batch_size,1002,2560 if name == 'esmc-fold2-6b' else 2*width)
                    def candidate():
                        if name == 'esmc-fold2-6b':
                            normalized = torch.nn.functional.layer_norm(x,(_m.hidden_size,),
                                _m.layer_norm_weight,_m.layer_norm_bias,_m.eps)
                            projected = torch.nn.functional.linear(normalized,_m.fc1_weight)
                            return torch.nn.functional.linear(swiglu_tensor(projected),_m.fc2_weight)
                        return swiglu_tensor(x)
                    return dispatch(x,eligible(x,shape,torch.bfloat16),'swiglu',lambda:_native(x),candidate)
                changes.append((module,native));module.forward=activation
            if name == 'esm3-open':
                for block in model.transformer.blocks:
                    if not block.use_geom_attn: continue
                    module = block.geom_attn; native = module.forward
                    def geometry(s,affine,affine_mask,sequence_id,chain_id,_m=module,_native=native):
                        out,used = frameless_output(_m,_native,s,affine,affine_mask,sequence_id,chain_id)
                        state['geometry_calls'] += int(used)
                        return out
                    changes.append((module,native));module.forward=geometry
        yield state
        expected = dict(gelu_calls=37) if name == 'esm2-3b' else dict(
            rope_calls=80 if name == 'esmc-fold2-6b' else 48,
            swiglu_calls=80 if name == 'esmc-fold2-6b' else 48)
        if name == 'esm3-open': expected['geometry_calls'] = 1
        if any(state[key] != count for key,count in expected.items()):
            raise RuntimeError(f'ESM operator coverage mismatch: {state}, expected {expected}')
    finally:
        if old_gelu is not None:
            import esm.modules
            esm.modules.gelu = old_gelu
        for module,native in reversed(changes): module.forward = native
        state['registry'] = registry.manifest()
