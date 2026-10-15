"""Opt-in N1000 pLM operations with Rust capability selection and exact arithmetic."""
from contextlib import contextmanager
import ctypes
from pathlib import Path
import torch
from fireprot.ops.registry import OperationRegistry, signature_from_tensor


@contextmanager
def optimized_scope(model, request, length):
    name = request['workload']['model_name']
    if length != 1000 or str(torch.__version__) != '2.12.0a0+git78258b9' or torch.version.hip != '7.14.60850':
        raise ValueError('pLM optimization is qualified only for N1000 and the pinned ROCm Torch build')
    architecture = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    if architecture != 'gfx90a':
        raise ValueError('pLM optimization requires gfx90a')
    registry = OperationRegistry()
    state = dict(implementation='unused_attention_mean' if name == 'esm2-3b' else 'hip_bf16_rope', calls=0)
    changes = []
    library = None
    if name != 'esm2-3b':
        library = ctypes.CDLL(str(Path(request['resources']['hip_rope']['path']).resolve()))
        entry = getattr(library, 'fireprot_esmc_rope' if name == 'esmc-fold2-6b' else 'fireprot_atlaslm_rope')
        entry.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int, ctypes.c_void_p]
        entry.restype = ctypes.c_int
        state['library'] = request['resources']['hip_rope']

    def run(callback, *args, **kwargs):
        state['calls'] += 1
        return callback(*args, **kwargs)
    registry.register(operation='plm_inference', name=state['implementation'], function=run,
        priority=10, version='n1000-v1', supports=lambda s: dict(s.attributes).get('eligible') == 'True')

    def dispatch(tensor, eligible, native, candidate, *args, **kwargs):
        signature = signature_from_tensor('plm_inference', tensor, architecture=architecture,
            mode='inference', mask='native', inplace=False, attributes=dict(eligible=eligible, model=name))
        selected = registry.resolve(signature)
        if selected.implementation is None:
            return native(*args, **kwargs)
        return selected.implementation.function(candidate, *args, **kwargs)

    try:
        if name == 'esm2-3b':
            for layer in model.layers:
                module = layer.self_attn
                native = module.forward
                def call(*args, _native=native, **kwargs):
                    tensor = kwargs.get('query', args[0] if args else None)
                    eligible = (tensor is not None and tuple(tensor.shape) == (1002, 1, 2560)
                        and tensor.dtype == torch.float32 and tensor.is_cuda and not torch.is_grad_enabled()
                        and not kwargs.get('need_head_weights', False) and not model.training)
                    def candidate(*a, **kw):
                        kw['need_weights'] = False
                        return _native(*a, **kw)
                    return dispatch(tensor, eligible, _native, candidate, *args, **kwargs)
                changes.append((module, native)); module.forward = call
        else:
            heads = 40 if name == 'esmc-fold2-6b' else 36
            for block in model.transformer.blocks:
                module = block.attn.rotary
                native = module.forward
                if name == 'esmc-fold2-6b':
                    module._update_cos_sin_cache(length+2, device=torch.device('cuda:0'), dtype=torch.bfloat16)
                    cos, sin = module._cos_cached, module._sin_cached
                else:
                    cos = module._cos_cached[:length+2].to(torch.bfloat16).contiguous()
                    sin = module._sin_cached[:length+2].to(torch.bfloat16).contiguous()
                def call(q, k, *args, _native=native, _cos=cos, _sin=sin, **kwargs):
                    eligible = (not args and not kwargs and tuple(q.shape) == (1,1002,heads,64)
                        and q.shape == k.shape and q.dtype == k.dtype == torch.bfloat16
                        and q.is_cuda and k.device == q.device == _cos.device == _sin.device
                        and q.is_contiguous() and k.is_contiguous() and not torch.is_grad_enabled()
                        and not model.training and not any(t.requires_grad or t.is_conj() or t.is_neg() for t in (q,k)))
                    # AtlasLM passes pos_id=None positionally. It has the same native zero-offset semantics.
                    if name == 'atlaslm-3b' and args == (None,) and not kwargs:
                        args = ()
                        eligible = (tuple(q.shape) == (1,1002,heads,64) and q.shape == k.shape
                            and q.dtype == k.dtype == torch.bfloat16 and q.is_cuda
                            and q.device == k.device == _cos.device == _sin.device
                            and q.is_contiguous() and k.is_contiguous() and not torch.is_grad_enabled()
                            and not model.training and not any(t.requires_grad or t.is_conj() or t.is_neg() for t in (q,k)))
                    def candidate(a,b):
                        ao,bo = torch.empty_like(a),torch.empty_like(b)
                        code = entry(a.data_ptr(),b.data_ptr(),_cos.data_ptr(),_sin.data_ptr(),
                                     ao.data_ptr(),bo.data_ptr(),1002,torch.cuda.current_stream(a.device).cuda_stream)
                        if code: raise RuntimeError(f'pLM HIP RoPE failed: {code}')
                        return ao,bo
                    return dispatch(q, eligible, _native, candidate, q,k,*args,**kwargs)
                changes.append((module,native));module.forward=call
        yield state
        expected = 36 if name == 'esm2-3b' else 80 if name == 'esmc-fold2-6b' else 48
        if state['calls'] != expected:
            raise RuntimeError(f'pLM optimized coverage mismatch: {state["calls"]} != {expected}')
    finally:
        for module,native in reversed(changes):
            module.forward=native
        state['registry']=registry.manifest()
