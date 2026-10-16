"""Skip only discarded meta-tensor initialization during checkpoint construction."""
from contextlib import contextmanager


@contextmanager
def skip_meta_initialization(enabled=True):
    import torch
    names=('normal_','uniform_','kaiming_uniform_','kaiming_normal_','xavier_uniform_',
           'xavier_normal_','trunc_normal_','constant_','zeros_','ones_')
    original={name:getattr(torch.nn.init,name) for name in names}
    state=dict(meta_calls=0,real_calls=0,skipped=0)
    def wrapper(fn):
        def call(tensor,*args,**kwargs):
            state['meta_calls' if tensor.is_meta else 'real_calls']+=1
            if enabled and tensor.is_meta:
                state['skipped']+=1
                return tensor
            return fn(tensor,*args,**kwargs)
        return call
    try:
        for name,fn in original.items():setattr(torch.nn.init,name,wrapper(fn))
        yield state
    finally:
        for name,fn in original.items():setattr(torch.nn.init,name,fn)
