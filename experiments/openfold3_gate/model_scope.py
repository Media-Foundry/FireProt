"""Exact-shape experimental triangle projection gate/packing fusion."""
from contextlib import contextmanager
import hashlib,inspect,json,textwrap
from pathlib import Path
import torch
from ..triangle_memory.copy_runtime import CopyRuntime
from ..triangle_memory.replay import patched,native,SOURCE_SHA
from .runtime import Gate

@contextmanager
def triangle_scope(model,output,library):
 if not torch.is_inference_mode_enabled():raise RuntimeError('Inference mode required')
 root=Path(__file__).parents[3];gate_path=root/'gate.so'
 assert hashlib.sha256(gate_path.read_bytes()).hexdigest()=='b7a980bf469a0f13581985033f904541cae5783eab01d4579054588e75c3614c'
 rt=CopyRuntime(library);gate=Gate(str(gate_path))
 src=inspect.getsource(patched).replace('def patched(rt):','def gated(rt, gate):')
 marker='    scope = dict(vars(native), _copy=rt)'
 old='p = linear_g(pair)\n            p.sigmoid_()\n            p *= linear_p(pair)\n            p *= mask'
 new='p = _gate(linear_g(pair), linear_p(pair), mask)'
 replacement='    old='+repr(old)+'\n    assert source.count(old)==1\n    source=source.replace(old,'+repr(new)+')\n    scope = dict(vars(native), _copy=rt, _gate=gate)'
 assert src.count(marker)==1;src=src.replace(marker,replacement);scope=dict(globals());exec(src,scope);candidate=scope['gated'](rt,gate)
 cls=native.TriangleMultiplicativeUpdate;original=cls._inference_forward;accepted=fallback=0
 def forward(self,z,mask=None,inplace_chunk_size=None,with_add=True,use_triton_triangle_kernels=False):
  nonlocal accepted,fallback
  if (tuple(z.shape)==(1,1536,1536,128) and z.dtype==torch.float32 and z.is_contiguous() and inplace_chunk_size==256 and with_add and not use_triton_triangle_kernels and self.c_hidden==128):
   accepted+=1;return candidate(self,z,mask,inplace_chunk_size,with_add,False)
  fallback+=1;return original(self,z,mask,inplace_chunk_size,with_add,use_triton_triangle_kernels)
 cls._inference_forward=forward
 try:yield
 finally:
  cls._inference_forward=original
  Path(output,'gate-fusion.json').write_text(json.dumps(dict(accepted=accepted,fallback=fallback,gate_calls=gate.calls,copies=rt.calls,copy_fallbacks=rt.fallbacks,library_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest()),indent=2)+'\n')
