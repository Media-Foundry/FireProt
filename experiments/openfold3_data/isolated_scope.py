"""Experimental single-item CPU featurization with isolated worker RNG state."""
from contextlib import contextmanager
from pathlib import Path
import json,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
@contextmanager
def isolated_data_scope(output):
 original=DataLoader._get_iterator;records=[]
 class IsolatedIterator(_SingleProcessDataLoaderIter):
  def __init__(self,loader):
   assert loader.num_workers==0 and len(loader.dataset)==1 and len(loader)==1
   assert loader.worker_init_fn is not None and loader.generator is not None
   self.worker_init=loader.worker_init_fn;self.done=False
   super().__init__(loader)
  def _next_data(self):
   if self.done:return super()._next_data()
   self.done=True;start=time.perf_counter()
   py_state=random.getstate();np_state=np.random.get_state();threads=torch.get_num_threads()
   deterministic=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
   try:
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
     torch.set_num_threads(1)
     torch.use_deterministic_algorithms(False)
     torch.manual_seed(self._base_seed)
     self.worker_init(0)
     result=super()._next_data()
     def check(value):
      if isinstance(value,torch.Tensor):assert value.device.type=='cpu'
      elif isinstance(value,dict):
       for x in value.values():check(x)
      elif isinstance(value,(list,tuple)):
       for x in value:check(x)
     check(result)
     return result
   finally:
    random.setstate(py_state);np.random.set_state(np_state)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(deterministic,warn_only=warn)
    records.append(dict(base_seed=self._base_seed,worker_id=0,cpu_threads=1,seconds=time.perf_counter()-start,parent_threads=threads,parent_deterministic=deterministic))
 def iterator(loader):
  if loader.num_workers!=0:return original(loader)
  return IsolatedIterator(loader)
 DataLoader._get_iterator=iterator
 try:yield
 finally:
  DataLoader._get_iterator=original
  Path(output,'isolated-data.json').write_text(json.dumps(dict(scope='Single input only; worker-0 initialization in isolated RNG/thread context; model settings restored',records=records),indent=2)+'\n')
