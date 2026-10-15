"""Read-only pretrained ESMC profiling and graph screen; not folding E2E evidence."""
import json, os, sys, time, statistics
from pathlib import Path
import torch
from fireprot_workers.plm import load_model
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.transport import save_arrays
request=json.loads(Path(sys.argv[1]).read_text())
output=Path(sys.argv[2]); output.mkdir(exist_ok=False)
torch.set_num_threads(8);torch.manual_seed(request['seed'])
_,sequence=read_sequence(request['input']['path'])
model,tokens,native=load_model(request,sequence)
report={'model':request['workload']['model_name'],'length':len(sequence),'hip_visible_devices':os.environ['HIP_VISIBLE_DEVICES'],'torch':str(torch.__version__),'hip':str(torch.version.hip),'precision':'bfloat16','seed':request['seed'],'cache':'warm; same process; no feature cache','layers':len(model.transformer.blocks)}
def timer(fn,n=10):
 values=[]
 for _ in range(n):
  torch.cuda.synchronize();start=time.perf_counter();r=fn();torch.cuda.synchronize();values.append(time.perf_counter()-start)
 return values,r
with torch.inference_mode():
 baseline=native()
 save_arrays(output/'native.npz',baseline)
 for _ in range(3):native()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  native();torch.cuda.synchronize()
 prof.export_chrome_trace(str(output/'trace.json'))
 report['top_operations']=[{'name':e.key,'count':e.count,'self_device_us':e.self_device_time_total,'cpu_us':e.self_cpu_time_total} for e in sorted(prof.key_averages(),key=lambda e:e.self_device_time_total,reverse=True)[:25]]
 # The native unpadded, one-chain forward chooses sequence_id=None. The
 # wrapper's bool_mask.all() host sync is outside graph capture; verify its
 # exact tensor result before attempting capture.
 def core():
  hidden,_,_,_=model.transformer(model.embed(tokens),sequence_id=None,layers_to_collect=[],output_attentions=False)
  return dict(tokens=tokens,embeddings=hidden)
 report['core_exact']=all(torch.equal(v,core()[k]) for k,v in baseline.items())
 try:
  if not report['core_exact']:raise RuntimeError('Core does not match native forward')
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):core()
  torch.cuda.current_stream().wait_stream(stream)
  graph=torch.cuda.CUDAGraph()
  start=time.perf_counter()
  with torch.cuda.graph(graph):captured=core()
  torch.cuda.synchronize();report['capture_seconds']=time.perf_counter()-start
  def replay():graph.replay();return captured
  report['graph_exact']=all(torch.equal(v,replay()[k]) for k,v in baseline.items())
  report['pairs']=[]
  for order in [('native','graph'),('graph','native'),('native','graph')]:
   row={}
   for name in order:
    values,result=timer(native if name=='native' else replay)
    row[name]=values
    if not all(torch.equal(v,result[k]) for k,v in baseline.items()):raise RuntimeError('Numerical drift')
   report['pairs'].append(row)
  save_arrays(output/'graph.npz',captured)
  report['native_median']=statistics.median(sum([p['native'] for p in report['pairs']],[]))
  report['graph_median']=statistics.median(sum([p['graph'] for p in report['pairs']],[]))
  report['speedup']=report['native_median']/report['graph_median']
 except Exception as exc:report['graph_error']=repr(exc)
report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
