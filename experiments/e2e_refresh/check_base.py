"""Enforce the existing Base summary-confidence threshold after each pair."""
import json
import math
from pathlib import Path
import sys

case=Path(sys.argv[1]);index=int(sys.argv[2])
a=json.loads((case/f'native-{index}/predictions.json').read_text())['predictions'][0]['confidence']
b=json.loads((case/f'optimized-{index}/predictions.json').read_text())['predictions'][0]['confidence']
def leaves(x,path=''):
 if isinstance(x,dict):return {k:v for key,value in x.items() for k,v in leaves(value,path+'/'+key).items()}
 if isinstance(x,list):return {k:v for i,value in enumerate(x) for k,v in leaves(value,path+'/'+str(i)).items()}
 return {path:x}
a=leaves(a);b=leaves(b);assert a.keys()==b.keys()
delta=[]
for k,v in a.items():
 if isinstance(v,bool) or not isinstance(v,(int,float)):assert v==b[k]
 else:
  assert math.isfinite(v) and math.isfinite(b[k]);delta.append(abs(v-b[k]))
assert max(delta,default=0)<=0.001
out=case/f'confidence-{index}.json';out.write_text(json.dumps(dict(passed=True,max_abs_delta=max(delta,default=0),numeric_leaves=len(delta),tolerance=0.001),indent=2)+'\n')
