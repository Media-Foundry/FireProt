"""Audit stale inventory at a model boundary without overlapping GPU timing."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path(sys.argv[1]);parent=int(sys.argv[2]);proc=Path('/proc')/str(parent)
assert b'/tmp/fireprot-e2e-refresh-run.py' in (proc/'cmdline').read_bytes()
assert (proc/'stat').read_text().split(') ',1)[1].split()[0]=='T'
audit=root/'openfold3-environment-refresh';audit.mkdir(exist_ok=False);completed=False

def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
try:
 begin=time.monotonic()
 while True:
  active=[]
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    f=(p/'stat').read_text().split(') ',1)[1].split()
    if int(f[1])==parent and f[0]!='Z':active.append(p.name)
   except (OSError,ValueError):continue
  if not active:break
  if time.monotonic()-begin>7200:raise TimeoutError('Current model did not finish')
  time.sleep(2)
 profile=json.loads((root/'openfold3-n1536/baseline-profile.json').read_text());oldpin=profile['environment_inventory'];oldpath=Path(oldpin['path']);assert pin(oldpath)==oldpin
 old=json.loads(oldpath.read_text());spec=audit/'spec.json';spec.write_text(json.dumps(old['specification'],indent=2)+'\n')
 print('Current model ended; capturing the unchanged full inventory scope',flush=True)
 begin=time.monotonic()
 with (audit/'current.json').open('x') as f:subprocess.run([str(root/'fireprot'),'capture-environment',str(spec)],cwd='/media/IntelSSD/fireprot',stdout=f,check=True)
 new=json.loads((audit/'current.json').read_text());assert new['specification']==old['specification']
 a={e['name']:e for e in old['files']};b={e['name']:e for e in new['files']}
 changed=[dict(name=k,before=a[k],after=b[k]) for k in sorted(a.keys()&b.keys()) if a[k]!=b[k]]
 result=dict(status='captured_pending_review',old_inventory=oldpin,new_inventory=pin(audit/'current.json'),same_specification=True,capture_seconds=time.monotonic()-begin,old_files=len(a),new_files=len(b),added=[b[k] for k in sorted(b.keys()-a.keys())],removed=[a[k] for k in sorted(a.keys()-b.keys())],changed=changed,scheduler_parent_paused=parent)
 (audit/'diff.json').write_text(json.dumps(result,indent=2)+'\n');completed=True
 print(json.dumps(dict(added=len(result['added']),removed=len(result['removed']),changed=len(changed),capture_seconds=result['capture_seconds'])),flush=True)
finally:
 if not completed and proc.exists():os.kill(parent,signal.SIGCONT)
