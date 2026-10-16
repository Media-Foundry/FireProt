"""Queue Base after the folding study, retaining exclusive timing and transfer pause."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path(sys.argv[1]);predecessor=int(sys.argv[2]);transfer=int(sys.argv[3]);case=root/'base-n1000'
p=Path('/proc')/str(predecessor)
if p.exists():assert b'/tmp/fireprot-e2e-refresh-run.py' in (p/'cmdline').read_bytes()
print('Waiting for folding queue',predecessor,flush=True)
while p.exists():
 try:
  if (p/'stat').read_text().split(') ',1)[1].split()[0]=='Z':break
 except FileNotFoundError:break
 time.sleep(.5)
for name in ('esmfold2-n1000','openfold3-n1536','mini-n1000','mini-n1536','mini-esm-n1000','atlasfold-n1000','atlasfold-n1536','tiny-n1000'):
 assert (root/name/'driver-status.json').is_file(),name
paused=False;tp=Path('/proc')/str(transfer)
if tp.exists():
 assert b'/tmp/fireprot-precision-transfer.py' in (tp/'cmdline').read_bytes()
 os.kill(transfer,signal.SIGSTOP);paused=True
try:
 if paused:
  start=time.monotonic()
  while True:
   active=[]
   for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
     f=(p/'stat').read_text().split(') ',1)[1].split()
     if int(f[1])==transfer and f[0]!='Z':active.append(p.name)
    except (OSError,ValueError):continue
   if not active:break
   if time.monotonic()-start>3600:raise TimeoutError('Current asset did not finish')
   time.sleep(2)
 print('Starting Base with transfer paused',flush=True)
 assert not (case/'launch.log').exists()
 env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
 for k in ('ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):env.pop(k,None)
 with (case/'launch.log').open('x') as log:result=subprocess.run(['bash',str(case/'run.sh')],cwd='/media/IntelSSD/fireprot',env=env,stdout=log,stderr=subprocess.STDOUT)
 print('Base finished',result.returncode,flush=True)
finally:
 if paused and tp.exists():os.kill(transfer,signal.SIGCONT);print('Asset transfer resumed after Base',flush=True)
