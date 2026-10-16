"""Run Rust studies serially, pausing asset copying at a complete-file boundary."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def checked(pin):
 p=Path(pin['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'];return p


def main():
 root=Path(sys.argv[1]);names=sys.argv[2:];plan=json.loads((root/'plan.json').read_text())
 cli=checked(plan['cli']);cases=[c for c in plan['cases'] if c['name'] in names]
 assert len(cases)==len(names)
 transfer_pid=1989212;paused=False
 proc=Path('/proc')/str(transfer_pid)
 if proc.exists():
  cmd=(proc/'cmdline').read_bytes();assert b'/tmp/fireprot-precision-transfer.py' in cmd
  os.kill(transfer_pid,signal.SIGSTOP);paused=True
 try:
  if paused:
   # Parent stops before it can start another file. The current rsync completes
   # normally; no TCP stream or partially copied final file is suspended.
   start=time.monotonic()
   while True:
    active=[]
    for p in Path('/proc').iterdir():
     if not p.name.isdigit():continue
     try:
      fields=(p/'stat').read_text().split(') ',1)[1].split()
      if int(fields[1])==transfer_pid and fields[0]!='Z':active.append(p.name)
     except (OSError,ValueError):continue
    if not active:break
    if time.monotonic()-start>3600:raise TimeoutError('Transfer file did not finish')
    time.sleep(2)
   print('Transfer paused at complete-file boundary',flush=True)
  env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
  for key in ('ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):env.pop(key,None)
  for case in cases:
   directory=root/case['name'];assert not (directory/'measurements').exists()
   spec=checked(case['spec']);print('Starting',case['name'],flush=True)
   with (directory/'driver.log').open('x') as log:
    result=subprocess.run([str(cli),'benchmark-pairs',str(spec),str(directory/'measurements')],cwd='/media/IntelSSD/fireprot',env=env,stdout=log,stderr=subprocess.STDOUT)
   (directory/'driver-status.json').write_text(json.dumps(dict(exit_code=result.returncode))+'\n')
   print('Finished',case['name'],'exit',result.returncode,flush=True)
 finally:
  if paused and proc.exists():os.kill(transfer_pid,signal.SIGCONT);print('Asset transfer resumed',flush=True)

if __name__=='__main__':main()
