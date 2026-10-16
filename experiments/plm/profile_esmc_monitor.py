"""Time unchanged AMD SMI subprocesses during an isolated ESMC invocation."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from collect_esmc_pointwise_pairs import archive, pin


def main():
    root = Path('/media/IntelSSD/fireprot')
    work = Path(sys.argv[1]).resolve(); work.mkdir(exist_ok=False)
    profile = json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-exact.json').read_text())
    request = json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
    real = Path(profile['amd_smi'])
    real_pin = pin(real)
    calls = work/'smi-calls.jsonl'
    wrapper = work/'timed-amd-smi'
    wrapper.write_text('''#!/usr/bin/python3
import hashlib, json, os, subprocess, sys, time
begin = time.perf_counter()
wall_begin = time.time_ns()
result = subprocess.run([REAL, *sys.argv[1:]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
elapsed = time.perf_counter()-begin
record = dict(argv=sys.argv[1:], elapsed_seconds=elapsed, started_unix_ns=wall_begin,
              returncode=result.returncode, stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
              stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
with open(RECORD, 'a') as f: f.write(json.dumps(record)+'\\n')
sys.stdout.buffer.write(result.stdout)
sys.stderr.buffer.write(result.stderr)
sys.exit(result.returncode)
'''.replace('REAL', repr(str(real))).replace('RECORD', repr(str(calls))))
    wrapper.chmod(0o755)
    profile['amd_smi'] = str(wrapper)
    request.update(run_id='esmc-monitor-diagnostic', output_dir=str(work/'prediction'), hip_device=6)
    for name, data in (('request',request),('profile',profile)):
        (work/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    env = dict(os.environ, HIP_VISIBLE_DEVICES='6')
    for name in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'): env.pop(name, None)
    cli = Path(profile['workers']).parent/'target/release/fireprot'
    start = time.perf_counter()
    with (work/'invocation.log').open('x') as log:
        subprocess.run([str(cli),'esm',str(work/'request.json'),str(work/'profile.json')],
                       cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    elapsed = time.perf_counter()-start
    assert pin(real)==real_pin
    prediction, actual = archive(work/'prediction')
    reference = root/'outputs/experiments/esmc-load-20261016/validation/native-pairs/baseline-1'
    _, expected = archive(reference)
    assert actual==expected
    supervisor = json.loads((work/'prediction/supervisor.json').read_text())
    assert supervisor['phase']=='succeeded' and supervisor['device_verified']
    assert supervisor['physical_device']['pci_address']=='0000:8e:00.0'
    observations = [json.loads(line) for line in calls.read_text().splitlines()]
    for item in observations:
        assert item['returncode']==0 and item['argv'] in (['list','--json'],['process','--json'])
    groups = {operation:[r['elapsed_seconds'] for r in observations if r['argv'][0]==operation]
              for operation in ('list','process')}
    assert len(groups['process']) == supervisor['isolation_samples']
    assert len(groups['list']) == supervisor['isolation_samples']+1
    report = dict(schema_version=1,
        boundary='Instrumented diagnostic, not an E2E comparison. Wrapper adds overhead. SMI wall times overlap worker execution; their sum is not recoverable critical-path time.',
        hip_device=6,precision='native BF16',length=1000,batch_size=4,seed=101,
        cache_state='Fresh worker, full checkpoint hashes, unflushed caches, offline-built existing HIP operators.',
        invocation_seconds=elapsed,supervisor_seconds=supervisor['elapsed_seconds'],
        model_forward_seconds=prediction['model_seconds'],all_arrays_exact=True,
        real_amd_smi=real_pin,wrapper=pin(wrapper),cli=pin(cli),request=pin(work/'request.json'),
        profile=pin(work/'profile.json'),accepted=pin(work/'prediction/accepted.json'),
        reference_accepted=pin(reference/'accepted.json'),calls=pin(calls),
        script=pin(Path(__file__)),collector_helper=pin(Path(__file__).with_name('collect_esmc_pointwise_pairs.py')),
        isolation_samples=supervisor['isolation_samples'],observations=observations,
        operation_seconds={k:dict(count=len(v),sum=sum(v),median=statistics.median(v),min=min(v),max=max(v))
                           for k,v in groups.items()})
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='observations'},indent=2))


if __name__=='__main__': main()
