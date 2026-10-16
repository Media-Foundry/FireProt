"""Require Rust to reject a controlled HIP6 conflict before model execution."""
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys

from collect_esmc_pointwise_pairs import pin


def main():
    root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve()
    case=work/'isolation-check';case.mkdir(exist_ok=False)
    binaries=json.loads((work/'binaries.json').read_text())
    for v in binaries.values():assert pin(Path(v['path']))==v
    profile=json.loads((work/'b4-pairs/candidate-profile.json').read_text())
    request=json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
    request.update(run_id='esmc-snapshot-monitor-conflict',output_dir=str(case/'rejected'))
    for name,value in (('request',request),('profile',profile)):(case/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for key in ('ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):env.pop(key,None)
    child=subprocess.Popen([binaries['hold']['path']],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,start_new_session=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout,selectors.EVENT_READ);assert selector.select(15)
        marker=json.loads(child.stdout.readline());assert marker==dict(pid=child.pid,pci_address='0000:8e:00.0')
        with (case/'stdout.log').open('x') as stdout,(case/'stderr.log').open('x') as stderr:
            result=subprocess.run([str(work/'fireprot-bench'),'esm',str(case/'request.json'),str(case/'profile.json')],
                                  cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=120)
        assert result.returncode!=0 and child.poll() is None
        output=case/'rejected';report=json.loads((output/'supervisor.json').read_text())
        assert report['phase']=='failed' and report['device_verified']
        assert 'external GPU processes' in report['failure'] and str(child.pid) in report['failure']
        assert report['physical_device']['pci_address']=='0000:8e:00.0'
        assert not (output/'accepted.json').exists() and not (output/'predictions.json').exists()
        child.stdin.write(b'stop\n');child.stdin.flush();assert child.wait(timeout=10)==0
        query=subprocess.run([binaries['native']['path'],'snapshot-v1','--json'],env=env,capture_output=True,check=True,timeout=10)
        rows=json.loads(query.stdout)['devices'];row,=[r for r in rows if int(r['gpu'])==4]
        assert row['process_list']==[dict(process_info='No running processes detected')]
        record=dict(passed=True,exit_code=result.returncode,controlled_pid=child.pid,supervisor=report,
                    supervisor_pin=pin(output/'supervisor.json'),request=pin(case/'request.json'),profile=pin(case/'profile.json'),
                    script=pin(Path(__file__)),scope='Expected rejection with a controlled test process; no performance observation.')
        (case/'report.json').write_text(json.dumps(record,indent=2)+'\n')
        print('Rust rejected the controlled HIP6 conflict before predictions; device idle again',flush=True)
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        for stream in (child.stdin,child.stdout,child.stderr):stream.close()


if __name__=='__main__':main()
