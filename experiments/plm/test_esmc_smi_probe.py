"""Fault-inject the native monitor and verify real HIP6 PID detection/removal."""
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import statistics
import subprocess
import sys
import time


def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def pids(rows, selected):
    found=[r for r in rows if int(r['gpu'])==selected]
    assert len(found)==1
    values=found[0]['process_list']
    if values==[dict(process_info='No running processes detected')]:return set()
    result=[int(v['process_info']['pid']) for v in values]
    assert len(set(result))==len(result) and all(p>0 for p in result)
    return set(result)


def main():
    work=Path(sys.argv[1]).resolve()
    binaries=json.loads((work/'binaries.json').read_text())
    for value in binaries.values():assert pin(value['path'])==value
    env=dict(os.environ)
    for name in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):
        env.pop(name,None)
    report=dict(schema_version=1,binaries=binaries,script=pin(__file__),faults=[],parity=[],holders=[])
    def save(): (work/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    fake=binaries['fake']['path'];native=binaries['native']['path'];official='/usr/bin/amd-smi'
    report['official']=pin(official)
    for case in ('active','empty','resize','init_error','shutdown_error','version_error','unsupported_version',
                 'socket_error','zero_sockets','too_many_sockets','socket_growth','processor_error',
                 'zero_processors','too_many_processors','processor_growth','null_processor','duplicate_processor',
                 'bdf_error','large_domain','duplicate_bdf','process_error','second_device_error','zero_pid',
                 'duplicate_pid','bad_resize','oversize_resize','continuous_growth','success_overflow'):
        result=subprocess.run([fake,'process','--json'],env=dict(env,FIREPROT_TEST_SMI_CASE=case),
                              capture_output=True,timeout=5)
        good=case in ('active','empty','resize')
        if good:
            assert result.returncode==0 and not result.stderr
            rows=json.loads(result.stdout)
            for index in (0,1):
                values=pids(rows,index)
                assert values==({4242} if case=='active' else set() if case=='empty' else set(range(100000000,100000300)))
        else:assert result.returncode!=0 and result.stdout==b'' and result.stderr
        report['faults'].append(dict(case=case,returncode=result.returncode,stderr=result.stderr.decode().strip()))
    for args in ([],['reset','--json'],['process'],['list','--json','extra'],['list','--csv']):
        result=subprocess.run([fake,*args],env=env,capture_output=True,timeout=5)
        assert result.returncode==64 and result.stdout==b''
    report['invalid_argument_cases']=5
    save()
    def query(program,operation):
        start=time.perf_counter()
        result=subprocess.run([program,operation,'--json'],env=env,capture_output=True,timeout=10,check=True)
        seconds=time.perf_counter()-start
        assert result.stdout
        return json.loads(result.stdout),seconds
    def mapping(rows):return {int(r['gpu']):r['bdf'].lower() for r in rows}
    reference,_=query(official,'list');actual,_=query(native,'list')
    assert mapping(reference)==mapping(actual)
    ids=[i for i,bdf in mapping(actual).items() if bdf=='0000:8e:00.0']
    assert ids==[4];selected=ids[0]
    report['device_mapping']=mapping(actual)
    rows,_=query(official,'process');assert pids(rows,selected)==set(), 'HIP6 must be idle before this test'
    for operation in ('list','process'):
        pairs=[]
        for order in ((False,True),(True,False),(False,True)):
            pair={}
            for candidate in order:
                rows,seconds=query(native if candidate else official,operation)
                if operation=='list': assert mapping(rows)==mapping(reference)
                else: assert pids(rows,selected)==set()
                pair['native' if candidate else 'official']=seconds
            pairs.append(pair)
        report['parity'].append(dict(operation=operation,pairs=pairs,
            median_seconds={a:statistics.median(p[a] for p in pairs) for a in ('official','native')}))
    save()
    for trial in range(2):
        child=subprocess.Popen([binaries['hold']['path']],env=dict(env,HIP_VISIBLE_DEVICES='6'),
                               stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        record=dict(trial=trial,pid=child.pid,queries=[])
        report['holders'].append(record)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout,selectors.EVENT_READ)
                assert selector.select(15),'HIP holder did not become ready'
            line=child.stdout.readline();assert line,child.stderr.read().decode()
            marker=json.loads(line)
            assert marker==dict(pid=child.pid,pci_address='0000:8e:00.0')
            ready=time.perf_counter()
            # The first new native process is queried immediately: no one-second
            # delay or process-list warmup is added to hide an initialization issue.
            for candidate in (True,False,True,False):
                before=time.perf_counter()
                rows,seconds=query(native if candidate else official,'process')
                observed=pids(rows,selected)
                assert child.poll() is None and observed=={child.pid}
                record['queries'].append(dict(native=candidate,seconds=seconds,
                    started_after_ready_seconds=before-ready,pids=sorted(observed)))
            child.stdin.write(b'stop\n');child.stdin.flush()
            assert child.wait(timeout=10)==0
            rows,seconds=query(native,'process');assert pids(rows,selected)==set()
            record['removal_seconds']=seconds
            rows,_=query(official,'process');assert pids(rows,selected)==set()
            record['passed']=True
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            for stream in (child.stdin,child.stdout,child.stderr):stream.close()
            save()
    for value in binaries.values():assert pin(value['path'])==value
    assert pin(official)==report['official']
    report['passed']=True;save()
    print(json.dumps(dict(passed=True,fault_cases=len(report['faults']),parity=report['parity'],holders=report['holders']),indent=2))


if __name__=='__main__':main()
