"""Qualify native/opt-in ESMFold2 outputs through Rust on physical HIP7."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

control = Path(sys.argv[1]).resolve()
env = dict(os.environ, HIP_VISIBLE_DEVICES='7', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
env.pop('CUDA_VISIBLE_DEVICES', None)
env.pop('ROCR_VISIBLE_DEVICES', None)

def write(name, value):
    (control / name).write_text(json.dumps(value, indent=2) + '\n')

def assert_device_free():
    # PCI 0000:93:00.0, location_id 0x9300, independently verified by HIP probe.
    nodes = Path('/sys/class/kfd/kfd/topology/nodes')
    ids = []
    for node in nodes.iterdir():
        props = dict(line.split(maxsplit=1) for line in (node / 'properties').read_text().splitlines())
        if props.get('location_id') == str(0x9300) and props.get('domain') == '0':
            ids.append((node / 'gpu_id').read_text().strip())
    assert ids == ['8466'], ids
    owners = []
    for proc in Path('/sys/class/kfd/kfd/proc').iterdir():
        try:
            used = int((proc / 'vram_8466').read_text())
        except FileNotFoundError:
            continue
        if used:
            owners.append((proc.name, used))
    assert not owners, owners
    result = subprocess.run(['/media/IntelSSD/fireprot/runtime/fireprot-hip-probe'],
                            env=env, capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(result.stdout)['pci_address'] == '0000:93:00.0'

try:
    root=control
    prep=json.loads((root/'preparation.json').read_text())
    for item in [prep['cli'],prep['bridge'],prep['profile'],prep['assets'],prep['screen']]+prep['specs']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    assert json.loads((root/'triangle-screen.json').read_text())['all_exact']
    cli=prep['cli']['path']
    records=[]
    for arm in ('native','fireprot'):
        assert_device_free()
        with (root/(arm+'-qualification.log')).open('x') as log:
            child=subprocess.Popen([cli,'measure-inference',str(root/(arm+'-spec.json')),str(root/(arm+'-measurement'))],env=env,stdout=log,stderr=subprocess.STDOUT)
            write('qualification-status.json',dict(stage=arm,pid=os.getpid(),child_pid=child.pid))
            code=child.wait()
        assert code==0,(arm,code)
        request=json.loads((root/(arm+'-request.json')).read_text());output=Path(request['output_dir'])
        accepted=json.loads((output/'accepted.json').read_text())
        for item in accepted['metadata']+accepted['outputs']:
            path=output/item['path']
            assert path.resolve().is_relative_to(output.resolve())
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        pred=json.loads((output/'predictions.json').read_text())
        runtime=json.loads((output/'operator-runtime.json').read_text())
        records.append(dict(arm=arm,output=str(output),model_seconds=pred['model_seconds'],request_seconds=pred['request_seconds'],runtime=runtime))
    left,right=[Path(r['output']) for r in records]
    cmp=subprocess.run([cli,'compare-npz',str(left/'sample-0.npz'),str(right/'sample-0.npz')],capture_output=True,text=True)
    arrays=json.loads(cmp.stdout)
    exact=len(arrays)==12 and all(x['bitwise_equal'] for x in arrays.values())
    cif_exact=(left/'native/sample-0.cif').read_bytes()==(right/'native/sample-0.cif').read_bytes()
    assert records[0]['runtime']['counts']==records[1]['runtime']['counts']
    write('qualification-result.json',dict(records=records,arrays=arrays,all12_exact=exact,full_cif_exact=cif_exact,comparison_exit_code=cmp.returncode,scope='One instrumented qualification per arm; not a repeated E2E speedup claim.'))
    assert exact and cif_exact
    write('qualification-driver-status.json',dict(exit_code=0,all12_exact=exact,full_cif_exact=cif_exact))
except BaseException as error:
    write('qualification-driver-status.json',dict(exit_code=1,error=repr(error),pid=os.getpid()))
    raise
