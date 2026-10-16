"""Build a frozen combined-query probe and a separate fault-injection binary."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    root=Path('/media/IntelSSD/fireprot'); work=Path(sys.argv[1]).resolve()
    previous=root/'outputs/experiments/esmc-native-monitor-20261016'
    binaries={}
    for kind in ('native','fake'):
        spec=json.loads((previous/'build-v2'/f'{kind}-build-spec.json').read_text())
        build=spec['build'];build['package']=f'esmc_snapshot_{kind}'
        build['versions']['recipe']=f'esmc-snapshot-{kind}-v1'
        for entry in build['files']:
            if entry['name']=='source':entry['path']=str(work/'fireprot_smi_probe.cpp')
            if entry['name']=='fake':entry['path']=str(work/'esmc_smi_fake.cpp')
        spec_path=work/f'{kind}-build-spec.json';receipt_path=work/f'{kind}-build-receipt.json'
        with spec_path.open('x') as f:json.dump(spec,f,indent=2);f.write('\n')
        with receipt_path.open('x') as f:
            subprocess.run([str(root/'target/release/fireprot'),'build-hip',str(spec_path),str(work/'build-cache')],
                           stdout=f,cwd=root,check=True)
        receipt=json.loads(receipt_path.read_text());assert len(receipt['receipt']['artifacts'])==1
        name,sha=next(iter(receipt['receipt']['artifacts'].items()))
        path=Path(receipt['directory'])/name;binaries[kind]=pin(path);assert binaries[kind]['sha256']==sha
    binaries['hold']=json.loads((previous/'qualification-v3/binaries.json').read_text())['hold']
    assert pin(Path(binaries['hold']['path']))==binaries['hold']
    with (work/'binaries.json').open('x') as f:json.dump(binaries,f,indent=2);f.write('\n')
    print('Both native and fake probes built; existing qualified HIP6 holder pinned.',flush=True)


if __name__=='__main__':main()
