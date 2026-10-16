"""Build an isolated B8/B16 ESMC extension without changing kernel arithmetic."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def pin(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve();work.mkdir(exist_ok=False)
    headers=work/'headers';headers.mkdir()
    files={'source':root/'experiments/plm/esmc_pointwise.hip','compat':root/'experiments/plm/esmc_pointwise_compat.hip',
           'qualified_esm':root/'kernels/amd/fireprot_esm.hip'}
    before={name:pin(path) for name,path in files.items()}
    destinations={name:(headers if name=='qualified_esm' else work)/path.name for name,path in files.items()}
    for name,path in files.items():shutil.copy2(path,destinations[name])
    p=destinations['source'];s=p.read_text();needle='(rows!=1002 && rows!=2004 && rows!=4008)';assert s.count(needle)==1
    p.write_text(s.replace(needle,'(rows!=1002 && rows!=2004 && rows!=4008 && rows!=8016 && rows!=16032)'))
    p=destinations['qualified_esm'];s=p.read_text();needle='(batch!=1 && batch!=2 && batch!=4)';assert s.count(needle)==1
    p.write_text(s.replace(needle,'(batch!=1 && batch!=2 && batch!=4 && !(Heads==40 && (batch==8 || batch==16)))'))
    spec=json.loads((root/'outputs/experiments/esmc-pointwise-20261016/build-spec-v4.json').read_text())
    b=spec['build'];b['package']='esmc_large_batch';b['versions']['recipe']='esmc-b8-b16-pointwise-v1'
    for f in b['files']:
        if f['name'] in destinations:f['path']=str(destinations[f['name']])
    for tree in b['trees']:
        if tree['name']=='qualified_esm':tree['path']=str(headers)
    b['arguments']['include']=['/opt/rocm/include',str(headers)]
    (work/'build-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    with (work/'build-receipt.json').open('x') as log:
        subprocess.run([str(root/'target/release/fireprot'),'build-hip',str(work/'build-spec.json'),str(work/'kernel-cache')],cwd=root,stdout=log,check=True)
    receipt=json.loads((work/'build-receipt.json').read_text());name,sha=next(iter(receipt['receipt']['artifacts'].items()))
    library=pin(Path(receipt['directory'])/name);assert library['sha256']==sha
    (work/'library.json').write_text(json.dumps(library,indent=2)+'\n')
    (work/'source-changes.json').write_text(json.dumps(dict(before=before,after={k:pin(p) for k,p in destinations.items()},
        change='Only admission bounds: ESMC heads40 RoPE batch8/16 and pointwise rows8016/16032; arithmetic and compile flags unchanged.'),indent=2)+'\n')
    print('B8/B16 prototype built',flush=True)


if __name__=='__main__':main()
