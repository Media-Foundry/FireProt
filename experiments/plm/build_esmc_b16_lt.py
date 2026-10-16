"""Bind a fully qualified offline B16 FFN choice into a direct HIP library."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve();report=json.loads((work/'screen/report.json').read_text())
    assert report['passed'] and report['assets_rehashed_after'] and report['all_80_projections_exact'] and report['full_model_exact']
    rows=[r for r in report['algorithms'] if r[:2]==['GemmTunableOp_BFloat16_TN','tn_13824_16032_2560_ld_2560_2560_13824']]
    assert len(rows)==1
    match=re.fullmatch(r'Gemm_Hipblaslt_(\d+)',rows[0][2]);assert match,'No fixed hipBLASLt candidate selected'
    algorithm=int(match[1]);assert report['forward_speedup']>1
    directory=work/'direct';directory.mkdir(exist_ok=False)
    source=root/'experiments/plm/esmc_lt_fixed.hip';text=source.read_text()
    needle='(rows != 1002 && rows != 4008)';assert text.count(needle)==1;text=text.replace(needle,'rows != 16032')
    needle='std::vector<int> indices{rows == 1002 ? 4335 : 4139};';assert text.count(needle)==1
    text=text.replace(needle,f'std::vector<int> indices{{{algorithm}}};')
    path=directory/'esmc_b16_lt.hip';path.write_text(text)
    spec=json.loads((root/'outputs/experiments/esmc-gemm-order-20261016/lt-build-spec.json').read_text())
    spec['build']['package']='esmc_b16_lt';spec['build']['versions']['recipe']=f'esmc-b16-ffn-algo{algorithm}-v1'
    for entry in spec['build']['files']:
        if entry['name']=='source':entry['path']=str(path)
    (directory/'build-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    with (directory/'build-receipt.json').open('x') as f:
        subprocess.run([str(root/'target/release/fireprot'),'build-hip',str(directory/'build-spec.json'),str(directory/'kernel-cache')],cwd=root,stdout=f,check=True)
    receipt=json.loads((directory/'build-receipt.json').read_text());name,sha=next(iter(receipt['receipt']['artifacts'].items()))
    library=pin(Path(receipt['directory'])/name);assert library['sha256']==sha
    (directory/'library.json').write_text(json.dumps(library,indent=2)+'\n')
    (directory/'selection.json').write_text(json.dumps(dict(algorithm=algorithm,screen=pin(work/'screen/report.json'),source_before=pin(source),source_after=pin(path),library=library),indent=2)+'\n')
    print('Built B16 fixed algorithm',algorithm,flush=True)


if __name__=='__main__':main()
