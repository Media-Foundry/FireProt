"""Build key-tile variants on the verified padded, grouped CK attention probe."""
from __future__ import annotations
import argparse
import json
import pathlib
import subprocess
from benchmarks.protenix.profile_inference import REPO_ROOT, _sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=pathlib.Path,required=True)
    parser.add_argument('--no-sequence-padding',action='store_true',help='Exact N1536 only; remove unnecessary sequence and output row checks')
    parser.add_argument('--padding-mode',choices=('all','none','q','k','output'),help='Optional padding-flag ablation')
    args=parser.parse_args()
    mode=args.padding_mode or ('none' if args.no_sequence_padding else 'all')
    source=REPO_ROOT/'runtime/ck_bias_stride_benchmark/probe.cpp'
    overlay=REPO_ROOT/'runtime/ck_schedule_reproduced'
    header=overlay/'ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp'
    if _sha256(source)!='c4f4b6e0f3322e6b4207922e4f2900042001d31a72996043a70a3d1ac6191d58':raise RuntimeError('Unverified probe source')
    if _sha256(header)!='0547ac78c5c239c7cf6677feec754d5089817f7fdaa69454108fb8d25cb1fa9c':raise RuntimeError('Unverified schedule overlay')
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    text=source.read_text()
    if mode!='all':
        for old,new in (
            ('ct::TileFmhaTraits<true,true,false,false,false,',
             'ct::TileFmhaTraits<'+('false' if mode in {'none','q'} else 'true')+','+('false' if mode in {'none','k'} else 'true')+',false,false,false,'),
            ('ct::Default2DEpilogueProblem<float,float,true,false>',
             'ct::Default2DEpilogueProblem<float,float,'+('false' if mode in {'none','output'} else 'true')+',false>'),
            ('if(!((batch==1000 && n==1000) || (batch==512 && n==1536))) return 2;',
             'if(batch!=512 || n!=1536 || n % BM || n % BN) return 2;'),
        ):
            if text.count(old)!=1:raise RuntimeError('Unexpected exact-shape source pattern')
            text=text.replace(old,new)
    probe=root/'probe.cpp';probe.write_text(text)
    report=dict(scope='CK key tile sweep after bias padding and batch grouping',source=str(probe),source_sha256=_sha256(probe),base_source_sha256=_sha256(source),no_sequence_padding=mode=='none',padding_mode=mode,overlay=str(overlay),overlay_header_sha256=_sha256(header),script_sha256=_sha256(pathlib.Path(__file__)),builds=[])
    tiles=((128,32,4),) if mode in {'q','k','output'} else ((128,64,4),(128,32,4)) if mode=='none' else ((128,64,4),(64,32,2),(128,32,4),(64,128,2),(128,128,4))
    for m,n,w in tiles:
        name=f'm{m}n{n}w{w}';library=root/(name+'.so')
        command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-fPIC','-shared','--offload-arch=gfx90a',f'-I{overlay}','-I/opt/rocm/include',f'-DBM={m}',f'-DBN={n}',f'-DWM={w}','-DBG=8',str(probe),'-o',str(library)]
        with (root/(name+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        item=dict(name=name,m=m,n=n,waves=w,group=8,command=command,returncode=result.returncode,library=str(library))
        if result.returncode==0:item['library_sha256']=_sha256(library)
        report['builds'].append(item);(root/'build.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    return 0 if all(x['returncode']==0 for x in report['builds']) else 1


if __name__=='__main__':raise SystemExit(main())
