"""Build distinct native/fake/holder executables through Rust's native service."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path('/media/IntelSSD/fireprot')
    work = Path(sys.argv[1]).resolve()
    canonical = root/'kernels/amd/tools/fireprot_smi_probe.cpp'
    staged = work/'esmc_smi_probe.cpp'
    if not staged.exists(): shutil.copy2(canonical, staged)
    if staged.read_bytes() != canonical.read_bytes():
        raise ValueError('Staged probe differs from the maintained source')
    base = json.loads((root/'outputs/experiments/esm-speed-20261016/build-spec.json').read_text())
    artifacts = {}
    for kind in ('native','fake','hold'):
        spec = json.loads(json.dumps(base))
        build = spec['build']; build['package'] = f'esmc_smi_{kind}'
        build['target'] = 'gfx90a' if kind=='hold' else 'x86_64'
        build['versions']['recipe'] = f'esmc-smi-{kind}-v1'
        source = work/('esmc_smi_hold.hip' if kind=='hold' else 'esmc_smi_probe.cpp')
        for item in build['files']:
            if item['name']=='source': item['path']=str(source)
            if item['name']=='compiler' and kind!='hold': item['path']='/opt/rocm/llvm/bin/clang++'
        build['arguments']['compile'] = ['-O2','-std=c++17','-Wall','-Wextra','-Werror',
                                         '--ld-path=/opt/rocm/llvm/bin/ld.lld']
        if kind=='hold':
            build['arguments']['compile'].append('--offload-arch=gfx90a')
            build['arguments']['link'] = ['-Wl,-rpath,/opt/rocm/lib']
        if kind=='native':
            build['files'].append(dict(name='amd-smi-library',path='/opt/rocm/lib/libamd_smi.so.27.0.0'))
            build['arguments']['link'] = ['-L/opt/rocm/lib','-lamd_smi','-Wl,-rpath,/opt/rocm/lib']
        if kind=='fake':
            build['files'].append(dict(name='fake',path=str(work/'esmc_smi_fake.cpp')))
            spec['sources'].append('fake')
        spec_path=work/f'{kind}-build-spec.json'; receipt_path=work/f'{kind}-build-receipt.json'
        with spec_path.open('x') as f: json.dump(spec,f,indent=2);f.write('\n')
        with receipt_path.open('x') as f:
            subprocess.run([str(root/'target/release/fireprot'),'build-hip',str(spec_path),str(work/'build-cache')],
                           cwd=root,stdout=f,check=True)
        receipt=json.loads(receipt_path.read_text())
        assert len(receipt['receipt']['artifacts'])==1
        name, sha=next(iter(receipt['receipt']['artifacts'].items()))
        artifacts[kind]=dict(path=str(Path(receipt['directory'])/name),sha256=sha)
        print(kind,'built',flush=True)
    with (work/'binaries.json').open('x') as f: json.dump(artifacts,f,indent=2);f.write('\n')


if __name__=='__main__':main()
