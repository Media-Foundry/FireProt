"""Build isolated gfx90a CK attention scheduling probes; never edit ROCm headers."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

from benchmarks.protenix.profile_inference import REPO_ROOT, _sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--rocm-root', type=pathlib.Path, default=pathlib.Path('/opt/rocm'))
    parser.add_argument('--tile', type=int, choices=(64, 128), default=128)
    parser.add_argument('--contiguous-qkv', action='store_true')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    relative = pathlib.Path('ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp')
    original = args.rocm_root / 'include' / relative
    expected = 'af3ae7102bb6c6439ce8501fec214571076eaec9540c1f02ac1a9d16bb0e1ef1'
    if _sha256(original) != expected:
        raise RuntimeError('This probe requires the measured CK header hash')
    header = original.read_text()
    old = ('const index_t i_block = blockIdx.y; // blockIdx.x\n'
           '            const index_t i_nhead = blockIdx.x; // blockIdx.y\n'
           '            const index_t i_batch = blockIdx.z;')
    new = ('const index_t i_block = blockIdx.y;\n'
           '            const index_t i_nhead = blockIdx.z % 4;\n'
           '            const index_t i_batch = blockIdx.x + gridDim.x * (blockIdx.z / 4);')
    if header.count(old) != 1:
        raise RuntimeError('Unexpected CK tile-index implementation')
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(header.replace(old, new))
    source_path = REPO_ROOT / 'kernels/amd/fireprot_ck_attention.cpp'
    source = source_path.read_text()
    replacements = {
        'Kernel::GridSize(batch,4,n,32,false)':
            'dim3(batch==1000?8:BG,(n+BM-1)/BM,4*(batch/(batch==1000?8:BG)))',
    }
    if args.contiguous_qkv:
        replacements.update({
            '128,128,128,bias_stride,0,128': '32,32,32,bias_stride,0,128',
            '32,32,32,n*bias_stride,0,0,32': 'n*32,n*32,n*32,n*bias_stride,0,0,32',
        })
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f'Unexpected source pattern: {old}')
        source = source.replace(old, new)
    probe = root / 'probe.cpp'
    probe.write_text(source)
    report = dict(original_header_sha256=expected, patched_header_sha256=_sha256(target),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  base_source_sha256=_sha256(source_path),
                  qkv_layout='BHND contiguous' if args.contiguous_qkv else 'BSHD', builds=[])
    for group in (8, 32, 512):
        name = f'batch_group{group}'
        command = [str(args.rocm_root / 'bin/hipcc'), '-O3', '-std=c++17', '-fPIC', '-shared',
                   '--offload-arch=gfx90a', f'-I{root}', f'-I{args.rocm_root / "include"}',
                   f'-DBM={args.tile}', f'-DWM={2 if args.tile == 64 else 4}', f'-DBG={group}',
                   str(probe), '-o', str(root / f'{name}.so')]
        with (root / f'{name}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report['builds'].append(dict(name=name, command=command, returncode=result.returncode))
        (root / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'Compilation failed: {root / (name + ".log")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
