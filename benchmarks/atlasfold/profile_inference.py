"""Profile the native AtlasFold monomer including every AtlasLM recycle pass.

Run in the pinned upstream ROCm environment. This is a diagnostic benchmark;
its event instrumentation is included in wall timings. Checkpoint hashing is
performed before the request timer and warms filesystem caches.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from benchmarks.protenix.profile_inference import _GpuProcessMonitor

ASSET_ROOT = Path('/media/PM982/engramfold/folding_e2e_20260921')
SOURCE_COMMIT = '8ab3aca0e18c8b814d5ca6756b2617a07d72c68d'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_physical_device() -> dict:
    """Match the HIP device to SMI by PCI address, never by ordinal equality."""
    libs = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
            if '/libamdhip64.so' in line}
    if len(libs) != 1:
        raise RuntimeError(f'Cannot resolve loaded HIP library: {sorted(libs)}')
    hip = ctypes.CDLL(next(iter(libs)))
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(64)
    if hip.hipDeviceGetPCIBusId(buf, 64, 0) != 0:
        raise RuntimeError('HIP PCI lookup failed')
    bdf = buf.value.decode().lower()
    devices = json.loads(subprocess.check_output(['amd-smi', 'list', '--json']))
    matches = [d for d in devices if d['bdf'].lower() == bdf]
    if len(matches) != 1:
        raise RuntimeError(f'No unique SMI match for {bdf}')
    device = matches[0]
    if device['gpu'] not in range(8):
        raise RuntimeError('Device is outside the authorized 0–7 range')
    records = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
    selected = next(d for d in records if d['gpu'] == device['gpu'])
    other = [int(p['process_info']['pid']) for p in selected['process_list']
             if isinstance(p['process_info'], dict)
             and int(p['process_info']['pid']) != os.getpid()]
    if other:
        raise RuntimeError(f'Selected GPU already has external processes: {other}')
    return device


def read_sequence(path: Path) -> tuple[str, str]:
    """Accept one FASTA record or one existing Protenix monomer case."""
    if path.suffix == '.json':
        records = json.loads(path.read_text())
        if len(records) != 1 or len(records[0]['sequences']) != 1:
            raise ValueError('Exactly one monomer sequence is required')
        chain = records[0]['sequences'][0]['proteinChain']
        if chain.get('count', 1) != 1:
            raise ValueError('Only one chain is supported')
        return records[0]['name'], chain['sequence']
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines or not lines[0].startswith('>') or sum(x.startswith('>') for x in lines) != 1:
        raise ValueError('Exactly one FASTA record is required')
    return lines[0][1:], ''.join(lines[1:])


class StageEvents:
    """Inclusive stage events; overlapping parent/child totals must not be summed."""
    def __init__(self, torch):
        self.torch = torch
        self.events = []
        self.originals = []

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        @functools.wraps(original)
        def call(*args, **kwargs):
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.events.append((label, start, end))
            return result
        self.originals.append((obj, name, original))
        setattr(obj, name, call)

    def summarize(self):
        result = {}
        for label, start, end in self.events:
            value = result.setdefault(label, {'calls': 0, 'gpu_ms': 0.0})
            value['calls'] += 1
            value['gpu_ms'] += start.elapsed_time(end)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=ASSET_ROOT/'weights/atlasfold/snapshot/weights/atlasfold-260703.pth')
    parser.add_argument('--lm-path', type=Path, default=ASSET_ROOT/'weights/atlaslm3b/snapshot/weights/atlaslm_3b_base.pth')
    parser.add_argument('--kernel', choices=('torch', 'triton'), default='torch')
    parser.add_argument('--backend', choices=('native', 'chunked', 'hip'), default='native')
    parser.add_argument('--allow-long1536', action='store_true')
    parser.add_argument('--stage-events', choices=('all', 'stages', 'off'), default='all')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--recycles', type=int, default=4)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.recycles < 0 or args.samples < 1 or args.repeats < 1:
        parser.error('Invalid recycle, sample, or repetition count')
    if args.backend != 'native' and args.kernel != 'torch':
        parser.error('FireProt backends require the torch upstream kernel route')
    if not os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('ROCR_VISIBLE_DEVICES'):
        parser.error('Set HIP_VISIBLE_DEVICES to one ordinal and unset ROCR_VISIBLE_DEVICES')
    if not os.environ['HIP_VISIBLE_DEVICES'].isdigit() or int(os.environ['HIP_VISIBLE_DEVICES']) not in range(8):
        parser.error('HIP_VISIBLE_DEVICES must be one ordinal in 0–7')
    name, sequence = read_sequence(args.input)
    if not sequence or any(x not in 'ACDEFGHIKLMNPQRSTVWYX' for x in sequence):
        parser.error('Use an uppercase single protein sequence with standard residues or X')
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(), 'length': len(sequence),
              'cache': 'Weights hashed before timer; filesystem warm; model fresh; no LM feature cache',
              'dtype': 'Native mixed BF16/FP32; BF16 autocast; FP32 diffusion',
              'mlm_prob': 0.15, 'trunk_passes': args.recycles + 1, 'records': []}
    monitor = None
    runtime = None
    def save():
        (args.out/'manifest.json').write_text(json.dumps(report, indent=2))
    save()
    try:
        import torch
        import numpy as np
        import atlasfold
        from atlasfold.pretrained import load_model, get_runner
        from atlasfold.runner import get_sampling_config
        source = Path(atlasfold.__file__).resolve().parents[2]
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
        if commit != SOURCE_COMMIT or dirty:
            raise RuntimeError('AtlasFold source must match the clean pinned checkout')
        report['source_commit'] = commit
        report['weights'] = {key: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                             for key, path in [('atlasfold', args.model_path), ('atlaslm', args.lm_path)]}
        torch.set_num_threads(8)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.init()
        report['device'] = resolve_physical_device()
        report['environment'] = {'torch': torch.__version__, 'hip': torch.version.hip,
                                 'gpu': torch.cuda.get_device_properties(0).name,
                                 'HIP_VISIBLE_DEVICES': os.environ['HIP_VISIBLE_DEVICES']}
        monitor = _GpuProcessMonitor(report['device']['gpu'], os.getpid(), 1)
        monitor.start()
        request_start = time.perf_counter()
        model = load_model('atlasfold', device='cuda', kernel=args.kernel,
                           model_path=args.model_path, lm_path=args.lm_path).eval()
        torch.cuda.synchronize()
        report['load_seconds'] = time.perf_counter() - request_start
        if args.backend != 'native':
            from fireprot.adapters.atlasfold.runtime import AtlasFoldRuntime
            runtime = AtlasFoldRuntime(model, len(sequence), args.backend,
                                       allow_long1536=args.allow_long1536)
        report['config'] = dataclasses.asdict(model.cfg)
        runner = get_runner(model)
        bucket = runner._get_length_bucket(len(sequence))
        report['effective_bucket_length'] = bucket
        report['padding_residues'] = bucket - len(sequence)
        report['sampling'] = dataclasses.asdict(get_sampling_config(bucket))
        stages = StageEvents(torch)
        if args.stage_events != 'off':
            for method in ('inference', 'run_trunk', 'run_lm_embedder'):
                stages.wrap(model, method, method)
            for attr in ('lm_stack', 'main_stack', 'confidence_head'):
                stages.wrap(getattr(model, attr), 'forward', attr)
            stages.wrap(model.diffusion_head, 'sample', 'diffusion')
        if args.stage_events == 'all':
            for block in model.lm.transformer.blocks:
                stages.wrap(block, 'forward', 'atlaslm_block')
            for module in model.modules():
                if type(module).__name__.startswith('Triangle'):
                    stages.wrap(module, 'forward', type(module).__name__)
        save()
        for repeat in range(args.repeats):
            stages.events.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode():
                result = runner.fold(name, sequence, num_samples=args.samples, seeds=args.seed,
                                     num_recycles=args.recycles, mlm_prob=0.15)
            torch.cuda.synchronize()
            record = {'repeat': repeat, 'phase': 'first' if repeat == 0 else 'warm',
                      'prediction_wall_seconds': time.perf_counter() - begin,
                      'stages': stages.summarize(),
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'outputs': []}
            for key, sample in result.outputs.items():
                stem = f'repeat{repeat}-seed{key[0]}-sample{key[1]}'
                arrays = dict(coordinates=sample.coordinates, plddt=sample.plddt, pae=sample.pae, ptm=sample.ptm)
                if not all(np.isfinite(x).all() for x in arrays.values()):
                    raise RuntimeError('Nonfinite native output')
                np.savez(args.out/f'{stem}.npz', **arrays)
                (args.out/f'{stem}.cif').write_text(sample.to_mmcif())
                record['outputs'].append(stem)
            if repeat == 0:
                report['first_request_seconds'] = time.perf_counter() - request_start
            report['records'].append(record)
            save()
        report['status'] = 'complete'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['fireprot'] = runtime.manifest() if runtime else None
        report['isolation'] = monitor.stop() if monitor else {'passed': False}
        save()
    if not report['isolation']['passed']:
        raise RuntimeError('GPU isolation check failed; reject performance claim')


if __name__ == '__main__':
    main()
