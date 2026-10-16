"""Measure lazy exact FFN GEMM composition over the qualified pointwise route."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open('x') as f: json.dump(value, f, indent=2); f.write('\n')
    return pin(path)


def main():
    root = Path('/media/IntelSSD/fireprot')
    work = Path(sys.argv[1]).resolve()
    original = root/'outputs/experiments/esmc-pointwise-20261016/validation/qualified-pointwise'
    deployment = work/'qualified-resident-gemm'
    shutil.copytree(original, deployment)
    shutil.copy2(__file__, work/'prepare-measured.py')
    module = deployment/'python/fireprot/adapters/plm/gemm_lazy.py'
    assert not module.exists()
    shutil.copy2(work/'esmc_lt_lazy_scope.py', module)
    runtime = deployment/'python/fireprot/adapters/plm/runtime.py'
    text = runtime.read_text(); before = pin(runtime)
    needle = '        with trial_scope(model, request, batch_size) as (selected, pointwise):\n'
    assert text.count(needle) == 1
    text = text.replace(needle,
        '        from .gemm_lazy import fixed_gemm_scope\n'
        '        with fixed_gemm_scope(request, batch_size) as gemm, trial_scope(model, request, batch_size) as (selected, pointwise):\n')
    needle = '                if pointwise is not None: state["pointwise"] = pointwise\n'
    assert text.count(needle) == 1
    runtime.write_text(text.replace(needle, needle+'                if gemm is not None: state["fixed_gemm"] = gemm\n'))
    loader = deployment/'python/fireprot_workers/plm.py'
    text = loader.read_text(); loader_before = pin(loader)
    needle = "if k not in ('hip_rope', 'hip_batch', 'hip_esm', 'hip_esmc_pointwise')"
    assert text.count(needle) == 1
    loader.write_text(text.replace(needle, "if k not in ('hip_rope', 'hip_batch', 'hip_esm', 'hip_esmc_pointwise', 'hip_esmc_lt')"))
    write(work/'deployment-manifest.json', dict(original=str(original), runtime_before=before,
        loader_before=loader_before, files={str(p.relative_to(deployment)):pin(p)['sha256']
                                          for p in sorted(deployment.rglob('*')) if p.is_file()}))
    receipt = json.loads((root/'outputs/experiments/esmc-gemm-order-20261016/lt-build-receipt.json').read_text())
    library = Path(receipt['directory'])/next(iter(receipt['receipt']['artifacts']))
    library_pin = pin(library)
    assert library_pin['sha256'] == '7b554428ba5a79ed1ef2f107f8e4d8ba8e306352a8efa88c0b116f04f1fdedb4'
    resident = json.loads((root/'configs/requests/plm-esmc-fold2-6b-resident64-pointwise.json').read_text())
    cases = []
    for count in (4,64):
        case = work/f'b{count}-pairs'; case.mkdir(exist_ok=False)
        request = (json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
                   if count == 4 else json.loads(json.dumps(resident)))
        request['resources']['hip_esmc_pointwise'] = resident['resources']['hip_esmc_pointwise']
        request['resources']['hip_esmc_lt'] = library_pin
        profiles = {}
        for arm, flag in (('baseline','0'), ('candidate','1')):
            profile = json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-pointwise.json').read_text())
            profile['workers'] = str(deployment/'python')
            profile['environment']['FIREPROT_BRIDGE_LIBRARY'] = str(deployment/'target/release/lib_fireprot_bridge.so')
            profile['environment']['FIREPROT_ESMC_FIXED_GEMM'] = flag
            assert profile['environment']['FIREPROT_ESMC_POINTWISE'] == '1'
            profiles[arm] = write(case/f'{arm}-profile.json', profile)
        pairs = []
        for index in range(3):
            pair = dict(first='candidate' if index == 1 else 'baseline')
            for arm in ('baseline','candidate'):
                value = dict(request, run_id=f'esmc-resident-gemm-{count}-{arm}-{index+1}', hip_device=6,
                             output_dir=str(case/f'{arm}-{index+1}'))
                pair[arm] = dict(request=write(case/f'{arm}-{index+1}.json', value), profile=profiles[arm])
            pairs.append(pair)
        spec = dict(schema_version=1,
            execution=dict(schema_version=1, timeout_seconds=3600, invocation_timeout_seconds=600, pairs=pairs),
            comparison=dict(scope='end_to_end',
                cache_state='Fresh workers; resident file/GPU caches not flushed; HIP libraries built offline; no output cache.',
                treatment_description=f'Lazy exact hipBLASLt FFN algorithm4139 over qualified residual4/SwiGLU8/pinned-loading/import-overlap/hash8/local-CPU route. N1000 x {count}, B4, native BF16, seed101, HIP6. Full original asset hashes and all exports retained.',
                expected_changes={
                    'launch.environment.FIREPROT_ESMC_FIXED_GEMM':dict(baseline='0',candidate='1'),
                    'invocation.profile_sha256':dict(baseline=profiles['baseline']['sha256'],candidate=profiles['candidate']['sha256'])}))
        write(case/'spec.json', spec); cases.append(case)
    env = dict(os.environ, HIP_VISIBLE_DEVICES='6')
    for name in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'): env.pop(name, None)
    for case in cases:
        with (case/'pairs.log').open('x') as log:
            subprocess.run([str(deployment/'target/release/fireprot'),'benchmark-pairs',str(case/'spec.json'),str(case/'measurements')],
                           cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert pin(library) == library_pin
        print(case.name, 'complete', flush=True)


if __name__ == '__main__': main()
