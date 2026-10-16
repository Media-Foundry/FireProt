"""Seal independent CPU-placement, hash-parallelism and final native comparisons."""
import hashlib
import json
from pathlib import Path
import sys


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


work = Path(sys.argv[1]).resolve()
pilot = work.parent/'esmc-affinity-20261016'
reports = {'python_affinity_pilot': json.loads((pilot/'collected.json').read_text())}
files = [work/'prepare_esmc_hash8_pairs.py', work/'collect_esmc_host_pairs.py', Path(__file__),
         work/'build-source.json', work/'binaries.json',
         work/'final-build/build-source-final.json', work/'final-build/binaries-final.json']
for name in ('hash', 'affinity-final', 'native-final'):
    case = work/name
    report = json.loads((case/'collected.json').read_text())
    assert report['all_arrays_exact'] and len(report['records']) == 6
    reports[name] = report
    manifest = json.loads((case/'deployment-manifest.json').read_text())
    for relative, digest in manifest['files'].items():
        assert pin(case/'qualified-hash8'/relative)['sha256'] == digest
    files += [case/p for p in ('collected.json', 'deployment-manifest.json', 'prepare-measured.py',
                              'spec.json', 'candidate-profile.json', 'binaries.json')]
pilot_manifest = json.loads((pilot/'deployment-manifest.json').read_text())
for relative, digest in pilot_manifest['files'].items():
    assert pin(pilot/'qualified-affinity'/relative)['sha256'] == digest
pilot_files = [pilot/p for p in ('collected.json', 'deployment-manifest.json', 'prepare-measured.py',
                               'esmc_cpu_affinity.py', 'collect_esmc_affinity_pairs.py')]
result = dict(schema_version=1, date='2026-10-16', baseline_commit='e6abebb',
    experiment=str(work), reports=reports,
    files={str(p.relative_to(work)): pin(p)['sha256'] for p in files},
    pilot_files={str(p): pin(p)['sha256'] for p in pilot_files},
    final_deployment=str(work/'native-final/qualified-hash8'),
    final_profile=pin(work/'native-final/candidate-profile.json'),
    build_source=json.loads((work/'final-build/build-source-final.json').read_text()),
    conditions=dict(hip_device=6, pci='0000:8e:00.0', length=1000, batch=4,
                    seed=101, precision='native BF16', cache='fresh workers; unflushed caches; no transformed checkpoints'),
    limitations=['Separate AB/BA/AB comparisons; do not combine medians or multiply ratios.',
                 'All original asset hashes and outputs are retained; no pure GPU-kernel speedup claim.',
                 'The early hash comparison uses the first Rust build; final affinity/native use the final failure-archival fix.',
                 'CPU affinity is verified for the main worker at readiness, not an immutable mask for every runtime thread.',
                 'No singleton, resident64, folding or other-model speedup inferred.'])
with (work/'evidence.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps({key: value['performance']['median_speedup'] for key, value in reports.items()}, indent=2))
