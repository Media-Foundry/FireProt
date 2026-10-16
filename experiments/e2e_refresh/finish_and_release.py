"""Verify the completed refresh before releasing the detached asset transfer."""
import json
from pathlib import Path
import subprocess
import sys
import time


root = Path(sys.argv[1]).resolve()
expected = (
    'esmc-b4', 'esm2-b4', 'esm3-b4', 'atlaslm-singleton',
    'atlaslm-b4-refresh', 'mini-n1000', 'mini-n1536', 'mini-esm-n1000',
    'tiny-n1000', 'atlasfold-n1000', 'atlasfold-n1536-declaration-retry',
    'openfold3-n1536-refreshed-env', 'esmfold2-n1000',
)
status_path = root / 'finish-and-release-status.json'
try:
    for name in ('base-n1000', 'atlaslm-b4-refresh',
                 'atlasfold-n1536-declaration-retry'):
        path = root / name / 'driver-status.json'
        while not path.exists():
            time.sleep(10)
        assert json.loads(path.read_text())['exit_code'] == 0, name
    scripts = root / 'measured-scripts'
    for script in ('collect-v3.py', 'collect_base.py'):
        with (root / (script + '.final.log')).open('x') as log:
            subprocess.run([sys.executable, str(scripts / script), str(root)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((root / 'collected-latest.json').read_text())
    assert all(report['cases'][name]['status'] == 'verified' for name in expected)
    base = json.loads((root / 'base-n1000/collected.json').read_text())
    assert base['status'] == 'user_policy_passed'
    assert base['generic_condition_check_completed']
    decision = dict(all_gpu_work_complete=True, ssh_target='husrcf@Precision',
                    evidence_collected=True, verified_cases=list(expected),
                    base_policy=base['policy'])
    release = root / 'release-asset-transfer.json'
    assert not release.exists()
    temporary = root / 'release-asset-transfer.tmp'
    temporary.write_text(json.dumps(decision, indent=2) + '\n')
    temporary.rename(release)
    status_path.write_text(json.dumps(dict(status='verified_and_transfer_released'))+'\n')
    print('All refresh evidence verified; detached asset transfer released', flush=True)
except BaseException as error:
    status_path.write_text(json.dumps(dict(status='failed', error=repr(error)))+'\n')
    raise
