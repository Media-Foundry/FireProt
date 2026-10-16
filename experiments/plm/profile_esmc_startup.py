"""Instrument an isolated ESMC worker and retain import/startup timing evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def pin(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve()
work.mkdir(exist_ok=False)
original=root/'outputs/experiments/esmc-load-20261016/validation/qualified-loading'
deployment=work/'diagnostic-deployment';shutil.copytree(original,deployment)
transport=deployment/'python/fireprot_workers/transport.py'
s=transport.read_text().replace('import ctypes\n','import ctypes\nimport time\n_TIMING_MARKS = {"module_start": time.perf_counter(), "worker_pid": os.getpid()}\n')
# os is imported later in the original module; place the initial marker after imports.
s=s.replace('_TIMING_MARKS = {"module_start": time.perf_counter(), "worker_pid": os.getpid()}\n','')
s=s.replace('import socket\n','import socket\n_TIMING_MARKS = {"module_start": time.perf_counter(), "worker_pid": os.getpid()}\n')
s=s.replace('        torch.cuda.init()','        _TIMING_MARKS["cuda_init_start"] = time.perf_counter()\n        torch.cuda.init()\n        _TIMING_MARKS["cuda_init_end"] = time.perf_counter()')
s=s.replace('        self.emit(dict(event=\'ready\'', '        _TIMING_MARKS["ready_emit"] = time.perf_counter()\n        self.emit(dict(event=\'ready\'')
s=s.replace("            raise RuntimeError('Missing or invalid supervisor start command')", "            raise RuntimeError('Missing or invalid supervisor start command')\n        _TIMING_MARKS[\"start_received\"] = time.perf_counter()")
transport.write_text(s)
main=deployment/'python/fireprot_workers/__main__.py';s=main.read_text()
s=s.replace('import traceback\n','import traceback\nimport time\n')
s=s.replace('from .transport import Channel, announce_outputs','from .transport import Channel, announce_outputs, _TIMING_MARKS')
s=s.replace('        import torch\n','        _TIMING_MARKS["torch_import_start"] = time.perf_counter()\n        import torch\n        _TIMING_MARKS["torch_import_end"] = time.perf_counter()\n')
s=s.replace('        bundle = predict(request, channel)', '''        from . import plm_loading
        original_loader = plm_loading.streamed_esmc_state
        def timed_loader(*args, **kwargs):
            _TIMING_MARKS['placement_start'] = time.perf_counter()
            value = original_loader(*args, **kwargs)
            _TIMING_MARKS['placement_end'] = time.perf_counter()
            return value
        plm_loading.streamed_esmc_state = timed_loader
        _TIMING_MARKS['predict_start'] = time.perf_counter()
        bundle = predict(request, channel)
        _TIMING_MARKS['predict_end'] = time.perf_counter()''')
s=s.replace('        announce_outputs(channel, bundle)','        _TIMING_MARKS["announce_start"] = time.perf_counter()\n        announce_outputs(channel, bundle)\n        _TIMING_MARKS["announce_end"] = time.perf_counter()')
s=s.replace('        channel.socket.close()','        channel.socket.close()\n        _TIMING_MARKS["main_done"] = time.perf_counter()\n        Path(os.environ["FIREPROT_OUTPUT_DIR"]).joinpath("startup-timing.json").write_text(json.dumps(_TIMING_MARKS, indent=2)+"\\n")')
main.write_text(s)
for path in (transport,main):compile(path.read_text(),str(path),'exec')
request=json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
request.update(run_id='esmc-startup-diagnostic',output_dir=str(work/'prediction'),hip_device=6)
profile=json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-exact.json').read_text())
profile['workers']=str(deployment/'python')
profile['environment']['FIREPROT_BRIDGE_LIBRARY']=str(deployment/'target/release/lib_fireprot_bridge.so')
profile['environment']['PYTHONPROFILEIMPORTTIME']='1'
for name,data in (('request',request),('profile',profile)):
    (work/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
begin=time.perf_counter()
with (work/'invocation.log').open('x') as log:
    subprocess.run([str(deployment/'target/release/fireprot'),'esm',str(work/'request.json'),str(work/'profile.json')],
                   env=dict(os.environ,HIP_VISIBLE_DEVICES='6'),stdout=log,stderr=subprocess.STDOUT,check=True)
end=time.perf_counter()
output=work/'prediction';marks=json.loads((output/'startup-timing.json').read_text())
reference=root/'outputs/experiments/esmc-load-20261016/validation/native-pairs/baseline-1'
for index in range(4):
    with zipfile.ZipFile(reference/f'sample-{index}.npz') as a,zipfile.ZipFile(output/f'sample-{index}.npz') as b:
        assert a.namelist()==b.namelist() and all(a.read(k)==b.read(k) for k in a.namelist())
supervisor=json.loads((output/'supervisor.json').read_text());assert supervisor['phase']=='succeeded'
report=dict(schema_version=1,boundary='instrumented startup diagnostic; not a performance comparison',
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            invocation_seconds=end-begin,supervisor_seconds=supervisor['elapsed_seconds'],
            marks=marks,all_arrays_exact=True,
            derived_seconds={name:marks[b]-marks[a] for name,a,b in (
                ('torch_import','torch_import_start','torch_import_end'),('cuda_init','cuda_init_start','cuda_init_end'),
                ('ready_handshake','ready_emit','start_received'),('predict','predict_start','predict_end'),
                ('checkpoint_placement','placement_start','placement_end'),('announce','announce_start','announce_end'))},
            files={str(p.relative_to(work)):pin(p)['sha256'] for p in (main,transport,work/'request.json',work/'profile.json',output/'startup-timing.json',output/'stderr.log',output/'accepted.json')})
(work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
