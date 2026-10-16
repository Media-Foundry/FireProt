"""Inspect fresh-process fixed-plan startup and verify all 80 real projections."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from fireprot.adapters.plm.runtime import optimized_scope
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from esmc_lt_scope import FixedProjection


class PlanRejected(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(f'Plan creation rejected: {code}')


class Projection(FixedProjection):
    def __init__(self, library, serialized):
        self.batch_size = 4
        self.plan = ctypes.c_void_p()
        begin = time.perf_counter()
        self.lib = ctypes.CDLL(str(library))
        self.lib.esmc_lt_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
        self.lib.esmc_lt_create.restype = ctypes.c_int
        self.lib.esmc_lt_create_serialized.argtypes = self.lib.esmc_lt_create.argtypes + [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.esmc_lt_create_serialized.restype = ctypes.c_int
        self.lib.esmc_lt_run.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_size_t, ctypes.c_void_p]
        self.lib.esmc_lt_run.restype = ctypes.c_int
        self.lib.esmc_lt_destroy.argtypes = [ctypes.c_void_p]
        self.lib.esmc_lt_destroy.restype = None
        self.lib.esmc_lt_algo_size.restype = ctypes.c_size_t
        self.lib.esmc_lt_export.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)]
        self.lib.esmc_lt_export.restype = ctypes.c_int
        loaded = time.perf_counter()
        size = ctypes.c_size_t()
        if serialized is None:
            rc = self.lib.esmc_lt_create(4008, ctypes.byref(self.plan), ctypes.byref(size))
        else:
            blob = ctypes.create_string_buffer(serialized)
            rc = self.lib.esmc_lt_create_serialized(4008, ctypes.byref(self.plan), ctypes.byref(size), blob, len(serialized))
        if rc:
            raise PlanRejected(rc)
        created = time.perf_counter()
        try:
            self.workspace = torch.empty(size.value, device='cuda:0', dtype=torch.uint8)
            self.host_seconds = dict(load_library=loaded-begin, create_plan=created-loaded,
                                     workspace=time.perf_counter()-created)
            blob = ctypes.create_string_buffer(self.lib.esmc_lt_algo_size())
            timing = (ctypes.c_double*5)()
            assert self.lib.esmc_lt_export(self.plan, blob, len(blob), timing) == 0
            self.serialized = blob.raw
            self.stages = dict(zip(('create_handle', 'version_and_descriptors', 'lookup_or_restore',
                                   'support_check', 'finish'), timing))
        except BaseException:
            self.close()
            raise


assert os.environ['HIP_VISIBLE_DEVICES'] == '6'
request = json.loads(Path(sys.argv[1]).read_text())
library = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
output.mkdir(exist_ok=False)
source = Path(sys.argv[4]) if len(sys.argv) > 4 else None
serialized = None
if source is not None:
    parent_report = json.loads((source.parent/'report.json').read_text())
    serialized = source.read_bytes()
    assert hashlib.sha256(serialized).hexdigest() == parent_report['algorithm_sha256']
    assert hashlib.sha256(library.read_bytes()).hexdigest() == parent_report['library_sha256']
torch.set_num_threads(8)
torch.manual_seed(request['seed'])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.cuda.init()
assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx90a'
assert not torch.cuda.tunable.is_enabled()
inp = Path(request['input']['path'])
assert hashlib.sha256(inp.read_bytes()).hexdigest() == request['input']['sha256']
sequences = [x['sequence'] for x in json.loads(inp.read_text())['sequences']]
assert len(sequences) == 4 and all(len(s) == 1000 for s in sequences)
with torch.inference_mode():
    begin = time.perf_counter()
    try:
        projection = Projection(library, serialized)
    except PlanRejected as error:
        report = dict(schema_version=1, mode='restore' if serialized else 'index', status='rejected',
            boundary='Fresh process initialization; support checks retained; model not executed.',
            return_code=error.code, host_seconds=time.perf_counter()-begin,
            hip_device=6, batch=4, library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            algorithm_sha256=hashlib.sha256(serialized).hexdigest() if serialized else None,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit(0)
    try:
        model, _, _ = load_model(request, sequences[0])
        forward = prepare_forward(model, request['workload']['model_name'], sequences)
        expected = {k: v.clone() for k, v in forward().items()}
        with optimized_scope(model, request, 1000, batch_size=4), projection.scope(check_each=True) as state:
            actual = forward()
        exact = {k: torch.equal(v.view(torch.uint8), actual[k].view(torch.uint8)) for k, v in expected.items()}
        assert all(exact.values()) and state['calls'] == 80
        if serialized is not None:
            assert serialized == projection.serialized
        (output/'algorithm.bin').write_bytes(projection.serialized)
        warm = Projection(library, projection.serialized)
        try:
            with optimized_scope(model, request, 1000, batch_size=4), warm.scope(check_each=True) as warm_state:
                warm_actual = forward()
            assert warm_state['calls'] == 80
            assert all(torch.equal(v.view(torch.uint8), warm_actual[k].view(torch.uint8)) for k, v in expected.items())
            assert warm.serialized == projection.serialized
            warm_report = dict(host_seconds=warm.host_seconds, create_stages_seconds=warm.stages,
                               all_80_projections_exact=True, output_exact=True)
        finally:
            torch.cuda.synchronize()
            warm.close()
        report = dict(schema_version=1, status='accepted', mode='restore' if serialized else 'index',
            boundary='Fresh process plan initialization after HIP init, before model loading; not E2E.',
            torch=str(torch.__version__), hip=str(torch.version.hip), hip_device=6, batch=4,
            length=1000, seed=request['seed'], precision='native BF16',
            input_sha256=request['input']['sha256'], library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            algorithm_sha256=hashlib.sha256(projection.serialized).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            algorithm_bytes=len(projection.serialized), host_seconds=projection.host_seconds,
            create_stages_seconds=projection.stages, all_80_projections_exact=True, output_exact=exact,
            warm_restore=warm_report)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        torch.cuda.synchronize()
        projection.close()
