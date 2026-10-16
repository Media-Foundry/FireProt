"""Experimental frozen ESMC GEMM selection; offline tuning is never run here."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

_TABLE_SHA256 = '9980c71947c81b3c759cdff4707376031126b0fa35045b7f0b3ece6345e3c060'
_loaded = None


@contextmanager
def tuned_gemm_scope(request, batch_size):
    global _loaded
    flag = os.environ.get('FIREPROT_ESMC_TUNED_GEMM', '0')
    if flag == '0':
        yield None
        return
    if flag != '1':
        raise ValueError('FIREPROT_ESMC_TUNED_GEMM must be 0 or 1')
    if (request['backend'] != 'fireprot' or request['workload']['model_name'] != 'esmc-fold2-6b'
            or request['workload']['precision'] != 'native' or batch_size not in (1,4)):
        raise ValueError('Frozen ESMC GEMMs require native precision and N1000 B1/B4')
    resource = request['resources']['esmc_gemm_algorithms']
    path = Path(resource['path']).resolve()
    if resource['sha256'] != _TABLE_SHA256 or hashlib.sha256(path.read_bytes()).hexdigest() != _TABLE_SHA256:
        raise ValueError('Unqualified ESMC GEMM algorithm table')
    import torch.cuda.tunable as tunable
    if tunable.is_enabled():
        raise RuntimeError('ESMC GEMM scope requires exclusive ownership of TunableOp')
    old_tuning = tunable.tuning_is_enabled()
    key = (str(path), _TABLE_SHA256)
    try:
        tunable.tuning_enable(False)
        if _loaded is None:
            # Torch has process-exit CSV export but no Python API to disable it.
            # Keep the pinned input read-only; discard any automatic export.
            tunable.set_filename(os.devnull)
            if not tunable.read_file(str(path)):
                raise RuntimeError('ESMC GEMM table rejected by runtime version validators')
            if len(tunable.get_results()) != 8:
                raise RuntimeError('Unexpected pre-existing or missing GEMM tuning records')
            _loaded = key
        elif _loaded != key:
            raise RuntimeError('Cannot switch frozen ESMC GEMM resources in one worker')
        tunable.enable(True)
        yield dict(table=resource, online_tuning=False, entries=8)
    finally:
        tunable.enable(False)
        tunable.tuning_enable(old_tuning)
