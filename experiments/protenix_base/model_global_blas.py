"""Isolated full-forward BLAS preference experiment; scientific settings unchanged."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import torch


@contextmanager
def install(model, output):
    requested=os.environ.get('FIREPROT_BASE_GLOBAL_BLAS','hipblas')
    if requested not in ('hipblas','hipblaslt'):raise ValueError('Unsupported experimental BLAS preference')
    if (model.training or not torch.version.hip or not torch.are_deterministic_algorithms_enabled()
            or torch.backends.cuda.matmul.allow_tf32):
        raise RuntimeError('Strict FP32 HIP inference required')
    previous=torch.backends.cuda.preferred_blas_library()
    records=dict(requested=requested,previous=str(previous),error=None,
                 scope='Full forward Torch BLAS preference; individual calls may internally fall back')
    try:
        torch.backends.cuda.preferred_blas_library(requested)
        records['effective']=str(torch.backends.cuda.preferred_blas_library())
        yield
    except BaseException as error:
        records['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        torch.backends.cuda.preferred_blas_library(previous)
        records['restored']=str(torch.backends.cuda.preferred_blas_library())
        Path(output,'base-global-blas.json').write_text(json.dumps(records,indent=2)+'\n')
