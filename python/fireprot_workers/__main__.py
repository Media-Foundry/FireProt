"""Model process entry point. Rust owns requests, monitoring and acceptance."""
import json
import os
from pathlib import Path
import platform
import traceback
from .transport import Channel, announce_outputs


def main():
    request = json.loads(Path(os.environ['FIREPROT_REQUEST']).read_text())
    channel = Channel(request)
    try:
        if request['backend'] != 'native' and not (
                (request['workload']['model'] == 'protenix' and request['backend'] == 'fireprot') or
                (request['workload']['model'] == 'atlasfold' and request['backend'] in ('chunked', 'hip', 'hip_fused')) or
                (request['workload']['model'] == 'openfold3' and request['backend'] == 'hip_fused') or
                (request['workload']['model'] in ('esmfold2', 'plm') and request['backend'] == 'fireprot')):
            raise ValueError('Accelerated worker backends are not integrated yet')
        import torch
        prepare_cpu = None
        mode = os.environ.get('FIREPROT_ESMC_IMPORT_OVERLAP', '0')
        if mode not in ('0', '1'):
            raise ValueError('FIREPROT_ESMC_IMPORT_OVERLAP must be 0 or 1')
        if mode == '1':
            if request['workload']['model'] != 'plm' or request['workload']['model_name'] != 'esmc-fold2-6b':
                raise ValueError('Import overlap is qualified only for standalone ESMC')
            def prepare_cpu():
                import esm
                from .plm import ESM_COMMIT
                if request['source']['commit'] != ESM_COMMIT or Path(esm.__file__).resolve().parent.parent != Path(request['source']['root']).resolve():
                    raise ValueError('Biohub ESM imported source differs from its pin')
                # Import definitions only. Model construction and weights wait for start.
                from esm.models.esmc.model import EsmcModel
        channel.ready(torch, prepare_cpu=prepare_cpu)
        if request['workload']['model'] == 'plm':
            if request['workload'].get('batch') is not None:
                from .plm_batch import predict
            else:
                from .plm import predict
        elif request['workload']['model'] == 'openfold3':
            from .openfold3 import predict
        elif request['workload']['model'] == 'atlasfold':
            from .atlasfold import predict
        elif request['workload']['model'] == 'protenix':
            from .protenix import predict
        elif request['workload']['model'] == 'esmfold2':
            from .esmfold2 import predict
        else:
            raise ValueError('This model worker is not integrated yet')
        bundle = predict(request, channel)
        bundle['environment'] = dict(torch=str(torch.__version__), hip=str(torch.version.hip),
                                     python=platform.python_version())
        announce_outputs(channel, bundle)
    except BaseException as error:
        traceback.print_exc()
        channel.emit(dict(event='failed', message=f'{type(error).__name__}: {error}'))
        raise
    finally:
        channel.socket.close()


if __name__ == '__main__':
    main()
