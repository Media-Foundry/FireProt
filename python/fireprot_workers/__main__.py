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
                (request['workload']['model'] == 'openfold3' and request['backend'] == 'hip_fused')):
            raise ValueError('Accelerated worker backends are not integrated yet')
        import torch
        channel.ready(torch)
        if request['workload']['model'] == 'openfold3':
            from .openfold3 import predict
        elif request['workload']['model'] == 'atlasfold':
            from .atlasfold import predict
        elif request['workload']['model'] == 'protenix':
            from .protenix import predict
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
