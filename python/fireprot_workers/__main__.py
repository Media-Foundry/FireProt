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
        if request['backend'] != 'native' or request['workload']['model'] != 'openfold3':
            raise ValueError('This worker currently implements only native OpenFold3; other integrations are pending')
        import torch
        channel.ready(torch)
        from .openfold3 import predict
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
