"""Small control messages and lossless tensor serialization for model workers."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket

class Channel:
    def __init__(self, request):
        self.request = request
        self.sequence = 0
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.connect(os.environ['FIREPROT_WORKER_SOCKET'])

    def emit(self, payload):
        envelope = dict(schema_version=1, run_id=self.request['run_id'],
                        sequence=self.sequence, payload=payload)
        self.socket.sendall((json.dumps(envelope, allow_nan=False) + '\n').encode())
        self.sequence += 1

    def ready(self, torch):
        torch.cuda.init()
        libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if '/libamdhip64.so' in line}
        if len(libraries) != 1:
            raise RuntimeError(f'Expected one loaded HIP library, found {libraries}')
        hip = ctypes.CDLL(next(iter(libraries)))
        hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
        pci = ctypes.create_string_buffer(64)
        if hip.hipDeviceGetPCIBusId(pci, 64, 0) != 0:
            raise RuntimeError('HIP PCI identity lookup failed')
        architecture = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
        self.emit(dict(event='ready', device=dict(worker_pid=os.getpid(), logical_device=0,
                       pci_address=pci.value.decode().lower(), architecture=architecture)))
        with self.socket.makefile('rb') as reader:
            start = reader.readline(4097)
        if len(start) > 4096 or json.loads(start) != {'schema_version': 1, 'command': 'start'}:
            raise RuntimeError('Missing or invalid supervisor start command')


def save_arrays(path, arrays):
    """Store raw BF16 bits as uint16 with explicit original dtype metadata."""
    import numpy as np
    import torch
    stored, descriptors = {}, {}
    for name, value in arrays.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            original = str(value.dtype).removeprefix('torch.')
            data = value.view(torch.uint16).numpy() if value.dtype == torch.bfloat16 else value.numpy()
        else:
            data = np.asarray(value)
            original = str(data.dtype)
        stored[name] = data
        descriptors[name] = dict(shape=list(data.shape), original_dtype=original,
                                 storage_dtype=str(data.dtype))
    np.savez(path, **stored)
    return descriptors


def announce_outputs(channel, bundle):
    output = Path(os.environ['FIREPROT_OUTPUT_DIR'])
    (output / 'predictions.json').write_text(json.dumps(bundle, indent=2, allow_nan=False) + '\n')
    paths = [output / 'predictions.json'] + [output / p['archive'] for p in bundle['predictions']]
    # Keep upstream structure files available and covered by the artifact hashes.
    paths += sorted((output / 'native').rglob('*.cif'))
    for diagnostic in ('operator-runtime.json', 'atlasfold-model.json'):
        if (output / diagnostic).is_file():
            paths.append(output / diagnostic)
    artifacts = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        artifacts.append(dict(path=str(path.relative_to(output)), sha256=digest.hexdigest()))
    channel.emit(dict(event='outputs', artifacts=artifacts))
