"""Tensor-environment inspection and loading of Rust-accepted native artifacts."""
import importlib.util
import json
import sysconfig


def load(package, directory, manifest_sha256):
    import torch
    from fireprot.ops._bridge import bridge

    architecture = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(':')[0]
    expected = dict(package=package, target=architecture, torch=str(torch.__version__),
                    hip=str(torch.version.hip), python_soabi=str(sysconfig.get_config_var('SOABI')))
    accepted = json.loads(bridge.accept_kernel(directory, manifest_sha256, json.dumps(expected)))
    spec = importlib.util.spec_from_file_location(accepted['module_name'], accepted['library'])
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load the accepted kernel library')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, dict(mode='published', kernel=accepted)
