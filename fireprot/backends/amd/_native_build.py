"""Torch compilation boundary for the Rust-owned native build cache.

This initial recipe is opt-in. Host dependency discovery is explicit and the
result records its coverage; it does not claim a hermetic compiler environment.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def load(package, sources):
    import torch
    from torch.utils import cpp_extension
    from fireprot.ops._bridge import bridge

    rocm = Path(cpp_extension.ROCM_HOME)
    torch_root = Path(torch.__file__).resolve().parent
    helper = Path(__file__).with_name('_compile_native.py')
    inputs = {}
    def add(path):
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f'missing build dependency: {path}')
        inputs[str(path.absolute())] = dict(name=str(path.absolute()), path=str(path.absolute()))

    for source in sources:
        add(source)
    for path in [__file__, helper, cpp_extension.__file__, sys.executable,
                 shutil.which(cpp_extension.get_cxx_compiler()), shutil.which('ninja'),
                 rocm/'bin/hipcc', rocm/'llvm/bin/clang++', rocm/'llvm/bin/ld.lld']:
        if path is None:
            raise RuntimeError('required compiler or Ninja is unavailable')
        add(path)
    # Include transitive Torch/ROCm/Python headers and device-library contents.
    roots = [torch_root/'include', rocm/'include', rocm/'llvm/lib/clang',
             Path(sysconfig.get_path('include'))]
    for root in roots:
        if not root.is_dir():
            raise RuntimeError(f'missing header tree: {root}')
        for path in sorted(root.rglob('*')):
            if path.is_file():
                add(path)
    for path in sorted((torch_root/'utils/hipify').glob('*.py')):
        add(path)
    for path in sorted((torch_root/'lib').glob('*.so*')):
        if path.is_file():
            add(path)
    flags = ['-O3', '--offload-arch=gfx90a']
    environment = {key: os.environ.get(key) for key in (
        'PATH', 'CXX', 'CC', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'CPATH',
        'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'ROCM_HOME',
        'HIP_PATH', 'HIP_CLANG_PATH', 'HIPCC_COMPILE_FLAGS_APPEND',
        'HIPCC_LINK_FLAGS_APPEND', 'PYTORCH_ROCM_ARCH')}
    environment['PYTORCH_ROCM_ARCH'] = 'gfx90a'
    spec = dict(schema_version=1, package=package, target='gfx90a',
        files=list(inputs.values()), arguments=dict(cxx=['-O3'], hip=flags),
        environment=environment, versions=dict(recipe='torch-hip-experimental-1',
            torch=str(torch.__version__), hip=str(torch.version.hip),
            python=sys.version, soabi=str(sysconfig.get_config_var('SOABI'))))
    cache = Path(os.environ.get('FIREPROT_EXTENSION_CACHE',
        str(Path(__file__).resolve().parents[3]/'runtime/torch_extensions')))/'rust-v1'

    def compile(directory, name):
        directory = Path(directory)
        copied = []
        for source in sources:
            target = directory / Path(source).name
            shutil.copyfile(source, target)
            copied.append(str(target))
        recipe = dict(name=name, sources=copied, build_directory=str(directory),
                      extra_cflags=['-O3'], extra_cuda_cflags=flags)
        recipe_path = directory/'recipe.json'
        recipe_path.write_text(json.dumps(recipe))
        env = dict(os.environ, PYTORCH_ROCM_ARCH='gfx90a')
        with (directory/'compiler.log').open('w') as log:
            subprocess.run([sys.executable, str(helper), str(recipe_path)], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        return [name + '.so']

    result = json.loads(bridge.cached_build(json.dumps(spec), cache, compile))
    name = result['receipt']['identity']['package'] + '_' + result['receipt']['identity']['sha256']
    library = Path(result['directory'])/(name+'.so')
    module_spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    result['dependency_scope'] = 'Torch/ROCm/Python headers, Torch libraries and listed compiler/helper files; system compiler runtime closure is not complete'
    return module, result
