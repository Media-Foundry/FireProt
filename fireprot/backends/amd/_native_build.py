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
    paths = [*sources, __file__, helper, cpp_extension.__file__, sys.executable,
             shutil.which(cpp_extension.get_cxx_compiler()), shutil.which('ninja'),
             rocm/'bin/hipcc', rocm/'llvm/bin/clang++', rocm/'llvm/bin/ld.lld',
             rocm/'lib/libamdhip64.so']
    if any(path is None for path in paths):
        raise RuntimeError('required compiler or Ninja is unavailable')
    # Python describes environment-owned roots; Rust traverses and rechecks them.
    roots = [torch_root/'include', rocm/'include', rocm/'llvm/lib/clang',
             Path(sysconfig.get_path('include')), torch_root/'lib',
             Path('/usr/include'), Path('/usr/local/include'), Path('/usr/lib/gcc')]
    trees = [dict(name=f'dependency-{index}', path=str(root.absolute()), suffixes=[])
             for index, root in enumerate(roots)]
    trees.append(dict(name='hipify', path=str(torch_root/'utils/hipify'), suffixes=['.py']))
    flags = ['-O3', '--offload-arch=gfx90a']
    environment = {key: os.environ.get(key) for key in (
        'PATH', 'CXX', 'CC', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'CPATH',
        'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'ROCM_HOME',
        'HIP_PATH', 'HIP_CLANG_PATH', 'HIPCC_COMPILE_FLAGS_APPEND',
        'HIPCC_LINK_FLAGS_APPEND', 'PYTORCH_ROCM_ARCH')}
    environment['PYTORCH_ROCM_ARCH'] = 'gfx90a'
    spec = dict(schema_version=1, package=package, target='gfx90a',
        files=[dict(name=str(Path(path).absolute()), path=str(Path(path).absolute()))
               for path in paths], trees=trees, arguments=dict(cxx=['-O3'], hip=flags),
        environment=environment, versions=dict(recipe='torch-hip-experimental-2',
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
    result['dependency_scope'] = 'Rust-enumerated Torch/ROCm/Python/system headers, GCC trees, Torch libraries and listed compiler/helper files; complete dynamic compiler/runtime closure remains unverified'
    return module, result
