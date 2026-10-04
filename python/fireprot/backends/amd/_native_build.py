"""Torch compilation boundary for the Rust-owned native build cache.

Maintained Torch extension loaders use this recipe. Host dependency discovery is explicit and the
result records its coverage; it does not claim a hermetic compiler environment.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from fireprot._paths import REPOSITORY_ROOT
import shutil
import subprocess
import sys
import sysconfig


def load(package, sources, *, cxx_flags=('-O3',),
         hip_flags=('-O3', '--offload-arch=gfx90a'), include_paths=()):
    import torch
    from torch.utils import cpp_extension
    from fireprot.ops._bridge import bridge

    rocm = Path(cpp_extension.ROCM_HOME)
    torch_root = Path(torch.__file__).resolve().parent
    helper = Path(__file__).with_name('_compile_native.py')
    guard = Path(os.environ.get('FIREPROT_BUILD_GUARD',
        str(REPOSITORY_ROOT/'target/release/fireprot-compiler-guard')))
    paths = [guard, *sources, __file__, REPOSITORY_ROOT/'python/fireprot/_paths.py', helper, cpp_extension.__file__, sys.executable,
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
    trees.extend(dict(name=f'extra-include-{index}', path=str(Path(root).absolute()), suffixes=[])
                 for index, root in enumerate(include_paths))
    trees.append(dict(name='hipify', path=str(torch_root/'utils/hipify'), suffixes=['.py']))
    inspector = shutil.which('ldd')
    cxx = shutil.which(cpp_extension.get_cxx_compiler())
    if inspector is None:
        raise RuntimeError('ldd is required for the experimental toolchain receipt')
    objects = [guard, sys.executable, cxx, shutil.which('ninja'), rocm/'bin/hipcc',
               rocm/'llvm/bin/clang++', rocm/'llvm/bin/ld.lld',
               rocm/'lib/libamdhip64.so', '/bin/sh', '/bin/bash']
    for program in ('cc1plus', 'collect2', 'ld'):
        resolved = subprocess.check_output([cxx, '-print-prog-name=' + program], text=True).strip()
        resolved = str(Path(resolved).absolute()) if '/' in resolved else shutil.which(resolved)
        if resolved is None:
            raise RuntimeError(f'missing compiler subprogram: {program}')
        objects.append(resolved)
    dynamic_inputs = dict(inspector=inspector, objects=[str(Path(path).absolute()) for path in objects])
    environment = {key: os.environ.get(key) for key in (
        'PATH', 'HOME', 'TMPDIR', 'MAX_JOBS', 'CXX', 'CC', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'CPATH',
        'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'ROCM_HOME',
        'HIP_PATH', 'HIP_CLANG_PATH', 'HIPCC_COMPILE_FLAGS_APPEND',
        'HIPCC_LINK_FLAGS_APPEND', 'PYTORCH_ROCM_ARCH')}
    environment.update(PYTORCH_ROCM_ARCH='gfx90a', PYTHONHASHSEED='0',
                       PYTHONDONTWRITEBYTECODE='1', LC_ALL='C', HIP_VISIBLE_DEVICES='')
    spec = dict(schema_version=1, package=package, target='gfx90a',
        files=[dict(name=str(Path(path).absolute()), path=str(Path(path).absolute()))
               for path in paths], trees=trees, dynamic_inputs=dynamic_inputs, arguments=dict(cxx=list(cxx_flags), hip=list(hip_flags),
            include=[str(Path(path).absolute()) for path in include_paths]),
        environment=environment, versions=dict(recipe='torch-hip-6',
            torch=str(torch.__version__), hip=str(torch.version.hip),
            python=sys.version, soabi=str(sysconfig.get_config_var('SOABI'))))
    cache = Path(os.environ.get('FIREPROT_EXTENSION_CACHE',
        str(REPOSITORY_ROOT/'runtime/torch_extensions')))/'rust-v1'

    process_spec = dict(build=spec, python=str(Path(sys.executable).absolute()),
                        helper=str(helper.absolute()), guard=str(guard.absolute()),
                        lock_timeout_ms=600_000,
                        sources=[str(Path(source).absolute()) for source in sources],
                        timeout_ms=600_000)
    result = json.loads(bridge.build_torch(json.dumps(process_spec), cache))
    name = result['receipt']['identity']['package'] + '_' + result['receipt']['identity']['sha256']
    library = Path(result['directory'])/(name+'.so')
    module_spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    result['dependency_scope'] = 'Rust-enumerated Torch/ROCm/Python/system headers, GCC trees, Torch libraries and listed compiler/helper files; load-time dependencies of declared compiler tools and HIP runtime are re-resolved in Rust; runtime plugins and the complete model environment remain outside this recipe'
    return module, result
