"""Disposable Torch compiler process; called only inside a Rust cache transaction."""
import json
import sys
from torch.utils.cpp_extension import load

if __name__ == '__main__':
    with open(sys.argv[1]) as stream:
        recipe = json.load(stream)
    load(**recipe, with_cuda=True, verbose=True)
