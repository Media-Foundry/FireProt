"""Experimental HIP packing for the native N1536/H16 linear_z layout."""
import ctypes
import hashlib
import json
from pathlib import Path

import torch
if __package__:
    from .global_softmax import GlobalSoftmaxDispatcher
else:
    from global_softmax import GlobalSoftmaxDispatcher


class PackedGlobalSoftmaxDispatcher(GlobalSoftmaxDispatcher):
    def __init__(self, fallback, kernel, receipt_path, tile_rows=128):
        super().__init__(fallback, kernel)
        if tile_rows not in (64, 128):
            raise ValueError("Expected 64 or 128 rows per tile")
        self.pack_receipt = json.loads(Path(receipt_path).read_text())
        library = Path(self.pack_receipt["library"])
        if hashlib.sha256(library.read_bytes()).hexdigest() != self.pack_receipt["library_sha256"]:
            raise ValueError("HIP packing library digest mismatch")
        self.pack_library = ctypes.CDLL(str(library))
        self.pack_entry = self.pack_library.fireprot_pack_bias_h16_f32
        self.pack_entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.pack_entry.restype = ctypes.c_int
        self.tile_rows = tile_rows
        self.hip_bias_copies = 0

    def pack_bias(self, bias):
        if (not self._tensors_supported((bias,))
                or tuple(bias.shape) != (1, 1, 16, 1536, 1536)
                or tuple(bias.stride()[-3:]) != (1, 24576, 16)
                or not torch.is_inference_mode_enabled() or torch.is_grad_enabled()
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()):
            raise ValueError("Unsupported HIP bias packing operand")
        with torch.cuda.device(bias.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("HIP bias packing excludes graph capture")
            output = torch.empty(bias.shape, device=bias.device, dtype=bias.dtype)
            status = self.pack_entry(bias.data_ptr(), output.data_ptr(), self.tile_rows,
                                     torch.cuda.current_stream().cuda_stream)
            if status:
                raise RuntimeError(f"HIP bias packing failed: {status}")
            self.hip_bias_copies += 1
            return output

    def probabilities(self, scores, biases):
        # Validate before launching anything; the parent checks the remaining contract.
        if (self._biases_supported(scores, biases)
                and tuple(scores.shape) == (1, 1, 16, 1536, 1536)
                and scores.is_contiguous() and tuple(biases[1].stride()[-3:]) == (1, 24576, 16)):
            biases = [biases[0], self.pack_bias(biases[1])]
        return super().probabilities(scores, biases)

    def manifest(self):
        report = super().manifest()
        report.update(hip_bias_copies=self.hip_bias_copies, pack_tile_rows=self.tile_rows,
                      pack_kernel=self.pack_receipt)
        return report
