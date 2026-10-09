# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Outpace Bio, Inc.
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2025 NVIDIA Corporation
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for FireProt: native QK/PV with four bounded HIP softmax groups.
# This entire file is Apache-2.0.

"""Experimental global diffusion attention using the published H4 HIP binary.

The caller supplies a prepared SoftmaxFP32 kernel and the existing dispatcher.
The existing source-pinned attention_scope remains responsible for installation.
No recycles, diffusion steps, QK/PV math, output dtype or weights change.
"""
import torch


class GlobalSoftmaxDispatcher:
    def __init__(self, fallback, kernel):
        self.fallback = fallback
        self.kernel = kernel
        self.calls = 0
        self.probability_calls = 0
        self.groups_launched = 0
        self.bias_copies = 0
        self.score_layout_fallbacks = 0

    def _biases_supported(self, scores_or_query, biases):
        if not isinstance(biases, (tuple, list)) or len(biases) != 2:
            return False
        mask, bias = biases
        return (self._tensors_supported((scores_or_query, mask, bias))
                and tuple(mask.shape) == (1, 1, 1, 1, 1536)
                and tuple(bias.shape) == (1, 1, 16, 1536, 1536))

    def _tensors_supported(self, tensors):
        return all(type(t) is torch.Tensor and t.is_cuda and t.dtype == torch.float32
                   and t.device.index == self.kernel.device and not t.requires_grad
                   and not t.is_conj() and not t.is_neg()
                   and all(stride >= 0 for stride in t.stride()) for t in tensors)

    def eligible(self, query, key, value, biases):
        if (not torch.is_inference_mode_enabled() or torch.is_grad_enabled()
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()):
            return False
        if not self._tensors_supported((query, key, value)):
            return False
        return (tuple(query.shape) == tuple(key.shape) == tuple(value.shape)
                == (1, 1, 16, 1536, 48) and self._biases_supported(query, biases)
                and not torch.cuda.is_current_stream_capturing())

    def probabilities(self, scores, biases):
        if (not torch.is_inference_mode_enabled() or torch.is_grad_enabled()
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not self._biases_supported(scores, biases)
                or tuple(scores.shape) != (1, 1, 16, 1536, 1536)
                or not scores.is_contiguous()):
            raise ValueError("Unsupported global diffusion softmax operands")
        mask, bias = biases
        with torch.cuda.device(scores.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Global softmax experiment excludes graph capture")
            if not bias.is_contiguous():
                bias = bias.contiguous()
                self.bias_copies += 1
            output = torch.empty_like(scores)
            stream = torch.cuda.current_stream().cuda_stream
            group_bytes = 4 * 1536 * 1536 * scores.element_size()
            for group in range(4):
                offset = group * group_bytes
                status = self.kernel.entry(
                    scores.data_ptr() + offset, mask.data_ptr(), bias.data_ptr() + offset,
                    output.data_ptr() + offset, 1, 1536, 0, mask.stride(-1),
                    *bias.stride()[-3:], 2, 1, stream)
                if status:
                    raise RuntimeError(f"Global softmax group {group} failed: {status}")
                self.groups_launched += 1
                self.kernel.launches_submitted += 1
            self.probability_calls += 1
            return output

    def __call__(self, native, query, key, value, biases, use_high_precision=False):
        if not self.eligible(query, key, value, biases):
            return self.fallback(native, query, key, value, biases,
                                 use_high_precision=use_high_precision)
        with torch.cuda.device(query.device), torch.amp.autocast("cuda", dtype=torch.float32):
            scores = torch.einsum("...qc, ...kc->...qk", query, key)
            if scores.is_contiguous():
                probabilities = self.probabilities(scores, biases)
            else:
                # Preserve the already-computed native QK for an unexpected layout.
                self.score_layout_fallbacks += 1
                for bias in biases:
                    scores += bias
                probabilities = torch.nn.functional.softmax(scores, dim=-1)
            result = torch.einsum("...qk, ...kc->...qc", probabilities, value)
        self.calls += 1
        return result.to(dtype=query.dtype)

    def manifest(self):
        return dict(calls=self.calls, probability_calls=self.probability_calls,
                    groups_launched=self.groups_launched, bias_copies=self.bias_copies,
                    score_layout_fallbacks=self.score_layout_fallbacks,
                    kernel=self.kernel.receipt,
                    scope="Strict N1536/H16/D48 global diffusion; four published H4 launches per call",
                    limits=["Submitted launch counts do not prove asynchronous completion.",
                            "Whole-model numerical and paired E2E gates are required.",
                            "The shared kernel launch counter includes these group launches."])
