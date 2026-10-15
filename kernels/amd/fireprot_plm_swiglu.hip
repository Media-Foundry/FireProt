// SPDX-License-Identifier: GPL-2.0-only WITH OpenFold-3-exception
// Original FireProt implementation. Preserve the intermediate BF16 SiLU result.
#include <hip/hip_runtime.h>
#include <cstdint>

__device__ __forceinline__ float decode(uint16_t x) {
    return __uint_as_float(uint32_t(x) << 16);
}
__device__ __forceinline__ uint16_t encode(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U)
        return uint16_t((bits >> 16) | 0x40U);
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

__global__ void swiglu(const uint16_t* __restrict__ input,
                       uint16_t* __restrict__ output, int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const int row = i / 6144, col = i % 6144;
    const float x = decode(input[row * 12288 + col]);
    const float gate = decode(input[row * 12288 + 6144 + col]);
    const float silu = decode(encode(x / (1.0f + expf(-x))));
    output[i] = encode(silu * gate);
}

extern "C" int fireprot_atlaslm_swiglu(const void* input, void* output,
                                       int rows, void* stream) {
    if (!input || !output || input == output || rows != 1002)
        return int(hipErrorInvalidValue);
    const int elements = rows * 6144;
    hipLaunchKernelGGL(swiglu, dim3((elements + 255) / 256), dim3(256), 0,
        reinterpret_cast<hipStream_t>(stream), static_cast<const uint16_t*>(input),
        static_cast<uint16_t*>(output), elements);
    return int(hipGetLastError());
}
