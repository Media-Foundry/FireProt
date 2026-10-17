// SPDX-License-Identifier: GPL-2.0-only WITH OpenFold-3-exception
// Experimental ESM3 residual: preserve the BF16 division rounding boundary.
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {
__device__ __forceinline__ float decode(uint16_t x) {
    return __uint_as_float(uint32_t(x) << 16);
}
__device__ __forceinline__ uint16_t encode(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U)
        return uint16_t((bits >> 16) | 0x40U);
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
template<int Unroll>
__global__ void residual(const uint16_t* __restrict__ x,
                         const uint16_t* __restrict__ r,
                         uint16_t* __restrict__ output, int count, float inverse) {
    const int base = blockIdx.x * blockDim.x * Unroll + threadIdx.x;
    #pragma unroll
    for (int j = 0; j < Unroll; ++j) {
        const int i = base + j * blockDim.x;
        if (i < count) {
            const float scaled = decode(encode(decode(r[i]) * inverse));
            output[i] = encode(decode(x[i]) + scaled);
        }
    }
}
union Packed8 {
    uint4 vector;
    uint16_t values[8];
};
__global__ void residual_vector(const uint4* __restrict__ x,
                                const uint4* __restrict__ r,
                                uint4* __restrict__ output, int count, float inverse) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count / 8) return;
    Packed8 a, b, c;
    a.vector = x[i];
    b.vector = r[i];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float scaled = decode(encode(decode(b.values[j]) * inverse));
        c.values[j] = encode(decode(a.values[j]) + scaled);
    }
    output[i] = c.vector;
}
}
extern "C" int fireprot_esm3_residual(const void* x, const void* r, void* output,
                                     int count, float inverse, int unroll, void* stream) {
    if (!x || !r || !output || x == output || r == output || count <= 0)
        return int(hipErrorInvalidValue);
    if (unroll != 1 && unroll != 4 && unroll != 8 && unroll != 16) return int(hipErrorInvalidValue);
    const auto s = reinterpret_cast<hipStream_t>(stream);
    if (unroll == 16) {
        if (count % 8) return int(hipErrorInvalidValue);
        hipLaunchKernelGGL(residual_vector, dim3((count / 8 + 255) / 256), dim3(256), 0, s,
            static_cast<const uint4*>(x), static_cast<const uint4*>(r),
            static_cast<uint4*>(output), count, inverse);
        return int(hipGetLastError());
    }
    const auto a = static_cast<const uint16_t*>(x);
    const auto b = static_cast<const uint16_t*>(r);
    const auto c = static_cast<uint16_t*>(output);
    const dim3 grid((count + 256 * unroll - 1) / (256 * unroll));
    if (unroll == 1) hipLaunchKernelGGL(residual<1>, grid, dim3(256), 0, s, a, b, c, count, inverse);
    if (unroll == 4) hipLaunchKernelGGL(residual<4>, grid, dim3(256), 0, s, a, b, c, count, inverse);
    if (unroll == 8) hipLaunchKernelGGL(residual<8>, grid, dim3(256), 0, s, a, b, c, count, inverse);
    return int(hipGetLastError());
}
