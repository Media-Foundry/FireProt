// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-OpenFold-3-exception
// FireProt original code; see LICENSE and licenses/ for additional AMD terms.
// Experimental FP32 [M,128] x [16,128]^T projection for gfx90a.
// No model dispatch is installed by this file. Exact GPU gates are required.
#include <hip/hip_runtime.h>
#include <cstdint>

using Vec4 = float __attribute__((ext_vector_type(4)));

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx90a__)
#error This experimental projection requires gfx90a
#endif

__device__ __forceinline__ int input_stage_index(int row, int k) {
    const int fragment = (k / 8) * 2 + (k & 1);
    const int k_lane = (k & 7) / 2;
    return fragment * 256 + (row / 16) * 64 + k_lane * 16 + (row & 15);
}

template<bool Packed>
__global__ __launch_bounds__(256, 1)
void project_128_16(const float* input, const float* weight, float* output, int rows) {
#if defined(__gfx90a__)
    __shared__ __align__(16) float stage[64 * 128];
    const int lane = threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int coordinate = lane & 15;
    const int k_lane = lane >> 4;
    const int row_base = blockIdx.x * 64;
    #pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
        const int flat = threadIdx.x * 4 + chunk * 1024;
        const int row = flat / 128;
        const int k = flat % 128;
        const Vec4 v = *reinterpret_cast<const Vec4*>(input + (row_base + row) * 128 + k);
        #pragma unroll
        for (int e = 0; e < 4; ++e)
            stage[input_stage_index(row, k + e)] = v[e];
    }
    __syncthreads();
    Vec4 accumulator = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 4
    for (int group = 0; group < 8; ++group) {
        float a[4], b[4];
        #pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int fragment = group * 4 + slot;
            const int k = (fragment / 2) * 8 + k_lane * 2 + (fragment & 1);
            a[slot] = weight[coordinate * 128 + k];
            b[slot] = stage[fragment * 256 + wave * 64 + lane];
        }
        #pragma unroll
        for (int slot = 0; slot < 4; ++slot)
            accumulator = __builtin_amdgcn_mfma_f32_16x16x4f32(
                a[slot], b[slot], accumulator, 0, 0, 0);
    }
    const int row = row_base + wave * 16 + coordinate;
    if constexpr (Packed) {
        #pragma unroll
        for (int e = 0; e < 4; ++e)
            output[(k_lane * 4 + e) * rows + row] = accumulator[e];
    } else {
        *reinterpret_cast<Vec4*>(output + row * 16 + k_lane * 4) = accumulator;
    }
#endif
}

extern "C" int fireprot_project_128_16(const float* input, const float* weight,
                                     float* output, int rows, int packed, void* stream) {
    if (!input || !weight || !output || rows <= 0 || rows > 1536 * 1536
        || rows % 64 || (packed != 0 && packed != 1)
        || (reinterpret_cast<uintptr_t>(input) & 15)
        || (reinterpret_cast<uintptr_t>(output) & 15))
        return int(hipErrorInvalidValue);
    #define RUN(P) hipLaunchKernelGGL((project_128_16<P>), dim3(rows / 64), dim3(256), 0, \
        reinterpret_cast<hipStream_t>(stream), input, weight, output, rows)
    if (packed) { RUN(true); } else { RUN(false); }
    #undef RUN
    return int(hipGetLastError());
}
