// Experimental packed N1536 QK/softmax fusion. No GPU acceptance or timing yet.
// Softmax follows kernels/amd/fireprot_softmax_fp32.cpp evaluation order.
// Retain the repository's PyTorch BSD attribution for that arithmetic reference.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

using acc4 = float __attribute__((ext_vector_type(4)));

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx90a__)
#error This experimental fusion requires gfx90a
#endif

__device__ __forceinline__ int stage_index(int query, int column) {
    // A bijection within each 256-column row; both producer and consumer use it.
    return query * 256 + (column ^ ((query & 7) << 2));
}

template<bool ApplySoftmax>
__global__ __launch_bounds__(1024, 1)
void qk_softmax(const float* query, const float* key_transposed,
                const float* mask, const float* bias, float* output,
                int64_t mb, int64_t mk, int64_t bh, int64_t bq, int64_t bk) {
#if defined(__gfx90a__)
    constexpr int N = 1536;
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int coordinate = lane & 15;
    const int k_lane = lane >> 4;
    const int query_base = blockIdx.x * 16;
    const uint64_t matrix = blockIdx.y;
    const uint64_t input_base = matrix * N * 32;
    const int row_query = query_base + wave;
    const uint64_t row_base = (matrix * N + row_query) * N;
    const int64_t mask_base = int64_t(matrix / 4) * mb;
    const int64_t bias_base = int64_t(matrix % 4) * bh + int64_t(row_query) * bq;
    __shared__ float stage[16 * 256];

    float q_values[8];
    #pragma unroll
    for (int fragment = 0; fragment < 8; ++fragment) {
        const int k = (fragment / 2) * 8 + 2 * k_lane + (fragment % 2);
        q_values[fragment] = query[input_base + uint64_t(query_base + coordinate) * 32 + k];
    }
    float values[32];
    #pragma unroll
    for (int t = 0; t < 32; ++t)
        values[t] = -std::numeric_limits<float>::infinity();

    // Compute and transpose one 256-column band at a time. This avoids retaining
    // all six old accumulator tiles alongside the final per-row softmax values.
    #pragma unroll
    for (int band = 0; band < 6; ++band) {
        acc4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
        const int key_base = band * 256 + wave * 16;
        #pragma unroll
        for (int fragment = 0; fragment < 8; ++fragment) {
            const int k = (fragment / 2) * 8 + 2 * k_lane + (fragment % 2);
            const float a = key_transposed[input_base + uint64_t(k) * N + key_base + coordinate];
            accumulator = __builtin_amdgcn_mfma_f32_16x16x4f32(
                a, q_values[fragment], accumulator, 0, 0, 0);
        }
        #pragma unroll
        for (int element = 0; element < 4; ++element)
            stage[stage_index(coordinate, wave * 16 + k_lane * 4 + element)] = accumulator[element];
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = band * 256 + lane + j * 64;
            float value = stage[stage_index(wave, lane + j * 64)];
            if constexpr (ApplySoftmax) {
                value = value + mask[mask_base + int64_t(key) * mk];
                value = value + bias[bias_base + int64_t(key) * bk];
                values[band * 4 + j] = value;
            } else {
                output[row_base + key] = value;
            }
        }
        if (band != 5) __syncthreads();
    }
    if constexpr (ApplySoftmax) {
        float maximum = values[0];
        #pragma unroll
        for (int t = 0; t < 32; ++t)
            maximum = maximum > values[t] ? maximum : values[t];
        #pragma unroll
        for (int shift = 32; shift > 0; shift /= 2) {
            const float other = __shfl_xor(maximum, shift, 64);
            maximum = maximum < other ? other : maximum;
        }
        float sum = 0.0f;
        #pragma unroll
        for (int t = 0; t < 32; ++t) {
            values[t] = std::exp(values[t] - maximum);
            sum += values[t];
        }
        #pragma unroll
        for (int shift = 32; shift > 0; shift /= 2)
            sum += __shfl_xor(sum, shift, 64);
        #pragma unroll
        for (int t = 0; t < 24; ++t)
            output[row_base + lane + t * 64] = sum == 0.0f
                ? std::numeric_limits<float>::quiet_NaN() : values[t] / sum;
    }
#endif
}

extern "C" int fireprot_qk_softmax_f32(const float* query, const float* key_transposed,
    const float* mask, const float* bias, float* output, int batch,
    int64_t mb, int64_t mk, int64_t bh, int64_t bq, int64_t bk,
    int softmax, void* stream) {
    if (!query || !key_transposed || !mask || !bias || !output
        || batch < 1 || batch > 1536 || mb < 0 || mk < 0 || bh < 0 || bq < 0 || bk < 0
        || (softmax != 0 && softmax != 1)) return int(hipErrorInvalidValue);
    #define RUN(S) hipLaunchKernelGGL((qk_softmax<S>), dim3(96, batch * 4), dim3(1024), 0, \
        reinterpret_cast<hipStream_t>(stream), query, key_transposed, mask, bias, output, \
        mb, mk, bh, bq, bk)
    if (softmax) { RUN(true); } else { RUN(false); }
    #undef RUN
    return int(hipGetLastError());
}
