// Experimental packed N1536 QK/softmax/PV fusion. Uncompiled and unqualified.
// Derived from qualified QK source b89129b5; PV even/odd K8 must pass separately.
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
    return query * 256 + (column ^ ((query & 3) | ((query & 12) << 1)));
}

template<bool CaptureProb>
__global__ __launch_bounds__(1024, 1)
void qkv_softmax(const float* query, const float* key_transposed,
                const float* value_matrix, const float* mask, const float* bias, float* output, float* probability_output,
                int64_t mb, int64_t mk, int64_t bh, int64_t bq, int64_t bk) {
#if defined(__gfx90a__)
    constexpr int N = 1536;
    const int lane = threadIdx.x & 63;
    // Fixed 1024x1x1 launch and wave64 make this index wave-uniform.
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int coordinate = lane & 15;
    const int k_lane = lane >> 4;
    // Interleave eight outer matrices sharing each head/query bias tile.
    // Non-multiple-of-eight batches retain the original block mapping.
    const uint32_t linear = blockIdx.y * 96 + blockIdx.x;
    const bool reuse = gridDim.y % 32 == 0;
    const uint32_t head_query = (linear % 3072) / 8;
    const int query_base = (reuse ? head_query % 96 : blockIdx.x) * 16;
    const uint64_t matrix = reuse
        ? ((linear / 3072) * 8 + linear % 8) * 4 + head_query / 96
        : blockIdx.y;
    const uint64_t input_base = matrix * N * 32;
    const int row_query = query_base + wave;
    const uint64_t row_base = (matrix * N + row_query) * N;
    const int64_t mask_base = int64_t(matrix / 4) * mb;
    const int64_t bias_base = int64_t(matrix % 4) * bh + int64_t(row_query) * bq;
    __shared__ __align__(16) float stage[16 * 256];

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
        for (int group = 0; group < 2; ++group) {
            float prefetched[4];
            #pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const int fragment = group * 4 + slot;
                const int k = (fragment / 2) * 8 + 2 * k_lane + (fragment % 2);
                prefetched[slot] = key_transposed[input_base + uint64_t(k) * N + key_base + coordinate];
            }
            #pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                accumulator = __builtin_amdgcn_mfma_f32_16x16x4f32(
                    prefetched[slot], q_values[group * 4 + slot], accumulator, 0, 0, 0);
            }
        }
        // Physical low column bits are XOR-permuted by the query coordinate.
        // Reorder registers to place four values in consecutive aligned dwords.
        const acc4 swapped_pairs = (coordinate & 1)
            ? __builtin_shufflevector(accumulator, accumulator, 1, 0, 3, 2) : accumulator;
        const acc4 packed = (coordinate & 2)
            ? __builtin_shufflevector(swapped_pairs, swapped_pairs, 2, 3, 0, 1) : swapped_pairs;
        const int destination = stage_index(coordinate, wave * 16 + k_lane * 4) & ~3;
        *reinterpret_cast<acc4*>(&stage[destination]) = packed;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = band * 256 + lane + j * 64;
            float value = stage[stage_index(wave, lane + j * 64)];
            values[band * 4 + j] = value;
        }
        if (band != 5) __syncthreads();
    }
    {
        // Delay input additions until all shared-memory exchange is complete.
        // Each element still evaluates score + mask, then + bias, before max.
        #pragma unroll
        for (int t = 0; t < 24; ++t) {
            const int key = lane + t * 64;
            values[t] = values[t] + mask[mask_base + int64_t(key) * mk];
            values[t] = values[t] + bias[bias_base + int64_t(key) * bk];
        }
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
        for (int t = 0; t < 24; ++t) {
            values[t] = sum == 0.0f
                ? std::numeric_limits<float>::quiet_NaN() : values[t] / sum;
            if constexpr (CaptureProb)
                probability_output[row_base + lane + t * 64] = values[t];
        }
        // All QK stage readers must finish before probability staging begins.
        __syncthreads();
        acc4 pv_accumulator = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int band = 0; band < 6; ++band) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                stage[stage_index(wave, lane + j * 64)] = values[band * 4 + j];
            __syncthreads();
            if (wave < 2) {
                const int feature_base = wave * 16;
                #pragma unroll 1
                for (int group = 0; group < 16; ++group) {
                    float av[4], bv[4];
                    #pragma unroll
                    for (int slot = 0; slot < 4; ++slot) {
                        const int fragment = group * 4 + slot;
                        const int local_k = (fragment / 2) * 8 + 2 * k_lane + (fragment % 2);
                        av[slot] = value_matrix[(matrix * N + band * 256 + local_k) * 32
                                                + feature_base + coordinate];
                        bv[slot] = stage[stage_index(coordinate, local_k)];
                    }
                    #pragma unroll
                    for (int slot = 0; slot < 4; ++slot)
                        pv_accumulator = __builtin_amdgcn_mfma_f32_16x16x4f32(
                            av[slot], bv[slot], pv_accumulator, 0, 0, 0);
                }
            }
            __syncthreads();
        }
        if (wave < 2) {
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                output[(matrix * N + query_base + coordinate) * 32
                       + wave * 16 + k_lane * 4 + e] = pv_accumulator[e];
        }
    }
#endif
}

extern "C" int fireprot_qkv_softmax_f32(const float* query, const float* key_transposed,
    const float* value, const float* mask, const float* bias, float* output,
    float* probability_output, int batch,
    int64_t mb, int64_t mk, int64_t bh, int64_t bq, int64_t bk, void* stream) {
    if (!query || !key_transposed || !value || !mask || !bias || !output
        || (batch != 32 && batch != 128) || mb < 0 || mk < 0 || bh < 0 || bq < 0 || bk < 0)
        return int(hipErrorInvalidValue);
    #define RUN(S) hipLaunchKernelGGL((qkv_softmax<S>), dim3(96, batch * 4), dim3(1024), 0, \
        reinterpret_cast<hipStream_t>(stream), query, key_transposed, value, mask, bias, \
        output, probability_output, mb, mk, bh, bq, bk)
    if (probability_output) { RUN(true); } else { RUN(false); }
    #undef RUN
    return int(hipGetLastError());
}
