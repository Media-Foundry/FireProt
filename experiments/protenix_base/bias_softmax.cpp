#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

// Experimental Protenix boundary: optional mask, ordered bias addition, private scores.
// Derived from kernels/amd/fireprot_softmax_fp32.cpp; retain native QK/PV arithmetic.
// Match the observed Torch persistent wave64 FP32 evaluation order.
// Shape and buffer ownership checks belong to the tensor boundary.
template<int N, int W>
__global__ void fused(const float* scores, const float* mask, const float* bias,
                      float* output, int rows, int64_t mb, int64_t mk,
                      int64_t bh, int64_t bq, int64_t bk, bool reuse) {
    constexpr int K = N <= 1024 ? 16 : 32;
    const int linear = blockIdx.x * W + threadIdx.y;
    const int lane = threadIdx.x;
    const int group = 4 * N * 8;
    const int row = reuse && rows % group == 0
        ? (linear / group) * group + (linear % 8) * (4 * N)
            + (linear / 8) % (4 * N)
        : linear;
    if (row >= rows) return; // Uniform across the entire wave.
    const int outer = row / (4 * N);
    const int head = (row / N) % 4;
    const int query = row % N;
    const int64_t base = int64_t(row) * N;
    const int64_t bias_base = int64_t(head) * bh + int64_t(query) * bq;
    const int64_t mask_base = int64_t(outer) * mb;
    float values[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        const int key = lane + t * 64;
        float value = -std::numeric_limits<float>::infinity();
        if (key < N) {
            value = scores[base + key];
            if (mask) value = value + mask[mask_base + int64_t(key) * mk];
            value = value + bias[bias_base + int64_t(key) * bk];
        }
        values[t] = value;
    }
    float maximum = values[0];
    #pragma unroll
    for (int t = 0; t < K; ++t)
        maximum = maximum > values[t] ? maximum : values[t];
    #pragma unroll
    for (int shift = 32; shift > 0; shift /= 2) {
        const float other = __shfl_xor(maximum, shift, 64);
        maximum = maximum < other ? other : maximum;
    }
    float sum = 0.0f;
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        values[t] = std::exp(values[t] - maximum);
        sum += values[t];
    }
    #pragma unroll
    for (int shift = 32; shift > 0; shift /= 2)
        sum += __shfl_xor(sum, shift, 64);
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        const int key = lane + t * 64;
        if (key < N)
            output[base + key] = sum == 0.0f
                ? std::numeric_limits<float>::quiet_NaN() : values[t] / sum;
    }
}

extern "C" int fireprot_base_bias_softmax_f32(const float* scores, const float* mask,
    const float* bias, float* output, int batch, int n, int64_t mb, int64_t mk,
    int64_t bh, int64_t bq, int64_t bk, int waves, int reuse, void* stream) {
    if (!scores || !bias || !output || batch <= 0 || batch > 1536
        || (n != 600 && n != 1000 && n != 1536) || (waves != 2 && waves != 4)
        || mb < 0 || mk < 0 || bh < 0 || bq < 0 || bk < 0
        || (reuse != 0 && reuse != 1)) return int(hipErrorInvalidValue);
    const int rows = batch * 4 * n;
    #define RUN(N,W) hipLaunchKernelGGL((fused<N,W>), dim3((rows+W-1)/W), \
        dim3(64,W), 0, reinterpret_cast<hipStream_t>(stream), scores, mask, bias, \
        output, rows, mb, mk, bh, bq, bk, bool(reuse))
    if (n == 600) {
        if (waves == 2) { RUN(600,2); } else { RUN(600,4); }
    } else if (n == 1000) {
        if (waves == 2) { RUN(1000,2); } else { RUN(1000,4); }
    } else {
        if (waves == 2) { RUN(1536,2); } else { RUN(1536,4); }
    }
    #undef RUN
    return int(hipGetLastError());
}
