// Experimental score-only probe. Lane mapping and K order require GPU validation.
// This is not an inference implementation or a claimed performance optimization.
#include <hip/hip_runtime.h>
#include <cstdint>

using acc4 = float __attribute__((ext_vector_type(4)));

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx90a__)
#error This experimental probe requires gfx90a
#endif

__global__ __launch_bounds__(64)
void qk_probe(const float* query, const float* key_transposed, float* output,
              int n, bool raw_tile) {
#if defined(__gfx90a__)
    const int lane = threadIdx.x;
    const int coordinate = lane & 15;
    const int k_lane = lane >> 4;
    const int key_base = blockIdx.x * 16;
    const int query_base = blockIdx.y * 16;
    const uint64_t batch = blockIdx.z;
    const uint64_t input_base = batch * uint64_t(n) * 32;
    acc4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};

    // raw_tile executes ONE K4 instruction for integer lane calibration.
    // The K32 trial follows the native candidate order supported by the LDS
    // review and retained-tile screen: even K entries, then odd, within each K8.
    // Keep raw calibration independent of this full-reduction address policy.
    const int fragments = raw_tile ? 1 : 8;
    #pragma unroll
    for (int fragment = 0; fragment < fragments; ++fragment) {
        const int k = raw_tile ? k_lane
                              : (fragment / 2) * 8 + 2 * k_lane + (fragment % 2);
        const float a = key_transposed[input_base + uint64_t(k) * n
                                        + key_base + coordinate];
        const float b = query[input_base + uint64_t(query_base + coordinate) * 32 + k];
        accumulator = __builtin_amdgcn_mfma_f32_16x16x4f32(
            a, b, accumulator, 0, 0, 0);
    }
    #pragma unroll
    for (int element = 0; element < 4; ++element) {
        if (raw_tile) {
            output[lane * 4 + element] = accumulator[element];
        } else {
            const int query_index = query_base + coordinate;
            const int key_index = key_base + k_lane * 4 + element;
            output[(batch * n + query_index) * uint64_t(n) + key_index] = accumulator[element];
        }
    }
#endif
}

extern "C" int fireprot_qk_mfma16_f32(const float* query, const float* key_transposed,
                                     float* output, int batches, int n, int raw_tile,
                                     hipStream_t stream) {
    if (!query || !key_transposed || !output || batches < 1 || batches > 4096
        || (n != 16 && n != 1536) || (raw_tile != 0 && raw_tile != 1)
        || (raw_tile && (n != 16 || batches != 1))) {
        return int(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qk_probe, dim3(n / 16, n / 16, batches), dim3(64), 0,
                      stream, query, key_transposed, output, n, bool(raw_tile));
    return int(hipGetLastError());
}
