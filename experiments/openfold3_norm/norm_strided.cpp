// Experimental handwritten FP32/C128 normalization for gfx90a.
// Welford evaluation order follows PyTorch 78258b9de6f4e44e7b8df165d4651bf2e14acf62,
// aten/src/ATen/native/cuda/layer_norm_kernel.cu. See licenses/PyTorch-BSD.txt.
// FireProt changes: independent rows per wave/subwave and a standalone HIP ABI.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

namespace {
struct Moments { float mean, m2, count; };

__device__ __forceinline__ Moments include_value(Moments a, float x) {
    float difference = x - a.mean;
    float count = a.count + 1.0f;
    float mean = a.mean + difference * (1.0f / count);
    return {mean, a.m2 + difference * (x - mean), count};
}

// Operand order deliberately retains the upstream reduction tree.
__device__ __forceinline__ Moments merge(Moments right, Moments left) {
    float difference = right.mean - left.mean;
    float count = left.count + right.count;
    if (count == 0.0f) return {0.0f, 0.0f, 0.0f};
    float reciprocal = 1.0f / count;
    float left_fraction = left.count * reciprocal;
    float right_fraction = right.count * reciprocal;
    return {left_fraction * left.mean + right_fraction * right.mean,
            left.m2 + right.m2 + difference * difference * left.count * right_fraction,
            count};
}

template<int Width>
__global__ void normalize(const float* __restrict__ x,
                          const float* __restrict__ weight,
                          const float* __restrict__ bias,
                          float* __restrict__ y,
                          float* __restrict__ mean_out,
                          float* __restrict__ rstd_out,
                          int64_t rows, int64_t inner_rows, int64_t outer_stride,
                          int64_t inner_stride, float epsilon) {
    constexpr int groups = 256 / Width;
    int lane = threadIdx.x % Width;
    int64_t row = int64_t(blockIdx.x) * groups + threadIdx.x / Width;
    if (row >= rows) return;
    Moments s{0.0f, 0.0f, 0.0f};
    float4 values{};
    if (lane < 32) {
        values = reinterpret_cast<const float4*>(x + (row / inner_rows) * outer_stride + (row % inner_rows) * inner_stride)[lane];
        s = include_value(s, values.x);
        s = include_value(s, values.y);
        s = include_value(s, values.z);
        s = include_value(s, values.w);
    }
    // In the native wave64 tree the first merge pairs 32 populated lanes with
    // 32 empty lanes. Width32 retains that merge explicitly before its tree.
    if constexpr (Width == 32) s = merge(s, {0.0f, 0.0f, 0.0f});
    for (int offset = Width / 2; offset; offset /= 2) {
        Moments other{__shfl_down(s.mean, offset, Width),
                      __shfl_down(s.m2, offset, Width),
                      __shfl_down(s.count, offset, Width)};
        s = merge(s, other);
    }
    // Native block has four waves: wave0 merges with empty wave2, then wave1.
    s = merge(s, {0.0f, 0.0f, 0.0f});
    s = merge(s, {0.0f, 0.0f, 0.0f});
    float mean = __shfl(s.mean, 0, Width);
    float inverse = rsqrtf(__shfl(s.m2, 0, Width) / 128.0f + epsilon);
    if (lane < 32) {
        float4 scale = weight ? reinterpret_cast<const float4*>(weight)[lane]
                              : make_float4(1, 1, 1, 1);
        float4 shift = bias ? reinterpret_cast<const float4*>(bias)[lane]
                           : make_float4(0, 0, 0, 0);
        float4 result;
        // Separate affine branches avoid adding a synthetic zero when absent.
        float* a = reinterpret_cast<float*>(&values);
        float* w = reinterpret_cast<float*>(&scale);
        float* b = reinterpret_cast<float*>(&shift);
        float* o = reinterpret_cast<float*>(&result);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float v = inverse * (a[i] - mean);
            if (weight && bias) v = w[i] * v + b[i];
            else if (weight) v = w[i] * v;
            // Keep the native bias-only FMA even when CSE hoists the multiply.
            else if (bias) v = fmaf(inverse, a[i] - mean, b[i]);
            o[i] = v;
        }
        reinterpret_cast<float4*>(y + row * 128)[lane] = result;
    }
    if (lane == 0) {
        if (mean_out) mean_out[row] = mean;
        if (rstd_out) rstd_out[row] = inverse;
    }
}
}

extern "C" int fireprot_norm128_strided(const float* x, const float* weight,
    const float* bias, float* y, float* mean, float* rstd,
    int64_t rows, int64_t inner_rows, int64_t outer_stride,
    int64_t inner_stride, float epsilon, int width, hipStream_t stream) {
    if (!x || !y || inner_rows <= 0 || inner_rows > 16777216
        || rows % inner_rows || outer_stride < 0 || inner_stride < 128
        || outer_stride > INT64_C(2147483648) || inner_stride > INT64_C(2147483648)
        || outer_stride % 4 || inner_stride % 4 || rows <= 0 || rows > 16777216 || !std::isfinite(epsilon)
        || epsilon <= 0 || (width != 32 && width != 64)) return -1;
    if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(y)
         | reinterpret_cast<uintptr_t>(weight) | reinterpret_cast<uintptr_t>(bias)) & 15) return -2;
    int groups = 256 / width;
    dim3 grid((rows + groups - 1) / groups), block(256);
    if (width == 32)
        hipLaunchKernelGGL(normalize<32>, grid, block, 0, stream,
                          x, weight, bias, y, mean, rstd, rows, inner_rows, outer_stride, inner_stride, epsilon);
    else
        hipLaunchKernelGGL(normalize<64>, grid, block, 0, stream,
                          x, weight, bias, y, mean, rstd, rows, inner_rows, outer_stride, inner_stride, epsilon);
    return static_cast<int>(hipGetLastError());
}
