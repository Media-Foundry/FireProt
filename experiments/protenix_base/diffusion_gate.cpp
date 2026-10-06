// Experimental FP32 attention-output gate plus layout conversion, gfx90a only.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

__global__ void gate_layout(const float4* __restrict__ attention,
                            float4* __restrict__ gate,
                            int row_stride4, int head_stride4) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 1000 * 192) return;
    const int row = index / 192;
    const int channel4 = index % 192;
    const int head = channel4 / 12;
    const int lane4 = channel4 % 12;
    const float4 a = attention[row * row_stride4 + head * head_stride4 + lane4];
    const float4 g = gate[index];
    gate[index] = make_float4(a.x * (1.0f / (1.0f + expf(-g.x))),
                             a.y * (1.0f / (1.0f + expf(-g.y))),
                             a.z * (1.0f / (1.0f + expf(-g.z))),
                             a.w * (1.0f / (1.0f + expf(-g.w))));
}

extern "C" int base_diffusion_gate(const float* attention, float* gate,
                                    int row_stride, int head_stride, void* stream) {
    if (!attention || !gate || attention == gate ||
        reinterpret_cast<uintptr_t>(attention) % 16 ||
        reinterpret_cast<uintptr_t>(gate) % 16) return 2;
    if (!((row_stride == 768 && head_stride == 48) ||
          (row_stride == 48 && head_stride == 48000))) return 3;
    hipLaunchKernelGGL(gate_layout, dim3((1000 * 192 + 255) / 256), dim3(256),
                      0, static_cast<hipStream_t>(stream),
                      reinterpret_cast<const float4*>(attention),
                      reinterpret_cast<float4*>(gate), row_stride / 4, head_stride / 4);
    return static_cast<int>(hipGetLastError());
}
