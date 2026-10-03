// FP32 gated residual probe. Preserve separate multiply/add rounding.
#include <hip/hip_runtime.h>
#include <cstdint>
__device__ float update(float g, float x, float z, bool add) {
    float value = (1.0f / (1.0f + expf(-g))) * x;
    return add ? z + value : value;
}
__global__ void residual(const float4* gate, const float4* value, float4* z,
                         std::int64_t count, bool add) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= count) return;
    float4 g = gate[i], v = value[i], old = add ? z[i] : make_float4(0,0,0,0);
    z[i] = make_float4(update(g.x,v.x,old.x,add),update(g.y,v.y,old.y,add),
                      update(g.z,v.z,old.z,add),update(g.w,v.w,old.w,add));
}
extern "C" int triangle_residual(float* gate, float* value, float* z,
                                  int n, int add, void* stream) {
    if((n != 1000 && n != 1536) || (add != 0 && add != 1)) return 2;
    if(!gate || !value || !z || reinterpret_cast<uintptr_t>(gate)%16 ||
       reinterpret_cast<uintptr_t>(value)%16 || reinterpret_cast<uintptr_t>(z)%16 ||
       gate==z || value==z) return 3;
    std::int64_t count = static_cast<std::int64_t>(n)*n*128/4;
    hipLaunchKernelGGL(residual,dim3((count+255)/256),dim3(256),0,
                       static_cast<hipStream_t>(stream),
                       reinterpret_cast<float4*>(gate),reinterpret_cast<float4*>(value),
                       reinterpret_cast<float4*>(z),count,add);
    return static_cast<int>(hipGetLastError());
}
