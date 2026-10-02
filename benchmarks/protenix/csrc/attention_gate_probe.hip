// Out-of-place FP32 sigmoid gate fusion. Inputs remain unchanged.
#include <hip/hip_runtime.h>
#include <cstdint>
__global__ void gate_kernel(const float4* gate,const float4* value,float4* out,int64_t count) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    float4 g=gate[i],v=value[i];
    out[i]=make_float4(v.x/(1.0f+expf(-g.x)),v.y/(1.0f+expf(-g.y)),
                      v.z/(1.0f+expf(-g.z)),v.w/(1.0f+expf(-g.w)));
}
extern "C" int attention_gate(const float* gate,const float* value,float* out,int64_t count,void* stream) {
    if(count<=0 || count%4) return 2;
    hipLaunchKernelGGL(gate_kernel,dim3((count/4+255)/256),dim3(256),0,
                      static_cast<hipStream_t>(stream),reinterpret_cast<const float4*>(gate),
                      reinterpret_cast<const float4*>(value),reinterpret_cast<float4*>(out),count/4);
    return static_cast<int>(hipGetLastError());
}

__global__ void split_kernel(const float4* packed,float4* q,float4* k,float4* v,int64_t vectors) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=vectors) return;
    int64_t base=(i/32)*96+i%32;
    float4 a=packed[base];
    constexpr float scale=0.1767766952966369f;
    q[i]=make_float4(a.x*scale,a.y*scale,a.z*scale,a.w*scale);
    k[i]=packed[base+32];v[i]=packed[base+64];
}
extern "C" int qkv_split_scale(const float* packed,float* q,float* k,float* v,int64_t rows,void* stream) {
    if(rows!=512*1536) return 2;
    hipLaunchKernelGGL(split_kernel,dim3((rows*32+255)/256),dim3(256),0,static_cast<hipStream_t>(stream),
                      reinterpret_cast<const float4*>(packed),reinterpret_cast<float4*>(q),
                      reinterpret_cast<float4*>(k),reinterpret_cast<float4*>(v),rows*32);
    return static_cast<int>(hipGetLastError());
}
