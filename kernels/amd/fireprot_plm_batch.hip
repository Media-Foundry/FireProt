// SPDX-License-Identifier: GPL-2.0-only WITH OpenFold-3-exception
// Original FireProt batch extensions; retain native BF16 intermediate rounding.
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
__device__ __forceinline__ void rotate(const uint16_t* x, uint16_t* y,
                                      int a, int b, float c, float s) {
    float u = decode(x[a]), v = decode(x[b]);
    y[a] = encode(decode(encode(u*c)) + decode(encode(-v*s)));
    y[b] = encode(decode(encode(v*c)) + decode(encode(u*s)));
}
template<int Heads, int CosStride>
__global__ void rope(const uint16_t* q, const uint16_t* k, const uint16_t* c,
                     const uint16_t* s, uint16_t* qo, uint16_t* ko, int length, int batch) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= batch*length*Heads*32) return;
    const int d=i%32, head=i/32, pos=(head/Heads)%length;
    const int a=head*64+d, b=a+32;
    const float cv=decode(c[pos*CosStride+d]), sv=decode(s[pos*CosStride+d]);
    rotate(q,qo,a,b,cv,sv); rotate(k,ko,a,b,cv,sv);
}
template<int Heads, int CosStride>
int launch_rope(const void* q,const void* k,const void* c,const void* s,
                void* qo,void* ko,int length,int batch,void* stream) {
    if (!q || !k || !c || !s || !qo || !ko || q==qo || k==ko
        || length!=1002 || (batch!=1 && batch!=2 && batch!=4))
        return int(hipErrorInvalidValue);
    hipLaunchKernelGGL((rope<Heads,CosStride>),dim3((batch*length*Heads*32+255)/256),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(q),static_cast<const uint16_t*>(k),
        static_cast<const uint16_t*>(c),static_cast<const uint16_t*>(s),static_cast<uint16_t*>(qo),
        static_cast<uint16_t*>(ko),length,batch);
    return int(hipGetLastError());
}
__global__ void swiglu(const uint16_t* __restrict__ input,
                       uint16_t* __restrict__ output, int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const int row = i / 6144, col = i % 6144;
    const float x = decode(input[row * 12288 + col]);
    const float gate = decode(input[row * 12288 + 6144 + col]);
    const float silu = decode(encode(x / (1.0f + expf(-x))));
    output[i] = encode(silu * gate);
}
}
extern "C" int fireprot_esmc_batch_rope(const void* q,const void* k,const void* c,const void* s,
                                       void* qo,void* ko,int length,int batch,void* stream) {
    return launch_rope<40,32>(q,k,c,s,qo,ko,length,batch,stream);
}
extern "C" int fireprot_atlaslm_batch_rope(const void* q,const void* k,const void* c,const void* s,
                                          void* qo,void* ko,int length,int batch,void* stream) {
    return launch_rope<36,64>(q,k,c,s,qo,ko,length,batch,stream);
}
extern "C" int fireprot_atlaslm_batch_swiglu(const void* input,void* output,int rows,void* stream) {
    if (!input || !output || input==output || (rows!=1002 && rows!=2004 && rows!=4008))
        return int(hipErrorInvalidValue);
    const int elements=rows*6144;
    hipLaunchKernelGGL(swiglu,dim3((elements+255)/256),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(input),
        static_cast<uint16_t*>(output),elements);
    return int(hipGetLastError());
}
