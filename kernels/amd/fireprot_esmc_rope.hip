// SPDX-License-Identifier: GPL-2.0-only WITH OpenFold-3-exception
// Original FireProt implementation. Preserve both BF16 products before addition.
#include <hip/hip_runtime.h>
#include <cstdint>
__device__ __forceinline__ float decode(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }
__device__ __forceinline__ uint16_t encode(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return uint16_t((bits >> 16) | 0x40U);
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ void rotate(const uint16_t* x, uint16_t* y, int a, int b, float c, float s) {
    float u = decode(x[a]), v = decode(x[b]);
    // Native PyTorch computes two BF16 multiplies and then a BF16 add.
    y[a] = encode(decode(encode(u*c)) + decode(encode(-v*s)));
    y[b] = encode(decode(encode(v*c)) + decode(encode(u*s)));
}
template<int Heads, int CosStride>
__global__ void rope(const uint16_t* q, const uint16_t* k, const uint16_t* c,
                     const uint16_t* s, uint16_t* qo, uint16_t* ko, int length) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= length*Heads*32) return;
    int d=i%32, head=i/32, pos=head/Heads;
    int a=head*64+d, b=a+32;
    float cv=decode(c[pos*CosStride+d]), sv=decode(s[pos*CosStride+d]);
    rotate(q,qo,a,b,cv,sv); rotate(k,ko,a,b,cv,sv);
}
extern "C" int fireprot_esmc_rope(const void* q,const void* k,const void* c,const void* s,
                                 void* qo,void* ko,int length,void* stream) {
    if (!q || !k || !c || !s || !qo || !ko || length<3 || length>1002) return int(hipErrorInvalidValue);
    hipLaunchKernelGGL((rope<40,32>),dim3((length*40*32+255)/256),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(q),static_cast<const uint16_t*>(k),
        static_cast<const uint16_t*>(c),static_cast<const uint16_t*>(s),static_cast<uint16_t*>(qo),static_cast<uint16_t*>(ko),length);
    return int(hipGetLastError());
}

extern "C" int fireprot_atlaslm_rope(const void* q,const void* k,const void* c,const void* s,
                                 void* qo,void* ko,int length,void* stream) {
    if (!q || !k || !c || !s || !qo || !ko || length<3 || length>1002) return int(hipErrorInvalidValue);
    hipLaunchKernelGGL((rope<36,64>),dim3((length*36*32+255)/256),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(q),static_cast<const uint16_t*>(k),
        static_cast<const uint16_t*>(c),static_cast<const uint16_t*>(s),static_cast<uint16_t*>(qo),static_cast<uint16_t*>(ko),length);
    return int(hipGetLastError());
}
