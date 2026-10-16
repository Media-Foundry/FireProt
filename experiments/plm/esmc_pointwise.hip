// SPDX-License-Identifier: GPL-2.0-only WITH OpenFold-3-exception
// Experimental ESMC pointwise fusion; every native BF16 rounding is retained.
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {
__device__ __forceinline__ float decode(uint16_t x) {
    return __uint_as_float(uint32_t(x) << 16);
}
__device__ __forceinline__ uint16_t encode(float x) {
    uint32_t b = __float_as_uint(x);
    if ((b & 0x7fffffffU) > 0x7f800000U) return uint16_t((b >> 16) | 0x40U);
    return uint16_t((b + 0x7fffU + ((b >> 16) & 1U)) >> 16);
}
template<int V> struct alignas(2*V) Pack { uint16_t values[V]; };
template<int V, bool Residual>
__global__ void pointwise(const Pack<V>* a, const Pack<V>* b, Pack<V>* out,
                          int count, float reciprocal) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Pack<V> x, y, z;
    if constexpr (Residual) { x = a[i]; y = b[i]; }
    else {
        const int row = i / (6912/V), col = i % (6912/V);
        x = a[row*(13824/V)+col]; y = a[row*(13824/V)+6912/V+col];
    }
    #pragma unroll
    for (int j=0; j<V; ++j) {
        if constexpr (Residual)
            z.values[j] = encode(decode(x.values[j]) + decode(encode(decode(y.values[j])*reciprocal)));
        else {
            const float value = decode(x.values[j]);
            z.values[j] = encode(decode(encode(value/(1.0f+expf(-value))))*decode(y.values[j]));
        }
    }
    out[i] = z;
}
template<int V, bool R>
int launch(const void* a, const void* b, void* out, int rows, float reciprocal, void* stream) {
    const int count = rows * (R ? 2560 : 6912) / V;
    hipLaunchKernelGGL((pointwise<V,R>),dim3((count+255)/256),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const Pack<V>*>(a),
        static_cast<const Pack<V>*>(b),static_cast<Pack<V>*>(out),count,reciprocal);
    return int(hipGetLastError());
}
}
extern "C" int esmc_pointwise(int kind, int vector, const void* a, const void* b,
                               void* out, int rows, float reciprocal, void* stream) {
    if (!a || !out || a==out || b==out || (kind!=0 && kind!=1) || (kind==0 && !b)
        || (rows!=1002 && rows!=2004 && rows!=4008)) return int(hipErrorInvalidValue);
    if (vector!=1 && vector!=2 && vector!=4 && vector!=8) return int(hipErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(a)%(2*vector) || reinterpret_cast<uintptr_t>(out)%(2*vector)
        || (b && reinterpret_cast<uintptr_t>(b)%(2*vector))) return int(hipErrorInvalidValue);
    #define CALL(V) case V: return kind==0 ? launch<V,true>(a,b,out,rows,reciprocal,stream) : launch<V,false>(a,b,out,rows,reciprocal,stream)
    switch(vector) { CALL(1); CALL(2); CALL(4); CALL(8); }
    #undef CALL
    return int(hipErrorInvalidValue);
}
