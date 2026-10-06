// Original FireProt code; see repository LICENSE.
#include <hip/hip_runtime.h>
#include <cstdint>

// Pack head-last K[B,N,H,D] into native GEMM's contiguous K[B,H,D,N].
// The measured attention head is D=32; bit copying preserves every FP32 bit.
__global__ void pack_key(const uint32_t* src, uint32_t* dst,
                        int64_t batch_stride, int64_t head_stride,
                        int64_t sequence_stride) {
    __shared__ uint32_t tile[32][33];
    const int bh=blockIdx.y, b=bh/4, h=bh%4;
    const int d=threadIdx.x, n=blockIdx.x*32+threadIdx.y;
    #pragma unroll
    for(int i=0;i<32;i+=8)
        tile[threadIdx.y+i][d]=src[b*batch_stride+h*head_stride+(n+i)*sequence_stride+d];
    __syncthreads();
    const int out_n=blockIdx.x*32+threadIdx.x;
    #pragma unroll
    for(int i=0;i<32;i+=8)
        dst[int64_t(bh)*32*1536+(threadIdx.y+i)*1536+out_n]=tile[threadIdx.x][threadIdx.y+i];
}

extern "C" int fireprot_triangle_pack_key(const void* src,void* dst,
    int batch,int64_t bs,int64_t hs,int64_t ns,void* stream) {
    if(!src || !dst || src==dst || batch!=128 || bs!=1536*128 || hs!=32 || ns!=128)
        return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(pack_key,dim3(1536/32,batch*4),dim3(32,8),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint32_t*>(src),
        static_cast<uint32_t*>(dst),bs,hs,ns);
    return int(hipGetLastError());
}
