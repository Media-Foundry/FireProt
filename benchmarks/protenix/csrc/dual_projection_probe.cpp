// Experimental gfx90a FP32 dual projection with fused sigmoid output.
#include <hip/hip_runtime.h>
#include <cmath>
using V4 = float __attribute__((ext_vector_type(4)));
#ifndef CHANNEL_TILE
#define CHANNEL_TILE 32
#endif
#ifndef K_TILE
#define K_TILE 32
#endif
__global__ void dual_projection(const float* __restrict__ x,
 const float* __restrict__ wg,const float* __restrict__ wp,float* __restrict__ out,int rows) {
    __shared__ float input[64][K_TILE+4];
    __shared__ float gate_weight[CHANNEL_TILE][K_TILE+4],value_weight[CHANNEL_TILE][K_TILE+4];
    const int tid=threadIdx.x,lane=tid%64,wave=tid/64;
    const int first=blockIdx.x*64,channel=blockIdx.y*CHANNEL_TILE;
    V4 gate[CHANNEL_TILE/16]={},value[CHANNEL_TILE/16]={};
    for(int base=0;base<128;base+=K_TILE){
        for(int i=tid;i<64*(K_TILE/4);i+=256){
            const int r=i/(K_TILE/4),k=(i%(K_TILE/4))*4;
            float4 v=make_float4(0,0,0,0);
            if(first+r<rows)v=*reinterpret_cast<const float4*>(x+(first+r)*128+base+k);
            *reinterpret_cast<float4*>(&input[r][k])=v;
        }
        for(int i=tid;i<CHANNEL_TILE*(K_TILE/4);i+=256){
            const int r=i/(K_TILE/4),k=(i%(K_TILE/4))*4,wi=(channel+r)*128+base+k;
            *reinterpret_cast<float4*>(&gate_weight[r][k])=*reinterpret_cast<const float4*>(wg+wi);
            *reinterpret_cast<float4*>(&value_weight[r][k])=*reinterpret_cast<const float4*>(wp+wi);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<K_TILE;k+=4){
            const float b=input[wave*16+lane%16][k+lane/16];
            #pragma unroll
            for(int g=0;g<CHANNEL_TILE/16;g++){
                const int r=g*16+lane%16;
                gate[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(gate_weight[r][k+lane/16],b,gate[g],0,0,0);
                value[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(value_weight[r][k+lane/16],b,value[g],0,0,0);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int g=0;g<CHANNEL_TILE/16;g++){
        #pragma unroll
        for(int j=0;j<4;j++){
            const int r=first+wave*16+lane%16;
            if(r<rows)out[(channel+g*16+lane/16*4+j)*rows+r]=(1.f/(1.f+expf(-gate[g][j])))*value[g][j];
        }
    }
}
extern "C" int run_dual_projection(const float* x,const float* wg,const float* wp,float* out,int n,void* stream){
    if(n!=1000 && n!=1536)return 2;
    if(!x||!wg||!wp||!out)return 3;
    hipLaunchKernelGGL(dual_projection,dim3((n*n+63)/64,128/CHANNEL_TILE),dim3(256),0,static_cast<hipStream_t>(stream),x,wg,wp,out,n*n);
    return static_cast<int>(hipGetLastError());
}
