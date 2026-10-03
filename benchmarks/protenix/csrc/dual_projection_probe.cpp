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
#ifndef ROW_TILE
#define ROW_TILE 64
#endif
#ifndef WAVES
#define WAVES 4
#endif
static_assert(ROW_TILE%(16*WAVES)==0 && 128%CHANNEL_TILE==0 && 128%K_TILE==0);
#if BOUNDED_LAUNCH
#define DUAL_BOUNDS __launch_bounds__(64*WAVES)
#else
#define DUAL_BOUNDS
#endif
__global__ DUAL_BOUNDS void dual_projection(const float* __restrict__ x,
 const float* __restrict__ wg,const float* __restrict__ wp,float* __restrict__ out,int rows,int stride) {
    __shared__ float input[ROW_TILE][K_TILE+4];
    __shared__ float gate_weight[CHANNEL_TILE][K_TILE+4],value_weight[CHANNEL_TILE][K_TILE+4];
    const int tid=threadIdx.x,lane=tid%64,wave=tid/64;
    const int first=blockIdx.x*ROW_TILE,channel=blockIdx.y*CHANNEL_TILE;
    V4 gate[ROW_TILE/(16*WAVES)][CHANNEL_TILE/16]={},value[ROW_TILE/(16*WAVES)][CHANNEL_TILE/16]={};
    for(int base=0;base<128;base+=K_TILE){
        for(int i=tid;i<ROW_TILE*(K_TILE/4);i+=64*WAVES){
            const int r=i/(K_TILE/4),k=(i%(K_TILE/4))*4;
            float4 v=make_float4(0,0,0,0);
            if(first+r<rows)v=*reinterpret_cast<const float4*>(x+(first+r)*128+base+k);
            *reinterpret_cast<float4*>(&input[r][k])=v;
        }
        for(int i=tid;i<CHANNEL_TILE*(K_TILE/4);i+=64*WAVES){
            const int r=i/(K_TILE/4),k=(i%(K_TILE/4))*4,wi=(channel+r)*128+base+k;
            *reinterpret_cast<float4*>(&gate_weight[r][k])=*reinterpret_cast<const float4*>(wg+wi);
            *reinterpret_cast<float4*>(&value_weight[r][k])=*reinterpret_cast<const float4*>(wp+wi);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<K_TILE;k+=4){
            #pragma unroll
            for(int p=0;p<ROW_TILE/(16*WAVES);p++) {
                const float b=input[p*(16*WAVES)+wave*16+lane%16][k+lane/16];
                #pragma unroll
                for(int g=0;g<CHANNEL_TILE/16;g++){
                    const int r=g*16+lane%16;
                    gate[p][g]=__builtin_amdgcn_mfma_f32_16x16x4f32(gate_weight[r][k+lane/16],b,gate[p][g],0,0,0);
                    value[p][g]=__builtin_amdgcn_mfma_f32_16x16x4f32(value_weight[r][k+lane/16],b,value[p][g],0,0,0);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int p=0;p<ROW_TILE/(16*WAVES);p++) {
        #pragma unroll
        for(int g=0;g<CHANNEL_TILE/16;g++){
            #pragma unroll
            for(int j=0;j<4;j++){
                const int r=first+p*(16*WAVES)+wave*16+lane%16;
                if(r<rows)out[(channel+g*16+lane/16*4+j)*stride+r]=(1.f/(1.f+expf(-gate[p][g][j])))*value[p][g][j];
            }
        }
    }
}
extern "C" int run_dual_projection(const float* x,const float* wg,const float* wp,float* out,int n,void* stream){
    if(n!=1000 && n!=1536)return 2;
    if(!x||!wg||!wp||!out)return 3;
    hipLaunchKernelGGL(dual_projection,dim3((n*n+ROW_TILE-1)/ROW_TILE,128/CHANNEL_TILE),dim3(64*WAVES),0,static_cast<hipStream_t>(stream),x,wg,wp,out,n*n,n*n);
    return static_cast<int>(hipGetLastError());
}

extern "C" int run_dual_projection_strided(const float* x,const float* wg,const float* wp,float* out,int n,int padding,void* stream){
    if((n!=1000 && n!=1536) || (padding!=0 && padding!=16 && padding!=64 && padding!=128))return 2;
    if(!x||!wg||!wp||!out)return 3;
    hipLaunchKernelGGL(dual_projection,dim3((n*n+ROW_TILE-1)/ROW_TILE,128/CHANNEL_TILE),dim3(64*WAVES),0,static_cast<hipStream_t>(stream),x,wg,wp,out,n*n,n*n+padding);
    return static_cast<int>(hipGetLastError());
}
