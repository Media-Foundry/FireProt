// Experimental gfx90a dual-projection tiles. Preserve FP32 MFMA accumulation order.
#include <hip/hip_runtime.h>
#include <cmath>
using V4 = float __attribute__((ext_vector_type(4)));
template<int R, int C, int K, int W>
__global__ __launch_bounds__(W*64) void dual_tile(const float* __restrict__ x,
 const float* __restrict__ wg, const float* __restrict__ wp,
 float* __restrict__ out, int rows, int stride) {
#if defined(__gfx90a__)
    static_assert(R%(16*W)==0 && C%16==0 && 128%K==0);
    __shared__ float input[R][K+4], gate_weight[C][K+4], value_weight[C][K+4];
    const int tid=threadIdx.x, lane=tid%64, wave=tid/64;
    const int first=blockIdx.x*R, channel=blockIdx.y*C;
    V4 gate[R/(16*W)][C/16]={}, value[R/(16*W)][C/16]={};
    for(int base=0;base<128;base+=K) {
        for(int i=tid;i<R*(K/4);i+=64*W) {
            const int r=i/(K/4), k=(i%(K/4))*4;
            float4 v=make_float4(0,0,0,0);
            if(first+r<rows) v=*reinterpret_cast<const float4*>(x+(first+r)*128+base+k);
            *reinterpret_cast<float4*>(&input[r][k])=v;
        }
        for(int i=tid;i<C*(K/4);i+=64*W) {
            const int r=i/(K/4), k=(i%(K/4))*4, wi=(channel+r)*128+base+k;
            *reinterpret_cast<float4*>(&gate_weight[r][k])=*reinterpret_cast<const float4*>(wg+wi);
            *reinterpret_cast<float4*>(&value_weight[r][k])=*reinterpret_cast<const float4*>(wp+wi);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<K;k+=4) {
            #pragma unroll
            for(int p=0;p<R/(16*W);p++) {
                const float b=input[p*(16*W)+wave*16+lane%16][k+lane/16];
                #pragma unroll
                for(int g=0;g<C/16;g++) {
                    const int r=g*16+lane%16;
                    gate[p][g]=__builtin_amdgcn_mfma_f32_16x16x4f32(gate_weight[r][k+lane/16],b,gate[p][g],0,0,0);
                    value[p][g]=__builtin_amdgcn_mfma_f32_16x16x4f32(value_weight[r][k+lane/16],b,value[p][g],0,0,0);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int p=0;p<R/(16*W);p++) {
        #pragma unroll
        for(int g=0;g<C/16;g++) {
            #pragma unroll
            for(int j=0;j<4;j++) {
                const int r=first+p*(16*W)+wave*16+lane%16;
                if(r<rows) out[(channel+g*16+lane/16*4+j)*stride+r]=
                    (1.f/(1.f+expf(-gate[p][g][j])))*value[p][g][j];
            }
        }
    }
#endif
}
extern "C" int base_dual_tile(const float* x,const float* wg,const float* wp,
 float* out,int rows,int stride,int variant,hipStream_t stream) {
    if(!x || !wg || !wp || !out || rows<=0 || rows>1000000 || stride<rows) return -1;
    #define RUN(ID,R,C,K,W) case ID: hipLaunchKernelGGL((dual_tile<R,C,K,W>),dim3((rows+R-1)/R,128/C),dim3(W*64),0,stream,x,wg,wp,out,rows,stride); break
    switch(variant) {
        RUN(0,128,128,16,8);
        RUN(1,64,128,16,4);
        RUN(2,128,64,16,4);
        RUN(3,128,128,16,4);
        RUN(4,256,128,16,8);
        RUN(5,64,64,32,4);
        RUN(6,128,128,32,8);
        default: return -2;
    }
    #undef RUN
    return static_cast<int>(hipGetLastError());
}
