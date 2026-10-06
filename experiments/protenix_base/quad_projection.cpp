// Experimental fused A/B projections; four FP32 products share input LDS loads.
#include <hip/hip_runtime.h>
#include <cmath>
using V4 = float __attribute__((ext_vector_type(4)));
template<int R,int C,int K,int W>
__global__ __launch_bounds__(W*64) void quad(const float* __restrict__ x,
 const float* __restrict__ ag,const float* __restrict__ ap,
 const float* __restrict__ bg,const float* __restrict__ bp,
 float* __restrict__ ao,float* __restrict__ bo,int rows,int stride) {
#if defined(__gfx90a__)
    static_assert(R%(16*W)==0 && C%16==0 && 128%K==0);
    __shared__ float input[R][K+4], weights[4][C][K+4];
    const int tid=threadIdx.x,lane=tid%64,wave=tid/64;
    const int first=blockIdx.x*R,channel=blockIdx.y*C;
    V4 sums[4][R/(16*W)][C/16]={};
    for(int base=0;base<128;base+=K) {
        for(int i=tid;i<R*(K/4);i+=64*W) {
            const int r=i/(K/4),k=(i%(K/4))*4;
            float4 v=make_float4(0,0,0,0);
            if(first+r<rows) v=*reinterpret_cast<const float4*>(x+(first+r)*128+base+k);
            *reinterpret_cast<float4*>(&input[r][k])=v;
        }
        for(int i=tid;i<C*(K/4);i+=64*W) {
            const int r=i/(K/4),k=(i%(K/4))*4,wi=(channel+r)*128+base+k;
            *reinterpret_cast<float4*>(&weights[0][r][k])=*reinterpret_cast<const float4*>(ag+wi);
            *reinterpret_cast<float4*>(&weights[1][r][k])=*reinterpret_cast<const float4*>(ap+wi);
            *reinterpret_cast<float4*>(&weights[2][r][k])=*reinterpret_cast<const float4*>(bg+wi);
            *reinterpret_cast<float4*>(&weights[3][r][k])=*reinterpret_cast<const float4*>(bp+wi);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<K;k+=4) {
            #pragma unroll
            for(int p=0;p<R/(16*W);p++) {
                const float b=input[p*(16*W)+wave*16+lane%16][k+lane/16];
                #pragma unroll
                for(int g=0;g<C/16;g++) {
                    #pragma unroll
                    for(int q=0;q<4;q++)
                        sums[q][p][g]=__builtin_amdgcn_mfma_f32_16x16x4f32(weights[q][g*16+lane%16][k+lane/16],b,sums[q][p][g],0,0,0);
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
                const int index=(channel+g*16+lane/16*4+j)*stride+r;
                if(r<rows) {
                    ao[index]=(1.f/(1.f+expf(-sums[0][p][g][j])))*sums[1][p][g][j];
                    bo[index]=(1.f/(1.f+expf(-sums[2][p][g][j])))*sums[3][p][g][j];
                }
            }
        }
    }
#endif
}
extern "C" int base_quad(const float* x,const float* ag,const float* ap,
 const float* bg,const float* bp,float* ao,float* bo,int rows,int stride,int variant,hipStream_t stream) {
    if(!x||!ag||!ap||!bg||!bp||!ao||!bo||rows<=0||rows>1000000||stride<rows) return -1;
    #define RUN(ID,R,C,K,W) case ID: hipLaunchKernelGGL((quad<R,C,K,W>),dim3((rows+R-1)/R,128/C),dim3(W*64),0,stream,x,ag,ap,bg,bp,ao,bo,rows,stride); break
    switch(variant) {
        RUN(0,128,128,8,8);
        RUN(1,128,128,16,8);
        RUN(2,64,128,8,4);
        RUN(3,64,128,16,4);
        RUN(4,128,64,16,8);
        default:return -2;
    }
    #undef RUN
    return static_cast<int>(hipGetLastError());
}
