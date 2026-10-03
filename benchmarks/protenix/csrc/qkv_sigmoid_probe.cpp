// Private QKV projection with sigmoid epilogue over native hipBLAS gate logits.
#include <hip/hip_runtime.h>
using V4=float __attribute__((ext_vector_type(4)));
#ifndef ROW_TILE
#define ROW_TILE 128
#endif
#ifndef CHANNEL_TILE
#define CHANNEL_TILE 64
#endif
#ifndef K_TILE
#define K_TILE 16
#endif
#ifndef WAVES
#define WAVES 8
#endif
constexpr int ChannelWaves=CHANNEL_TILE/16;
constexpr int RowWaves=WAVES/ChannelWaves;
constexpr int RowsPerWave=ROW_TILE/RowWaves;
static_assert(WAVES%ChannelWaves==0 && RowsPerWave%16==0 && 128%CHANNEL_TILE==0);
__global__ __launch_bounds__(WAVES*64) void project_qkv(const float* __restrict__ x,
 const float* __restrict__ wq,const float* __restrict__ wk,const float* __restrict__ wv,
 const float* __restrict__ gate, float* __restrict__ out,int rows) {
    __shared__ float input[ROW_TILE][K_TILE+4];
    __shared__ float qweight[CHANNEL_TILE][K_TILE+4],kweight[CHANNEL_TILE][K_TILE+4],vweight[CHANNEL_TILE][K_TILE+4];
    const int tid=threadIdx.x,lane=tid%64,wave=tid/64;
    const int first=blockIdx.x*ROW_TILE,channel=blockIdx.y*CHANNEL_TILE;
    const int wave_row=(wave/ChannelWaves)*RowsPerWave,wave_col=(wave%ChannelWaves)*16;
    V4 q[RowsPerWave/16]={},k[RowsPerWave/16]={},v[RowsPerWave/16]={};
    for(int base=0;base<128;base+=K_TILE){
        for(int i=tid;i<ROW_TILE*(K_TILE/4);i+=WAVES*64){
            const int r=i/(K_TILE/4),c=(i%(K_TILE/4))*4;
            float4 value=make_float4(0,0,0,0);
            if(first+r<rows)value=*reinterpret_cast<const float4*>(x+(first+r)*128+base+c);
            *reinterpret_cast<float4*>(&input[r][c])=value;
        }
        for(int i=tid;i<CHANNEL_TILE*(K_TILE/4);i+=WAVES*64){
            const int r=i/(K_TILE/4),c=(i%(K_TILE/4))*4,wi=(channel+r)*128+base+c;
            *reinterpret_cast<float4*>(&qweight[r][c])=*reinterpret_cast<const float4*>(wq+wi);
            *reinterpret_cast<float4*>(&kweight[r][c])=*reinterpret_cast<const float4*>(wk+wi);
            *reinterpret_cast<float4*>(&vweight[r][c])=*reinterpret_cast<const float4*>(wv+wi);
        }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<K_TILE;kk+=4){
            const int c=kk+lane/16,wr=wave_col+lane%16;
            const float bq=qweight[wr][c],bk=kweight[wr][c],bv=vweight[wr][c];
            #pragma unroll
            for(int g=0;g<RowsPerWave/16;g++){
                const float a=input[wave_row+g*16+lane%16][c];
                q[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(a,bq,q[g],0,0,0);
                k[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(a,bk,k[g],0,0,0);
                v[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(a,bv,v[g],0,0,0);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int g=0;g<RowsPerWave/16;g++){
        #pragma unroll
        for(int j=0;j<4;j++){
            const int r=first+wave_row+g*16+lane/16*4+j,c=channel+wave_col+lane%16;
            if(r<rows){
                out[r*128+c]=q[g][j]*0.1767766952966369f;
                out[(rows+r)*128+c]=k[g][j];
                out[(2*rows+r)*128+c]=v[g][j];
                out[(3*rows+r)*128+c]=1.0f/(1.0f+expf(-gate[r*128+c]));
            }
        }
    }
}
extern "C" int qkv_sigmoid_projection(const float* x,const float* wq,const float* wk,const float* wv,const float* gate,float* out,int rows,void* stream){
    if(rows!=1000000 && rows!=512*1536)return 2;
    if(!x||!wq||!wk||!wv||!gate||!out)return 3;
    hipLaunchKernelGGL(project_qkv,dim3((rows+ROW_TILE-1)/ROW_TILE,128/CHANNEL_TILE),dim3(WAVES*64),0,static_cast<hipStream_t>(stream),x,wq,wk,wv,gate,out,rows);
    return static_cast<int>(hipGetLastError());
}
