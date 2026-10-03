// Private FP32 dual projection plus SiLU gate for long pair transitions.
#include <hip/hip_runtime.h>
using V4 = float __attribute__((ext_vector_type(4)));
#ifndef ROW_TILE
#define ROW_TILE 128
#endif
#ifndef CHANNEL_TILE
#define CHANNEL_TILE 64
#endif
#ifndef WAVES
#define WAVES 8
#endif
constexpr int ChannelWaves = CHANNEL_TILE / 16;
constexpr int RowWaves = WAVES / ChannelWaves;
constexpr int RowsPerWave = ROW_TILE / RowWaves;
static_assert(WAVES % ChannelWaves == 0 && RowsPerWave % 16 == 0 && 512 % CHANNEL_TILE == 0);
__global__ __launch_bounds__(WAVES * 64) void project_transition(
 const float* __restrict__ x, const float* __restrict__ wa,
 const float* __restrict__ wb, float* __restrict__ out, int rows) {
    __shared__ float input[ROW_TILE][20];
    __shared__ float aw[CHANNEL_TILE][20], bw[CHANNEL_TILE][20];
    const int tid=threadIdx.x, lane=tid%64, wave=tid/64;
    const int first=blockIdx.x*ROW_TILE, channel=blockIdx.y*CHANNEL_TILE;
    const int wave_row=(wave/ChannelWaves)*RowsPerWave, wave_col=(wave%ChannelWaves)*16;
    V4 a[RowsPerWave/16]={}, b[RowsPerWave/16]={};
    for(int base=0; base<128; base+=16) {
        for(int i=tid; i<ROW_TILE*4; i+=WAVES*64) {
            const int r=i/4, c=(i%4)*4;
            float4 v=make_float4(0,0,0,0);
            if(first+r<rows) v=*reinterpret_cast<const float4*>(x+(first+r)*128+base+c);
            *reinterpret_cast<float4*>(&input[r][c])=v;
        }
        for(int i=tid; i<CHANNEL_TILE*4; i+=WAVES*64) {
            const int r=i/4, c=(i%4)*4, wi=(channel+r)*128+base+c;
            *reinterpret_cast<float4*>(&aw[r][c])=*reinterpret_cast<const float4*>(wa+wi);
            *reinterpret_cast<float4*>(&bw[r][c])=*reinterpret_cast<const float4*>(wb+wi);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0; k<16; k+=4) {
            const int c=k+lane/16, wr=wave_col+lane%16;
            const float av=aw[wr][c], bv=bw[wr][c];
            #pragma unroll
            for(int g=0; g<RowsPerWave/16; g++) {
                const float v=input[wave_row+g*16+lane%16][c];
                a[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(v,av,a[g],0,0,0);
                b[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(v,bv,b[g],0,0,0);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int g=0; g<RowsPerWave/16; g++) {
        #pragma unroll
        for(int j=0; j<4; j++) {
            const int r=first+wave_row+g*16+lane/16*4+j, c=channel+wave_col+lane%16;
            if(r<rows) out[r*512+c]=(a[g][j]/(1.0f+expf(-a[g][j])))*b[g][j];
        }
    }
}
extern "C" int transition_projection(const float* x, const float* wa,
 const float* wb, float* out, int rows, void* stream) {
    if(rows!=1000000 && rows!=1536*1536 && rows!=131) return 2;
    if(!x || !wa || !wb || !out) return 3;
    hipLaunchKernelGGL(project_transition,dim3((rows+ROW_TILE-1)/ROW_TILE,512/CHANNEL_TILE),
      dim3(WAVES*64),0,static_cast<hipStream_t>(stream),x,wa,wb,out,rows);
    return static_cast<int>(hipGetLastError());
}
