// Vectorize both global sides of the existing sigmoid/transpose operation.
#include <hip/hip_runtime.h>
#include <cstdint>
template<int Rows>
__global__ void pack(const float4* gate,const float4* projection,float4* output,int rows) {
    __shared__ float tile[Rows][129];
    int base=blockIdx.x*Rows;
    for(int i=threadIdx.x;i<Rows*32;i+=blockDim.x) {
        int r=i/32,c=(i%32)*4;
        if(base+r<rows) {
            float4 g=gate[(base+r)*32+c/4],p=projection[(base+r)*32+c/4];
            tile[r][c]=(1.0f/(1.0f+expf(-g.x)))*p.x;
            tile[r][c+1]=(1.0f/(1.0f+expf(-g.y)))*p.y;
            tile[r][c+2]=(1.0f/(1.0f+expf(-g.z)))*p.z;
            tile[r][c+3]=(1.0f/(1.0f+expf(-g.w)))*p.w;
        }
    }
    __syncthreads();
    for(int i=threadIdx.x;i<128*(Rows/4);i+=blockDim.x) {
        int c=i/(Rows/4),r=(i%(Rows/4))*4;
        if(base+r<rows) output[(c*rows+base+r)/4]=make_float4(
            tile[r][c],tile[r+1][c],tile[r+2][c],tile[r+3][c]);
    }
}
extern "C" int triangle_pack_vector(float* gate,float* projection,float* output,int n,int tile,void* stream) {
    if((n!=1000 && n!=1536) || (tile!=32 && tile!=64)) return 2;
    if(!gate || !projection || !output || reinterpret_cast<uintptr_t>(gate)%16 ||
       reinterpret_cast<uintptr_t>(projection)%16 || reinterpret_cast<uintptr_t>(output)%16) return 3;
    int rows=n*n;
    if(tile==32) hipLaunchKernelGGL(pack<32>,dim3((rows+31)/32),dim3(256),0,
        static_cast<hipStream_t>(stream),reinterpret_cast<float4*>(gate),reinterpret_cast<float4*>(projection),reinterpret_cast<float4*>(output),rows);
    else hipLaunchKernelGGL(pack<64>,dim3((rows+63)/64),dim3(256),0,
        static_cast<hipStream_t>(stream),reinterpret_cast<float4*>(gate),reinterpret_cast<float4*>(projection),reinterpret_cast<float4*>(output),rows);
    return static_cast<int>(hipGetLastError());
}
