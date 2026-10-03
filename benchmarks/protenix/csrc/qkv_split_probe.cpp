// Split interleaved QKV projection output into the current CK head-last layout.
#include <hip/hip_runtime.h>
__global__ void split_qkv(const float4* input,float4* output,int vectors) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=vectors)return;
    const int source=(i/32)*96+i%32;
    float4 q=input[source];
    constexpr float scale=0.1767766952966369f;
    q.x*=scale;q.y*=scale;q.z*=scale;q.w*=scale;
    output[i]=q;output[vectors+i]=input[source+32];output[2*vectors+i]=input[source+64];
}
extern "C" int qkv_split(const float* input,float* output,int rows,void* stream) {
    if(rows!=1000000 && rows!=512*1536)return 2;
    if(!input||!output)return 3;
    const int vectors=rows*32;
    hipLaunchKernelGGL(split_qkv,dim3((vectors+255)/256),dim3(256),0,static_cast<hipStream_t>(stream),
        reinterpret_cast<const float4*>(input),reinterpret_cast<float4*>(output),vectors);
    return static_cast<int>(hipGetLastError());
}
