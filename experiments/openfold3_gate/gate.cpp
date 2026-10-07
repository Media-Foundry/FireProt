// FireProt experimental FP32 sigmoid/projection/mask fusion for gfx90a.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
__global__ void gate_kernel(const float4* gate,const float4* projection,
 const float* mask,float4* output,int64_t vectors,int cols,int64_t sr,int64_t sc) {
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i>=vectors)return;
 int64_t pair=i/32;
 float m=mask[(pair/cols)*sr+(pair%cols)*sc];
 float4 g=gate[i],p=projection[i],v;
 float* gp=reinterpret_cast<float*>(&g);float* pp=reinterpret_cast<float*>(&p);
 float* vp=reinterpret_cast<float*>(&v);
 #pragma unroll
 for(int k=0;k<4;++k){float s=1.0f/(1.0f+expf(-gp[k]));float product=s*pp[k];vp[k]=product*m;}
 output[i]=v;
}
extern "C" int fireprot_gate(const float* g,const float* p,const float* m,float* out,
 int rows,int cols,int64_t sr,int64_t sc,hipStream_t stream){
 if(!g||!p||!m||!out||rows<1||cols<1||rows>1536||cols>1536||sr<0||sc<0)return -1;
 if((reinterpret_cast<uintptr_t>(g)|reinterpret_cast<uintptr_t>(p)|reinterpret_cast<uintptr_t>(out))&15)return -2;
 int64_t vectors=int64_t(rows)*cols*32;
 hipLaunchKernelGGL(gate_kernel,dim3((vectors+255)/256),dim3(256),0,stream,
 reinterpret_cast<const float4*>(g),reinterpret_cast<const float4*>(p),m,
 reinterpret_cast<float4*>(out),vectors,cols,sr,sc);
 return int(hipGetLastError());
}
__global__ void gate_pack_kernel(const float4* gate,const float4* projection,
 const float* mask,float* output,int64_t pairs,int cols,int64_t sr,int64_t sc){
 __shared__ float tile[32][132];
 int64_t base=int64_t(blockIdx.x)*32;
 #pragma unroll
 for(int j=0;j<4;++j){
  int v=threadIdx.x+j*256,r=v/32,c=(v%32)*4;int64_t pair=base+r;
  if(pair<pairs){
   float4 g=gate[pair*32+c/4],p=projection[pair*32+c/4];
   float m=mask[(pair/cols)*sr+(pair%cols)*sc];
   float* gp=reinterpret_cast<float*>(&g);float* pp=reinterpret_cast<float*>(&p);
   #pragma unroll
   for(int k=0;k<4;++k){float s=1.0f/(1.0f+expf(-gp[k]));float product=s*pp[k];tile[r][c+k]=product*m;}
  }
 }
 __syncthreads();
 #pragma unroll
 for(int j=0;j<4;++j){
  int v=threadIdx.x+j*256,c=v/8,r=(v%8)*4;
  #pragma unroll
  for(int k=0;k<4;++k)if(base+r+k<pairs)output[int64_t(c)*pairs+base+r+k]=tile[r+k][c];
 }
}
extern "C" int fireprot_gate_pack(const float* g,const float* p,const float* m,float* out,
 int rows,int cols,int64_t sr,int64_t sc,hipStream_t stream){
 if(!g||!p||!m||!out||rows<1||cols<1||rows>1536||cols>1536||sr<0||sc<0)return -1;
 if((reinterpret_cast<uintptr_t>(g)|reinterpret_cast<uintptr_t>(p))&15)return -2;
 int64_t pairs=int64_t(rows)*cols;
 hipLaunchKernelGGL(gate_pack_kernel,dim3((pairs+31)/32),dim3(256),0,stream,
 reinterpret_cast<const float4*>(g),reinterpret_cast<const float4*>(p),m,out,pairs,cols,sr,sc);
 return int(hipGetLastError());
}
