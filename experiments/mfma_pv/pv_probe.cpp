// Bounded PV accumulation-order diagnostic, not an inference kernel.
#include <hip/hip_runtime.h>
#include <cstdint>
using acc4 = float __attribute__((ext_vector_type(4)));
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx90a__)
#error This experiment requires gfx90a.
#endif
template<int Order>
__global__ __launch_bounds__(64)
void pv_probe(const float* probability,const float* value,float* output) {
#if defined(__gfx90a__)
 constexpr int N=1536,D=32;
 const int lane=threadIdx.x,coordinate=lane&15,klane=lane>>4;
 const int qbase=blockIdx.y*16,dbase=blockIdx.x*16;
 const uint64_t batch=blockIdx.z;
 acc4 accumulator={0.f,0.f,0.f,0.f};
 #pragma unroll 1
 for(int fragment=0;fragment<N/4;++fragment) {
  int k;
  if constexpr(Order==0) k=fragment*4+klane;
  else if constexpr(Order==1) k=(fragment/2)*8+2*klane+(fragment%2);
  else k=(fragment/2)*8+2*klane+(1-fragment%2);
  const float a=value[(batch*N+k)*D+dbase+coordinate];
  const float b=probability[(batch*N+qbase+coordinate)*N+k];
  accumulator=__builtin_amdgcn_mfma_f32_16x16x4f32(a,b,accumulator,0,0,0);
 }
 #pragma unroll
 for(int e=0;e<4;++e)
  output[(batch*N+qbase+coordinate)*D+dbase+4*klane+e]=accumulator[e];
#endif
}
extern "C" int fireprot_pv_order(const float* probability,const float* value,float* output,
 int batches,int query_tiles,int order,hipStream_t stream) {
 if(!probability||!value||!output||batches<1||batches>512||(query_tiles!=1&&query_tiles!=96)||order<0||order>2)return int(hipErrorInvalidValue);
 #define RUN(O) hipLaunchKernelGGL((pv_probe<O>),dim3(2,query_tiles,batches),dim3(64),0,stream,probability,value,output)
 if(order==0){RUN(0);}else if(order==1){RUN(1);}else{RUN(2);}
 #undef RUN
 return int(hipGetLastError());
}
