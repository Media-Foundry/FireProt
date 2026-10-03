#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <cstdint>
union Pack { uint4 raw; uint16_t bits[8]; };
__device__ __forceinline__ float decode(uint16_t x) { return __uint_as_float(uint32_t(x)<<16); }
__global__ void mask_bias_kernel(uint4* s,const uint4* bias,int64_t vectors,int64_t span,int n,int offset,int valid) {
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i>=vectors)return;
 Pack a,b;a.raw=s[i];b.raw=bias[i%span];
 int row=int(i/span)+offset;int key=int((i*8)%n);
 bool masked=row>=valid || key>=valid;
 float mask_value=float(c10::BFloat16(-1.e9f));
 #pragma unroll
 for(int j=0;j<8;j++) {
  float value=decode(a.bits[j]);
  if(masked)value=float(c10::BFloat16(value+mask_value));
  a.bits[j]=c10::BFloat16(value+decode(b.bits[j])).x;
 }
 s[i]=a.raw;
}
void mask_bias_(torch::Tensor s,torch::Tensor bias,int64_t offset,int64_t valid) {
 TORCH_CHECK(s.is_cuda()&&bias.is_cuda()&&s.device()==bias.device());
 TORCH_CHECK(s.scalar_type()==at::kBFloat16&&bias.scalar_type()==at::kBFloat16);
 TORCH_CHECK(s.is_contiguous()&&bias.is_contiguous());
 TORCH_CHECK(s.dim()==5&&s.size(0)==1&&s.size(2)==4);
 int64_t n=s.size(3);TORCH_CHECK(n==s.size(4)&&n%8==0&&valid%8==0);
 TORCH_CHECK(valid>=0&&valid<=n&&offset>=0&&offset+s.size(1)<=n);
 TORCH_CHECK(bias.numel()==4*n*n&&s.numel()%8==0);
 TORCH_CHECK(reinterpret_cast<uintptr_t>(s.data_ptr())%16==0&&reinterpret_cast<uintptr_t>(bias.data_ptr())%16==0);
 c10::DeviceGuard guard(s.device());
 int64_t vectors=s.numel()/8;
 hipLaunchKernelGGL(mask_bias_kernel,dim3((vectors+255)/256),dim3(256),0,c10::hip::getCurrentHIPStream(),reinterpret_cast<uint4*>(s.data_ptr()),reinterpret_cast<const uint4*>(bias.data_ptr()),vectors,4*n*n/8,int(n),int(offset),int(valid));
 C10_HIP_CHECK(hipGetLastError());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("mask_bias_",&mask_bias_);}
