#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <cstdint>
__global__ void transpose_kernel(const uint16_t* in,uint16_t* out,int rows,int cols,int tiles_cols){
 __shared__ uint16_t tile[32][33];
 int bx=blockIdx.x%tiles_cols,by=blockIdx.x/tiles_cols;
 #pragma unroll
 for(int p=threadIdx.x;p<1024;p+=256){int r=by*32+p/32,c=bx*32+p%32;if(r<rows&&c<cols)tile[p/32][p%32]=in[int64_t(r)*cols+c];}
 __syncthreads();
 #pragma unroll
 for(int p=threadIdx.x;p<1024;p+=256){int r=by*32+p%32,c=bx*32+p/32;if(r<rows&&c<cols)out[int64_t(c)*rows+r]=tile[p%32][p/32];}
}
torch::Tensor transpose_bf16(torch::Tensor x){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&x.dim()==2);
 TORCH_CHECK(x.size(0)>0&&x.size(1)>0&&x.size(0)<=1536*1536&&x.size(1)<=1536*1536);
 c10::DeviceGuard guard(x.device());int rows=x.size(0),cols=x.size(1),tc=(cols+31)/32;
 auto y=torch::empty({cols,rows},x.options());
 hipLaunchKernelGGL(transpose_kernel,dim3(int64_t((rows+31)/32)*tc),dim3(256),0,c10::hip::getCurrentHIPStream(),reinterpret_cast<const uint16_t*>(x.data_ptr()),reinterpret_cast<uint16_t*>(y.data_ptr()),rows,cols,tc);
 C10_HIP_CHECK(hipGetLastError());return y;
}
