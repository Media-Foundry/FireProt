// Bit-preserving Q/V packing and K transpose; Q scaling is performed upstream.
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
__global__ void pack_qkv_kernel(const uint16_t* x,uint16_t* q,uint16_t* k,uint16_t* v,int n){
 __shared__ uint16_t tile[32][33];
 int outer=blockIdx.y/4,head=blockIdx.y%4,start=blockIdx.x*32;
 #pragma unroll
 for(int p=threadIdx.x;p<1024;p+=256){
  int row=p/32,col=p%32,j=start+row;
  int64_t src=(int64_t(outer)*n+j)*384+head*32+col;
  int64_t dst=((int64_t(outer)*4+head)*n+j)*32+col;
  q[dst]=x[src];v[dst]=x[src+256];tile[row][col]=x[src+128];
 }
 __syncthreads();
 #pragma unroll
 for(int p=threadIdx.x;p<1024;p+=256){
  int col=p/32,row=p%32,j=start+row;
  k[((int64_t(outer)*4+head)*32+col)*n+j]=tile[row][col];
 }
}
std::vector<torch::Tensor> pack_qkv(torch::Tensor x){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&x.dim()==4&&x.size(0)==1&&x.size(3)==384);
 int n=x.size(1);TORCH_CHECK((n==1024||n==1536)&&x.size(2)==n);
 c10::DeviceGuard guard(x.device());
 auto q=torch::empty({1,n,4,n,32},x.options());auto v=torch::empty_like(q);auto k=torch::empty({1,n,4,32,n},x.options());
 hipLaunchKernelGGL(pack_qkv_kernel,dim3(n/32,n*4),dim3(256),0,c10::hip::getCurrentHIPStream(),reinterpret_cast<const uint16_t*>(x.data_ptr()),reinterpret_cast<uint16_t*>(q.data_ptr()),reinterpret_cast<uint16_t*>(k.data_ptr()),reinterpret_cast<uint16_t*>(v.data_ptr()),n);
 C10_HIP_CHECK(hipGetLastError());return {q,k.transpose(-1,-2),v};
}
