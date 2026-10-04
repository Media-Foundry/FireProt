#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>
// Match the installed PyTorch persistent softmax's wave64 reduction order.
template<int N,int W>
__global__ void fused(const c10::BFloat16* s,const c10::BFloat16* bias,c10::BFloat16* out,int rows,int offset,int valid) {
 constexpr int K=N==1024?16:32;
 // Nearby waves reuse one bias row across eight outer triangle rows.
 // Preserve each row's arithmetic and use the original order for short tails.
 int linear_row=blockIdx.x*W+threadIdx.y, lane=threadIdx.x;
 int row=rows%(4*N*8)==0 ? (linear_row/(4*N*8))*(4*N*8)+(linear_row%8)*(4*N)+(linear_row/8)%(4*N) : linear_row;
 if(row>=rows)return;
 float values[K];int outer=row/(4*N)+offset;
 int64_t base=int64_t(row)*N,biasbase=int64_t(row%(4*N))*N;
 float mask_value=float(c10::BFloat16(-1.e9f));
 #pragma unroll
 for(int t=0;t<K;t++){
  int j=lane+t*64;float value=-std::numeric_limits<float>::infinity();
  if(j<N){
   value=float(s[base+j]);
   if(outer>=valid||j>=valid)value=float(c10::BFloat16(value+mask_value));
   value=float(c10::BFloat16(value+float(bias[biasbase+j])));
  }
  values[t]=value;
 }
 float maximum=values[0];
 #pragma unroll
 for(int t=0;t<K;t++)maximum=maximum>values[t]?maximum:values[t];
 #pragma unroll
 for(int shift=32;shift>0;shift/=2){float b=__shfl_xor(maximum,shift,64);maximum=maximum<b?b:maximum;}
 float sum=0.f;
 #pragma unroll
 for(int t=0;t<K;t++){values[t]=::expf(values[t]-maximum);sum+=values[t];}
 #pragma unroll
 for(int shift=32;shift>0;shift/=2)sum+=__shfl_xor(sum,shift,64);
 #pragma unroll
 for(int t=0;t<K;t++){int j=lane+t*64;if(j<N)out[base+j]=c10::BFloat16(values[t]/sum);}
}
torch::Tensor mask_softmax(torch::Tensor s,torch::Tensor bias,int offset,int valid,int waves){
 TORCH_CHECK(s.is_cuda()&&bias.is_cuda()&&s.device()==bias.device());
 TORCH_CHECK(s.scalar_type()==at::kBFloat16&&bias.scalar_type()==at::kBFloat16);
 TORCH_CHECK(s.is_contiguous()&&bias.is_contiguous()&&s.dim()==5&&s.size(0)==1&&s.size(2)==4);
 int n=s.size(3);TORCH_CHECK(n==s.size(4)&&(n==1024||n==1536)&&bias.numel()==4L*n*n);
 TORCH_CHECK(offset>=0&&offset+s.size(1)<=n&&valid>=0&&valid<=n);
 TORCH_CHECK(waves==2||waves==4);
 c10::DeviceGuard guard(s.device());auto out=torch::empty_like(s);int rows=s.numel()/n;
 #define LAUNCH(N,W) hipLaunchKernelGGL((fused<N,W>),dim3((rows+W-1)/W),dim3(64,W),0,c10::hip::getCurrentHIPStream(),s.data_ptr<c10::BFloat16>(),bias.data_ptr<c10::BFloat16>(),out.data_ptr<c10::BFloat16>(),rows,offset,valid)
 if(n==1024){if(waves==2){LAUNCH(1024,2);}else{LAUNCH(1024,4);}}
 else{if(waves==2){LAUNCH(1536,2);}else{LAUNCH(1536,4);}}
 C10_HIP_CHECK(hipGetLastError());return out;
}
