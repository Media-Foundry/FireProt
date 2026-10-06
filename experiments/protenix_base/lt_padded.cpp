// Isolated gfx90a FP32 N1000 triangle contraction with padded input batches.
// Caller owns device memory and workspace. No model dispatch registration.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <cstdint>
static constexpr size_t workspace_size=32*1024*1024;
static void checked(hipblasStatus_t status){if(status!=HIPBLAS_STATUS_SUCCESS)throw std::runtime_error("hipBLASLt status");}
struct State {
 hipblasLtHandle_t handle{};
 hipblasLtMatmulDesc_t desc{};
 hipblasLtMatrixLayout_t input{},output{};
 hipblasLtMatmulPreference_t preference{};
 std::array<hipblasLtMatmulHeuristicResult_t,16> algorithms{};
 int count=0;
 void initialize(int outgoing){
  checked(hipblasLtCreate(&handle));
  checked(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32F,HIP_R_32F));
  hipblasOperation_t first=outgoing?HIPBLAS_OP_T:HIPBLAS_OP_N,second=outgoing?HIPBLAS_OP_N:HIPBLAS_OP_T;
  checked(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&first,sizeof(first)));
  checked(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&second,sizeof(second)));
  checked(hipblasLtMatrixLayoutCreate(&input,HIP_R_32F,1000,1000,1000));
  checked(hipblasLtMatrixLayoutCreate(&output,HIP_R_32F,1000,1000,1000));
  int32_t batch=128;int64_t stride_in=1000*1000+128,stride_out=1000*1000;
  for(auto layout:{input,output})checked(hipblasLtMatrixLayoutSetAttribute(layout,HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)));
  checked(hipblasLtMatrixLayoutSetAttribute(input,HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride_in,sizeof(stride_in)));
  checked(hipblasLtMatrixLayoutSetAttribute(output,HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride_out,sizeof(stride_out)));
  checked(hipblasLtMatmulPreferenceCreate(&preference));size_t bytes=workspace_size;
  checked(hipblasLtMatmulPreferenceSetAttribute(preference,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
  checked(hipblasLtMatmulAlgoGetHeuristic(handle,desc,input,input,output,output,preference,algorithms.size(),algorithms.data(),&count));
 }
 ~State(){if(preference)hipblasLtMatmulPreferenceDestroy(preference);if(input)hipblasLtMatrixLayoutDestroy(input);if(output)hipblasLtMatrixLayoutDestroy(output);if(desc)hipblasLtMatmulDescDestroy(desc);if(handle)hipblasLtDestroy(handle);}
};
static std::mutex mutex;
static std::map<std::pair<int,int>,std::unique_ptr<State>> states;
static State& state(int outgoing){
 if(outgoing!=0&&outgoing!=1)throw std::runtime_error("direction");
 int device=-1;if(hipGetDevice(&device)!=hipSuccess)throw std::runtime_error("device");
 auto& ptr=states[{device,outgoing}];if(!ptr){auto tmp=std::make_unique<State>();tmp->initialize(outgoing);ptr=std::move(tmp);}return *ptr;
}
extern "C" int lt_base_count(int outgoing){try{std::lock_guard<std::mutex> guard(mutex);return state(outgoing).count;}catch(...){return -1;}}
extern "C" int lt_base_bmm(float*a,float*b,float*c,int outgoing,int algorithm,void*stream,void*workspace,size_t bytes){
 for(const void*p:{a,b,c})if(!p||reinterpret_cast<uintptr_t>(p)%16)return 2;
 if(!workspace||bytes<workspace_size)return 2;
 try{
  std::lock_guard<std::mutex> guard(mutex);auto& s=state(outgoing);
  if(algorithm<0||algorithm>=s.count||s.algorithms[algorithm].state!=HIPBLAS_STATUS_SUCCESS)return 2;
  const float alpha=1,beta=0;
  checked(hipblasLtMatmul(s.handle,s.desc,&alpha,b,s.input,a,s.input,&beta,c,s.output,c,s.output,&s.algorithms[algorithm].algo,workspace,bytes,static_cast<hipStream_t>(stream)));
  return hipGetLastError()==hipSuccess?0:3;
 }catch(...){return 3;}
}
