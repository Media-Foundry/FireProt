// Experimental FP32 CK attention, adapted from AMD CK example 47.
// The installed CK headers retain their own MIT license and version metadata.
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include <array>
#include <vector>
template <ck::index_t... I> using S = ck::Sequence<I...>;
using Pass = ck::tensor_operation::element_wise::PassThrough;
using Add = ck::tensor_operation::element_wise::ScaleAdd;
#ifndef MB
#define MB 64
#endif
#ifndef NB
#define NB 64
#endif
#ifdef FULL_G1K
#define G1K NB
#else
#define G1K 32
#endif
#define BNT ((NB < 64) ? NB : 64)
using Op = ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<
    2,1,1,1,1,float,float,float,float,ck::Tuple<float>,ck::Tuple<>,float,float,
    Pass,Pass,Add,Pass,Pass,
    ck::tensor_operation::device::GemmSpecialization::MNKOPadding,
    ck::tensor_operation::device::TensorSpecialization::Default,
    ck::tensor_operation::device::TensorSpecialization::Default,
    ck::tensor_operation::device::TensorSpecialization::Default,
    ck::tensor_operation::device::TensorSpecialization::Default,
#ifdef MF32
    1,MB*2,MB,NB,32,32,32,4,4,1,32,32,1,NB/32,1,
    S<4,MB/2,1>,S<1,0,2>,S<1,0,2>,2,4,4,true,
    S<4,MB/2,1>,S<1,0,2>,S<1,0,2>,2,4,4,true,
    S<MB/4,8,1>,S<0,2,1>,S<0,2,1>,1,4,1,false,
    1,1,S<1,MB/4,1,8>,4,
#else
    1,256,MB,NB,32,32,G1K,4,4,1,16,16,MB/64,NB/16,2,
    S<4,64,1>,S<1,0,2>,S<1,0,2>,2,4,4,true,
    S<256/BNT,BNT,1>,S<1,0,2>,S<1,0,2>,2,4,4,true,
    S<32,8,1>,S<0,2,1>,S<0,2,1>,1,4,1,false,
    1,2,S<1,32,1,8>,4,
#endif
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled,1>;

extern "C" int ck_attention(float* q,float* k,float* v,float* b,float* o,int batch,int n,void* stream) {
    // This C ABI is private to the benchmark; no arbitrary shapes are accepted.
    if(batch != 1000 || n != 1000) return 2;
    Op op;
    auto arg=op.MakeArgument(q,k,v,o,std::array<void*,1>{b},{},
        {batch,4,n,32},{n*128,32,128,1},
        {batch,4,n,32},{n*128,32,128,1},
        {batch,4,32,n},{n*128,32,1,128},
        {batch,4,n,32},{n*128,32,128,1},
        std::array<std::vector<ck::index_t>,1>{{{batch,4,n,n}}},
        std::array<std::vector<ck::index_t>,1>{{{0,n*n,n,1}}},
        {},{},Pass{},Pass{},Add{1.0f},Pass{},Pass{});
    if(!op.IsSupportedArgument(arg)) return 1;
    op.MakeInvoker().Run(arg,StreamConfig{static_cast<hipStream_t>(stream),false});
    return static_cast<int>(hipGetLastError());
}
