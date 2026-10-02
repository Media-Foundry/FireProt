// Explicit CK CShuffle probe for the FP32 N1536 triangle contraction.
// AMD CK headers are supplied separately under the MIT license.
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include <exception>

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using Pass = ck::tensor_operation::element_wise::PassThrough;
template<int... V> using S = ck::Sequence<V...>;
#ifndef FIREPROT_TILE
#define FIREPROT_TILE 64
#endif
#ifndef FIREPROT_PIPELINE
#define FIREPROT_PIPELINE 1
#endif

template<bool Out>
using Op = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl_CShuffle_V3<
    std::conditional_t<Out, Row, Col>, std::conditional_t<Out, Col, Row>,
    ck::Tuple<>, Row, float, float, ck::Tuple<>, float, float, float,
    Pass, Pass, Pass, ck::tensor_operation::device::GemmSpecialization::Default,
    256, FIREPROT_TILE, FIREPROT_TILE, 32, 4, 4, 32, 32,
    FIREPROT_TILE/64, FIREPROT_TILE/64,
    std::conditional_t<Out,S<8,16,2>,S<(FIREPROT_TILE==64 ? 8:4),(FIREPROT_TILE==64 ? 32:64),1>>,
    std::conditional_t<Out,S<1,0,2>,S<0,2,1>>,
    std::conditional_t<Out,S<1,0,2>,S<0,2,1>>,
    Out ? 2:1, 2, Out ? 2:4, true,
    std::conditional_t<Out,S<8,16,2>,S<(FIREPROT_TILE==64 ? 8:4),(FIREPROT_TILE==64 ? 32:64),1>>,
    std::conditional_t<Out,S<1,0,2>,S<0,2,1>>,
    std::conditional_t<Out,S<1,0,2>,S<0,2,1>>,
    Out ? 2:1, 2, Out ? 2:4, true,
    1, 1, S<1,16,1,16>, S<4>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    static_cast<ck::BlockGemmPipelineVersion>(FIREPROT_PIPELINE - 1)>;

template<bool Out>
int launch(float* a, float* b, float* c, void* stream) {
    constexpr int n=1536;
    auto arg = Op<Out>::MakeArgument(a,b,{},c,n,n,n,128,n,n,{},n,n*n,n*n,{},n*n,
                                     Pass{},Pass{},Pass{},1);
    if (!Op<Out>::IsSupportedArgument(arg)) return 2;
    Op<Out>::MakeInvoker().Run(arg,StreamConfig{static_cast<hipStream_t>(stream),false});
    return static_cast<int>(hipGetLastError());
}
extern "C" int ck_packed_bmm(float* a,float* b,float* c,int batch,int n,int outgoing,void* stream) {
    if(batch!=128 || n!=1536 || (outgoing!=0 && outgoing!=1)) return 2;
    try { return outgoing ? launch<true>(a,b,c,stream) : launch<false>(a,b,c,stream); }
    catch(const std::exception&) { return 3; }
}
