// Fuse a triangle gate projection with sigmoid and channel-first output.
// AMD CK headers are supplied separately under their MIT license.
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <exception>
using Row=ck::tensor_layout::gemm::RowMajor;
using Col=ck::tensor_layout::gemm::ColumnMajor;
using Pass=ck::tensor_operation::element_wise::PassThrough;
template<int... V> using S=ck::Sequence<V...>;
#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
struct SigmoidMultiply {
    __host__ __device__ void operator()(float& e,const float& gate,const float& value) const {
        e=(1.0f/(1.0f+expf(-gate)))*value;
    }
};
using Op=ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<
    Row,Col,ck::Tuple<Row>,Row,float,float,float,float,ck::Tuple<float>,float,
    Pass,Pass,SigmoidMultiply,ck::tensor_operation::device::GemmSpecialization::NPadding,
    1,256,TILE_M,TILE_N,32,4,4,32,32,TILE_M/64,TILE_N/64,
    S<8,16,2>,S<1,0,2>,S<1,0,2>,2,2,2,true,
    S<8,16,2>,S<1,0,2>,S<1,0,2>,2,2,2,true,
    1,1,S<1,16,1,16>,4>;
extern "C" int ck_projection_gate(float* weight,float* features,float* value,float* output,
                                   int n,void* stream) {
    if(n!=1000 && n!=1536)return 2;
    if(!weight||!features||!value||!output)return 3;
    try {
        const int rows=n*n;
        // W[128,128] @ X.T[128,rows], D/E row-major [128,rows].
        auto arg=Op::MakeArgument(weight,features,{value},output,128,rows,128,
                                  128,128,{rows},rows,Pass{},Pass{},SigmoidMultiply{});
        if(!Op::IsSupportedArgument(arg))return 4;
        Op::MakeInvoker().Run(arg,StreamConfig{static_cast<hipStream_t>(stream),false});
        return static_cast<int>(hipGetLastError());
    } catch(const std::exception&) {return 5;}
}
