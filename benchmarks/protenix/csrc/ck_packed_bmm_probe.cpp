// Direct FP32 CK BMM probe for N1536 packed triangle contractions.
// CK headers are provided by AMD under the MIT license.
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include <exception>

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using Pass = ck::tensor_operation::element_wise::PassThrough;
template<int... Values> using S = ck::Sequence<Values...>;
#ifndef XDL
#define XDL 32
#endif
#ifndef TILE
#define TILE 128
#endif
#ifndef REP
#define REP (TILE / (2 * XDL))
#endif
#ifndef FIREPROT_K0
#define FIREPROT_K0 8
#endif
#ifndef PREFETCH
#define PREFETCH 1
#endif

template<bool Outgoing>
using Op = ck::tensor_operation::device::DeviceBatchedGemmXdl<
    float, float, float, float,
    std::conditional_t<Outgoing, Row, Col>,
    std::conditional_t<Outgoing, Col, Row>, Row, Pass, Pass, Pass,
    256, TILE, TILE, FIREPROT_K0, 4, XDL, XDL, REP, REP,
    std::conditional_t<Outgoing, S<8,16,2>, S<(TILE == 64 ? 8 : 4),(TILE == 64 ? 32 : 64),1>>,
    std::conditional_t<Outgoing, S<1,0,2>, S<0,2,1>>,
    std::conditional_t<Outgoing, S<1,0,2>, S<0,2,1>>,
    Outgoing ? 2 : 1, 2, Outgoing ? 2 : 4, true,
    std::conditional_t<Outgoing, S<8,16,2>, S<(TILE == 64 ? 8 : 4),(TILE == 64 ? 32 : 64),1>>,
    std::conditional_t<Outgoing, S<1,0,2>, S<0,2,1>>,
    std::conditional_t<Outgoing, S<1,0,2>, S<0,2,1>>,
    Outgoing ? 2 : 1, 2, Outgoing ? 2 : 4, true, 7, 1, PREFETCH>;

template<bool Outgoing>
int launch(float* a, float* b, float* c, void* stream) {
    constexpr int n = 1536;
    auto argument = Op<Outgoing>::MakeArgument(a, b, c, n, n, n,
                                              n, n, n, n*n, n*n, n*n, 128);
    if (!Op<Outgoing>::IsSupportedArgument(argument)) return 2;
    Op<Outgoing>::MakeInvoker().Run(argument, StreamConfig{static_cast<hipStream_t>(stream), false});
    return static_cast<int>(hipGetLastError());
}

extern "C" int ck_packed_bmm(float* a, float* b, float* c,
                              int batch, int n, int outgoing, void* stream) {
    if (batch != 128 || n != 1536 || (outgoing != 0 && outgoing != 1)) return 2;
    try {
        return outgoing ? launch<true>(a, b, c, stream) : launch<false>(a, b, c, stream);
    } catch (const std::exception&) {
        return 3;
    }
}
