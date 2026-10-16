// Diagnostic CK BF16 GEMM variants. Not a qualified production operation.
#include <hip/hip_runtime.h>
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using Pass = ck::tensor_operation::element_wise::PassThrough;
template <ck::index_t... X> using S = ck::Sequence<X...>;

template <int M, int N, int K, int X = 32>
using Gemm = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
    Row, Col, Row, ck::bhalf_t, ck::bhalf_t, ck::bhalf_t, float, ck::bhalf_t,
    Pass, Pass, Pass, ck::tensor_operation::device::GemmSpecialization::MNPadding,
    1, 256, M, N, K, 8, 8, X, X, M / (2 * X), N / (2 * X),
    S<K/8, 2048/K, 1>, S<1,0,2>, S<1,0,2>, 2, 8, 8, (K < 128),
    S<K/8, 2048/K, 1>, S<1,0,2>, S<1,0,2>, 2, 8, 8, (K < 128),
    1, 1, S<1,32,1,8>, X / 4>;

template <class Op>
int run(const void* a, const void* b, void* c, int m, int n, int k, hipStream_t stream) {
    Op op;
    auto arg = op.MakeArgumentPointer(a, b, c, m, n, k, k, k, n, Pass{}, Pass{}, Pass{});
    if (!op.IsSupportedArgument(arg.get())) return 1;
    op.MakeInvokerPointer()->Run(arg.get(), StreamConfig{stream, false});
    return hipGetLastError() == hipSuccess ? 0 : 2;
}

extern "C" int esmc_ck_tile(int variant, const void* a, const void* b, void* c,
                             int m, int n, int k, void* stream) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0) return 3;
    try {
        auto s = static_cast<hipStream_t>(stream);
        switch (variant) {
        case 0: return run<Gemm<128,128,32>>(a,b,c,m,n,k,s);
        case 1: return run<Gemm<128,128,64>>(a,b,c,m,n,k,s);
        case 2: return run<Gemm<128,128,128>>(a,b,c,m,n,k,s);
        case 3: return run<Gemm<256,128,32>>(a,b,c,m,n,k,s);
        case 4: return run<Gemm<128,256,32>>(a,b,c,m,n,k,s);
        case 5: return run<Gemm<128,128,64,16>>(a,b,c,m,n,k,s);
        default: return 3;
        }
    } catch (...) { return 4; }
}
