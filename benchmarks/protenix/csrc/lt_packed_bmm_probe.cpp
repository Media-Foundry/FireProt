// Standalone, single-device and single-thread FP32 hipBLASLt BMM probe.
// This diagnostic owns its workspace outside the PyTorch allocator.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include <memory>
#include <stdexcept>

static void check(hipblasStatus_t s) {
    if(s != HIPBLAS_STATUS_SUCCESS) throw std::runtime_error("hipBLASLt failure");
}
struct State {
    hipblasLtHandle_t handle{};
    hipblasLtMatmulDesc_t desc{};
    hipblasLtMatrixLayout_t layout{};
    hipblasLtMatmulPreference_t preference{};
    hipblasLtMatmulHeuristicResult_t algorithm{};
    void* workspace{};
    static constexpr size_t workspace_bytes = 32 * 1024 * 1024;
    void initialize(bool outgoing) {
        check(hipblasLtCreate(&handle));
        check(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32F,HIP_R_32F));
        // Column-major C^T uses reversed physical operands b, a.
        hipblasOperation_t first=outgoing ? HIPBLAS_OP_T:HIPBLAS_OP_N;
        hipblasOperation_t second=outgoing ? HIPBLAS_OP_N:HIPBLAS_OP_T;
        check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&first,sizeof(first)));
        check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&second,sizeof(second)));
        check(hipblasLtMatrixLayoutCreate(&layout,HIP_R_32F,1536,1536,1536));
        int32_t batches=128; int64_t stride=1536*1536;
        check(hipblasLtMatrixLayoutSetAttribute(layout,HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batches,sizeof(batches)));
        check(hipblasLtMatrixLayoutSetAttribute(layout,HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride,sizeof(stride)));
        check(hipblasLtMatmulPreferenceCreate(&preference));
        size_t bytes=workspace_bytes;
        check(hipblasLtMatmulPreferenceSetAttribute(preference,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
        int count=0;
        check(hipblasLtMatmulAlgoGetHeuristic(handle,desc,layout,layout,layout,layout,preference,1,&algorithm,&count));
        if(count!=1 || algorithm.state!=HIPBLAS_STATUS_SUCCESS) throw std::runtime_error("No algorithm");
        if(hipMalloc(&workspace,workspace_bytes)!=hipSuccess) throw std::runtime_error("Workspace allocation failed");
    }
    ~State() {
        if(workspace) (void)hipFree(workspace);
        if(preference) (void)hipblasLtMatmulPreferenceDestroy(preference);
        if(layout) (void)hipblasLtMatrixLayoutDestroy(layout);
        if(desc) (void)hipblasLtMatmulDescDestroy(desc);
        if(handle) (void)hipblasLtDestroy(handle);
    }
};
extern "C" int lt_packed_bmm(float* a,float* b,float* c,int batch,int n,int outgoing,void* stream) {
    if(batch!=128 || n!=1536 || (outgoing!=0 && outgoing!=1)) return 2;
    try {
        static std::unique_ptr<State> states[2];
        auto& state=states[outgoing];
        if(!state) { auto candidate=std::make_unique<State>(); candidate->initialize(outgoing); state=std::move(candidate); }
        const float alpha=1.0f,beta=0.0f;
        check(hipblasLtMatmul(state->handle,state->desc,&alpha,b,state->layout,a,state->layout,
                             &beta,c,state->layout,c,state->layout,&state->algorithm.algo,
                             state->workspace,State::workspace_bytes,static_cast<hipStream_t>(stream)));
        return static_cast<int>(hipGetLastError());
    } catch(const std::exception&) { return 3; }
}
