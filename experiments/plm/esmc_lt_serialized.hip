// Experimental version-bound hipBLASLt algorithm serialization and startup timing.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <memory>
#include <chrono>
#include <cstring>
#include <vector>

namespace {
struct Plan {
    hipblasLtHandle_t handle{};
    hipblasLtMatmulDesc_t matmul{};
    hipblasLtMatrixLayout_t a{}, b{}, c{};
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace{};
    int device{};
    double timing[5]{};
    ~Plan() {
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
        if (matmul) hipblasLtMatmulDescDestroy(matmul);
        if (handle) hipblasLtDestroy(handle);
    }
};
}

static int create_impl(int rows, void** result, size_t* workspace, const void* serialized, size_t bytes) {
    if (!result || !workspace || (rows != 1002 && rows != 4008)) return -1;
    *result = nullptr;
    try {
        auto p = std::make_unique<Plan>();
        auto previous = std::chrono::steady_clock::now();
        auto mark = [&](int index) {
            auto now = std::chrono::steady_clock::now();
            p->timing[index] = std::chrono::duration<double>(now - previous).count();
            previous = now;
        };
        if (hipGetDevice(&p->device) != hipSuccess || p->device != 0) return -2;
        if (hipblasLtCreate(&p->handle) != HIPBLAS_STATUS_SUCCESS) return -3;
        mark(0);
        int version = 0;
        if (hipblasLtGetVersion(p->handle, &version) != HIPBLAS_STATUS_SUCCESS || version != 100401) return -4;
        char revision[128]{};
        if (hipblasLtGetGitRevision(p->handle, revision) != HIPBLAS_STATUS_SUCCESS || std::strncmp(revision, "8d1ae90e", 8)) return -4;
        if (hipblasLtMatmulDescCreate(&p->matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) return -5;
        hipblasOperation_t transpose = HIPBLAS_OP_T;
        if (hipblasLtMatmulDescSetAttribute(p->matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                          &transpose, sizeof(transpose)) != HIPBLAS_STATUS_SUCCESS) return -6;
        // Column-major weight^T [N,K] times input^T [K,M] yields output^T [N,M].
        if (hipblasLtMatrixLayoutCreate(&p->a, HIP_R_16BF, 2560, 13824, 2560) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&p->b, HIP_R_16BF, 2560, rows, 2560) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&p->c, HIP_R_16BF, 13824, rows, 13824) != HIPBLAS_STATUS_SUCCESS) return -7;
        mark(1);
        std::vector<int> indices{rows == 1002 ? 4335 : 4139};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        if (serialized) {
            if (bytes != sizeof(p->algo)) return -12;
            std::memcpy(&p->algo, serialized, sizeof(p->algo));
            if (hipblaslt_ext::getIndexFromAlgo(p->algo) != indices[0]) return -13;
        } else {
            if (hipblaslt_ext::getAlgosFromIndex(p->handle, indices, results) != HIPBLAS_STATUS_SUCCESS || results.size() != 1) return -8;
            p->algo = results[0].algo;
        }
        mark(2);
        const float alpha = 1, beta = 0;
        if (hipblaslt_ext::matmulIsAlgoSupported(p->handle, p->matmul, &alpha, p->a, p->b,
                                                &beta, p->c, p->c, p->algo, p->workspace) != HIPBLAS_STATUS_SUCCESS) return -9;
        mark(3);
        if (p->workspace > 32 * 1024 * 1024) return -10;
        *workspace = p->workspace;
        mark(4);
        *result = p.release();
        return 0;
    } catch (...) { return -11; }
}

extern "C" int esmc_lt_run(void* plan, const void* input, const void* weight,
                           void* output, void* workspace, size_t workspace_bytes, void* stream) {
    if (!plan || !input || !weight || !output) return -1;
    auto* p = static_cast<Plan*>(plan);
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device != p->device) return -2;
    if (workspace_bytes < p->workspace || (p->workspace && !workspace)) return -3;
    const float alpha = 1, beta = 0;
    return static_cast<int>(hipblasLtMatmul(p->handle, p->matmul, &alpha, weight, p->a,
        input, p->b, &beta, output, p->c, output, p->c, &p->algo,
        workspace, workspace_bytes, static_cast<hipStream_t>(stream)));
}

extern "C" void esmc_lt_destroy(void* plan) { delete static_cast<Plan*>(plan); }

extern "C" int esmc_lt_create(int rows, void** result, size_t* workspace) {
    return create_impl(rows, result, workspace, nullptr, 0);
}

extern "C" int esmc_lt_create_serialized(int rows, void** result, size_t* workspace,
                                         const void* bytes, size_t size) {
    if (!bytes || size != sizeof(hipblasLtMatmulAlgo_t)) return -12;
    return create_impl(rows, result, workspace, bytes, size);
}

extern "C" int esmc_lt_export(void* plan, void* bytes, size_t size, double* timing) {
    if (!plan || !bytes || size != sizeof(hipblasLtMatmulAlgo_t) || !timing) return -1;
    auto* p = static_cast<Plan*>(plan);
    std::memcpy(bytes, &p->algo, size);
    std::memcpy(timing, p->timing, sizeof(p->timing));
    return 0;
}

extern "C" size_t esmc_lt_algo_size() { return sizeof(hipblasLtMatmulAlgo_t); }
