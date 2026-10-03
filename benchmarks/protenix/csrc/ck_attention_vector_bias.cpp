// CK-derived probe: retain Q/K tail handling and widen aligned bias loads.
// The included implementation and installed CK headers retain their licenses.
#include "../../../fireprot/backends/amd/csrc/fireprot_ck_attention.cpp"
#include <cstdint>

struct VectorBiasPipeline : Pipeline {
    static constexpr ct::index_t kAlignmentBias = 4;
};
static_assert(Pipeline::kPadSeqLenQ && Pipeline::kPadSeqLenK);
static_assert(Pipeline::kAlignmentBias == 1);
using VectorBiasKernel = ct::FmhaFwdKernel<VectorBiasPipeline, Epilogue>;

extern "C" int ck_attention_vector_bias(float* q, float* k, float* v, float* bias,
                                        float* output, int batch_size, int seq_len,
                                        void* raw_stream) {
    if(batch_size != 1000 || seq_len != 1000) return 2;
    for(const void* pointer : {q,k,v,bias,output})
        if(pointer == nullptr || reinterpret_cast<uintptr_t>(pointer) % 16) return 3;
    constexpr int N = 1000; // Row stride and logical K tail both divide four.
    auto args=VectorBiasKernel::MakeKargs(q,k,v,bias,nullptr,nullptr,nullptr,nullptr,nullptr,output,
        N,N,32,32,4,1,1.0f,0.0f,
        128,128,128,N,0,128, 0,0,0,
        32,32,32,N*N,0,0,32, 0,0,0,
        N*128,N*128,N*128,0,0,0,N*128, 0,0,0,
        -1,-1,0,0,0.0f,false,std::tuple<uint64_t,uint64_t>{0,0},0,0);
    ct::stream_config config{static_cast<hipStream_t>(raw_stream),false};
    auto launch=ct::make_kernel<VectorBiasKernel::kBlockPerCu>(VectorBiasKernel{},
        VectorBiasKernel::GridSize(1000,4,N,32,false),VectorBiasKernel::BlockSize(),0,args);
    launch(config);
    return static_cast<int>(hipGetLastError());
}
