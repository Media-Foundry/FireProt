// FP32 CK Tile attention candidate. AMD CK headers are MIT licensed.
// Exact observed N=1000/B=1000 or N=1536/B=512, H=4, D=32; no explicit mask.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include <tuple>
#include <cstdint>
namespace ct = ck_tile;
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 64
#endif
#ifndef WM
#define WM 4
#endif
using Shape = ct::TileFmhaShape<ct::sequence<BM,BN,16,32,BN,32>,
    ct::sequence<WM,1,1>,ct::sequence<16,16,16>,
    ct::sequence<WM,1,1>,ct::sequence<16,16,16>,true>;
using Traits = ct::TileFmhaTraits<true,true,false,false,false,
    ct::BlockAttentionBiasEnum::ELEMENTWISE_BIAS,false,false,false,
    ct::BlockAttentionQuantScaleEnum::NO_SCALE>;
using Variant = ct::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>;
using Problem = ct::BlockFmhaPipelineProblem<float,float,float,float,float,float,
    uint8_t,float,float,float,float,Shape,false,Variant,
    ct::SimplifiedGenericAttentionMask<false>,false,Traits>;
using Pipeline = ct::BlockFmhaPipelineQRKSVS<Problem>;
using Epilogue = ct::Default2DEpilogue<ct::Default2DEpilogueProblem<float,float,true,false>>;
using Kernel = ct::FmhaFwdKernel<Pipeline,Epilogue>;
extern "C" int ck_attention(float* q,float* k,float* v,float* b,float* o,int batch,int n,void* stream) {
    if(!((batch==1000 && n==1000) || (batch==512 && n==1536))) return 2;
    // Must match Python's N1536 padded bias allocation; padding is not a token.
    const int bias_stride = n == 1536 ? n + 16 : n;
    auto args=Kernel::MakeKargs(q,k,v,b,nullptr,nullptr,nullptr,nullptr,nullptr,o,
        n,n,32,32,4,1,1.0f,0.0f,
        128,128,128,bias_stride,0,128, 0,0,0,
        32,32,32,n*bias_stride,0,0,32, 0,0,0,
        n*128,n*128,n*128,0,0,0,n*128, 0,0,0,
        -1,-1,0,0,0.0f,false,std::tuple<uint64_t,uint64_t>{0,0},0,0);
    ct::stream_config config{static_cast<hipStream_t>(stream),false};
    auto launch=ct::make_kernel<Kernel::kBlockPerCu>(Kernel{},Kernel::GridSize(batch,4,n,32,false),Kernel::BlockSize(),0,args);
    launch(config);
    return static_cast<int>(hipGetLastError());
}

// Separate exact-shape kernel: N1536 is divisible by both sequence tiles.
// Keep the N1000 instantiation above padded and unchanged.
namespace exact1536 {
using ExactShape = ct::TileFmhaShape<ct::sequence<128,32,16,32,32,32>,
    ct::sequence<4,1,1>,ct::sequence<16,16,16>,
    ct::sequence<4,1,1>,ct::sequence<16,16,16>,true>;
using ExactTraits = ct::TileFmhaTraits<false,false,false,false,false,
    ct::BlockAttentionBiasEnum::ELEMENTWISE_BIAS,false,false,false,
    ct::BlockAttentionQuantScaleEnum::NO_SCALE>;
using ExactProblem = ct::BlockFmhaPipelineProblem<float,float,float,float,float,float,
    uint8_t,float,float,float,float,ExactShape,false,Variant,
    ct::SimplifiedGenericAttentionMask<false>,false,ExactTraits>;
using ExactPipeline = ct::BlockFmhaPipelineQRKSVS<ExactProblem>;
using ExactEpilogue = ct::Default2DEpilogue<ct::Default2DEpilogueProblem<float,float,false,false>>;
using ExactKernel = ct::FmhaFwdKernel<ExactPipeline,ExactEpilogue>;
}
extern "C" int ck_attention_exact1536(float* q,float* k,float* v,float* bias,float* output,
                                     int batch_size,int seq_len,void* raw_stream) {
    if(batch_size != 512 || seq_len != 1536) return 2;
    using exact1536::ExactKernel;
    constexpr int N = 1536;
    constexpr int S = 1552; // physical bias row padding only, never extra tokens
    auto args=ExactKernel::MakeKargs(q,k,v,bias,nullptr,nullptr,nullptr,nullptr,nullptr,output,
        N,N,32,32,4,1,1.0f,0.0f,
        128,128,128,S,0,128, 0,0,0,
        32,32,32,N*S,0,0,32, 0,0,0,
        N*128,N*128,N*128,0,0,0,N*128, 0,0,0,
        -1,-1,0,0,0.0f,false,std::tuple<uint64_t,uint64_t>{0,0},0,0);
    ct::stream_config config{static_cast<hipStream_t>(raw_stream),false};
    auto launch=ct::make_kernel<ExactKernel::kBlockPerCu>(ExactKernel{},
        ExactKernel::GridSize(512,4,N,32,false),ExactKernel::BlockSize(),0,args);
    launch(config);
    return static_cast<int>(hipGetLastError());
}

// N1000 keeps tail masking but permits aligned four-element bias loads.
namespace vector1000 {
using VectorShape = ct::TileFmhaShape<ct::sequence<128,32,16,32,32,32>,
    ct::sequence<4,1,1>,ct::sequence<16,16,16>,
    ct::sequence<4,1,1>,ct::sequence<16,16,16>,true>;
using VectorProblem = ct::BlockFmhaPipelineProblem<float,float,float,float,float,float,
    uint8_t,float,float,float,float,VectorShape,false,Variant,
    ct::SimplifiedGenericAttentionMask<false>,false,Traits>;
using PaddedPipeline = ct::BlockFmhaPipelineQRKSVS<VectorProblem>;
struct VectorBiasPipeline : PaddedPipeline {
    static constexpr ct::index_t kAlignmentBias = 4;
};
static_assert(PaddedPipeline::kPadSeqLenQ && PaddedPipeline::kPadSeqLenK);
static_assert(PaddedPipeline::kAlignmentBias == 1);
using VectorBiasKernel = ct::FmhaFwdKernel<VectorBiasPipeline, Epilogue>;

}
extern "C" int ck_attention_vector_bias1000(float* q, float* k, float* v, float* bias,
                                        float* output, int batch_size, int seq_len,
                                        void* raw_stream) {
    using vector1000::VectorBiasKernel;
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
