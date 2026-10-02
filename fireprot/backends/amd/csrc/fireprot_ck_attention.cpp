// FP32 CK Tile attention candidate. AMD CK headers are MIT licensed.
// Exact observed N=1000/B=1000 or N=1536/B=512, H=4, D=32; no explicit mask.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include <tuple>
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
    auto args=Kernel::MakeKargs(q,k,v,b,nullptr,nullptr,nullptr,nullptr,nullptr,o,
        n,n,32,32,4,1,1.0f,0.0f,
        128,128,128,n,0,128, 0,0,0,
        32,32,32,n*n,0,0,32, 0,0,0,
        n*128,n*128,n*128,0,0,0,n*128, 0,0,0,
        -1,-1,0,0,0.0f,false,std::tuple<uint64_t,uint64_t>{0,0},0,0);
    ct::stream_config config{static_cast<hipStream_t>(stream),false};
    auto launch=ct::make_kernel<Kernel::kBlockPerCu>(Kernel{},Kernel::GridSize(batch,4,n,32,false),Kernel::BlockSize(),0,args);
    launch(config);
    return static_cast<int>(hipGetLastError());
}
