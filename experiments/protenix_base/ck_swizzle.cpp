// Isolated CK scheduling experiment. Generated CK header retains its AMD MIT notice.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "fmha_swizzle.hpp"
#include <tuple>
#include <cstdint>
namespace ct=ck_tile;
using Shape=ct::TileFmhaShape<ct::sequence<128,32,16,32,32,32>,ct::sequence<4,1,1>,ct::sequence<16,16,16>,ct::sequence<4,1,1>,ct::sequence<16,16,16>,true>;
using Traits=ct::TileFmhaTraits<true,true,false,false,false,ct::BlockAttentionBiasEnum::ELEMENTWISE_BIAS,false,false,false,ct::BlockAttentionQuantScaleEnum::NO_SCALE>;
using Variant=ct::ComposedAttention<0,CK_TILE_FMHA_FWD_FAST_EXP2>;
using Problem=ct::BlockFmhaPipelineProblem<float,float,float,float,float,float,uint8_t,float,float,float,float,Shape,false,Variant,ct::SimplifiedGenericAttentionMask<false>,false,Traits>;
using Base=ct::BlockFmhaPipelineQRKSVS<Problem>;
struct Pipeline:Base {static constexpr ct::index_t kAlignmentBias=4;};
using Epilogue=ct::Default2DEpilogue<ct::Default2DEpilogueProblem<float,float,true,false>>;
template<int G> int run(float*q,float*k,float*v,float*b,float*o,void*stream) {
 using Kernel=ct::FireProtSwizzleFmhaFwdKernel<Pipeline,Epilogue,G>;
 constexpr int N=1000;
 auto args=Kernel::MakeKargs(q,k,v,b,nullptr,nullptr,nullptr,nullptr,nullptr,o,
  N,N,32,32,4,1,1.0f,0.0f,128,128,128,N,0,128,0,0,0,
  32,32,32,N*N,0,0,32,0,0,0,N*128,N*128,N*128,0,0,0,N*128,0,0,0,
  -1,-1,0,0,0.0f,false,std::tuple<uint64_t,uint64_t>{0,0},0,0);
 ct::stream_config config{static_cast<hipStream_t>(stream),false};
 auto launch=ct::make_kernel<Kernel::kBlockPerCu>(Kernel{},Kernel::GridSize(1000,4,N,32,false),Kernel::BlockSize(),0,args);
 launch(config);return static_cast<int>(hipGetLastError());
}
extern "C" int fireprot_base_ck_tiles(float*q,float*k,float*v,float*b,float*o,int variant,void*stream) {
 for(const void*p:{q,k,v,b,o})if(!p||reinterpret_cast<uintptr_t>(p)%16)return 3;
 switch(variant) {
  case 0:return run<1>(q,k,v,b,o,stream); // Same logical scheduling as maintained kernel.
  case 1:return run<16>(q,k,v,b,o,stream);
  case 2:return run<64>(q,k,v,b,o,stream);
  case 3:return run<1000>(q,k,v,b,o,stream);
  default:return 2;
 }
}
