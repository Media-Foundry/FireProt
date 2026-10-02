// CPU coordinate dump for measured CK FP32 MFMA layouts. No GPU kernels are launched.
// Define BM, WM, G0M/G0K and G1M/G1K as documented in protenix-long1536.md.
// FP32 CK Tile attention candidate. AMD CK headers are MIT licensed.
// Exact observed N=1000/B=1000 or N=1536/B=512, H=4, D=32; no explicit mask.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include <tuple>
namespace ck_tile::impl::warp_gemm_dispatcher {
template<> struct Dispatcher<float,float,float,32,32,8,true,false> {
 using Type = WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
  WarpGemmAttributeMfmaImplF32F32F32M32N32K2<WGAttrCtlEnum::Default_>,4>>;
};
template<> struct Dispatcher<float,float,float,32,32,8,true,true> {
 using Type = WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
  WarpGemmAttributeMfmaImplF32F32F32M32N32K2<WGAttrCtlEnum::Default_>,4>>;
};
}
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
static_assert(BN == 64, "Coordinate encoding requires BN=64");
using Shape = ct::TileFmhaShape<ct::sequence<BM,BN,16,32,BN,32>,
    ct::sequence<WM,1,1>,ct::sequence<G0M,G0M,G0K>,
    ct::sequence<WM,1,1>,ct::sequence<G1M,G1M,G1K>,true>;
using Traits = ct::TileFmhaTraits<true,true,false,false,false,
    ct::BlockAttentionBiasEnum::ELEMENTWISE_BIAS,false,false,false,
    ct::BlockAttentionQuantScaleEnum::NO_SCALE>;
using Variant = ct::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>;
using Problem = ct::BlockFmhaPipelineProblem<float,float,float,float,float,float,
    uint8_t,float,float,float,float,Shape,false,Variant,
    ct::SimplifiedGenericAttentionMask<false>,false,Traits>;
using Pipeline = ct::BlockFmhaPipelineQRKSVS<Problem>;
using Epilogue = ct::Default2DEpilogue<ct::Default2DEpilogueProblem<float,float,true,false>>;

#include <cstdio>
struct MatchingShape : Shape { using Gemm0WarpTile = typename Shape::Gemm1WarpTile; };
struct MatchingProblem : Problem { using BlockFmhaShape = MatchingShape; };
// Host equivalent of AMD CK sweep_tile_span (MIT).
template<class Span,class F> void host_sweep(Span,const F& f) {
 ct::static_ford<typename ct::remove_cvref_t<Span>::Impl>{}([&](auto i){f(ct::detail::make_tile_distributed_index(i));});
}
template<class T> void dump() {
 for(int t=0;t<WM*64;++t) {
  T tensor{};
  constexpr auto spans=T::get_distributed_spans();
  host_sweep(spans[ct::number<0>{}], [&](auto r) {
   host_sweep(spans[ct::number<1>{}], [&](auto c) {
    auto index=ct::make_tuple(r,c);
    auto xy=ct::get_x_indices_from_distributed_indices(T::get_tile_distribution(),index,[&](){ if constexpr(T::StaticTileDistribution::NDimP == 1) return ct::array<ct::index_t,1>{t%64}; else return ct::array<ct::index_t,2>{t/64,t%64}; }());
    tensor(index)=static_cast<float>(xy[ct::number<0>{}]*64+xy[ct::number<1>{}]);
   });
  });
  ct::static_for<0,T::get_thread_buffer_size(),1>{}([&](auto i){ std::printf("%d ",int(tensor.get_thread_buffer()[i])); });
  std::puts("");
 }
}
int main(){
 using A=decltype(ct::BlockFmhaPipelineQRKSVSDefaultPolicy::GetQKBlockGemm<Problem>().MakeCBlockTile());
 using B=decltype(ct::BlockFmhaPipelineQRKSVSDefaultPolicy::GetQKBlockGemm<MatchingProblem>().MakeCBlockTile());
 dump<A>();dump<B>();
}
