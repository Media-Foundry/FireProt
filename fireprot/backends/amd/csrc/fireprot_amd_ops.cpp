#include <torch/extension.h>

torch::Tensor fireprot_layer_norm_f32_128(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double epsilon);

torch::Tensor fireprot_layer_norm_f32_128_scale(
    torch::Tensor input,
    torch::Tensor weight,
    double epsilon);

torch::Tensor fireprot_layer_norm_f32_128_plain(
    torch::Tensor input,
    double epsilon);

torch::Tensor fireprot_layer_norm_f32_128_channel_first(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double epsilon);

torch::Tensor fireprot_silu_mul_f32(
    torch::Tensor activation,
    torch::Tensor gate);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "layer_norm_f32_128",
      &fireprot_layer_norm_f32_128,
      "FireProt CK-style FP32 LayerNorm with width 128");
  module.def(
      "layer_norm_f32_128_scale",
      &fireprot_layer_norm_f32_128_scale,
      "FireProt CK-style FP32 LayerNorm with scale and width 128");
  module.def(
      "layer_norm_f32_128_plain",
      &fireprot_layer_norm_f32_128_plain,
      "FireProt CK-style FP32 LayerNorm without affine parameters");
  module.def(
      "layer_norm_f32_128_channel_first",
      &fireprot_layer_norm_f32_128_channel_first,
      "FireProt fused channel-first transpose and FP32 LayerNorm");
  module.def(
      "silu_mul_f32",
      &fireprot_silu_mul_f32,
      "FireProt fused FP32 SiLU multiplied by a gate");
}
