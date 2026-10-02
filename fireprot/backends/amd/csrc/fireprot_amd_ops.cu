#include <torch/extension.h>

#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int kWaveSize = 64;
constexpr int kLayerNormWidth = 128;
constexpr int kWavesPerBlock = 4;

__device__ __forceinline__ float wave_sum(float value) {
#pragma unroll
  for (int offset = kWaveSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down(value, offset, kWaveSize);
  }
  return __shfl(value, 0, kWaveSize);
}

template <bool HasScale, bool HasOffset>
__global__ void layer_norm_f32_128_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    std::int64_t rows,
    float epsilon) {
  const int lane = threadIdx.x & (kWaveSize - 1);
  const int wave = threadIdx.x / kWaveSize;
  const std::int64_t row =
      static_cast<std::int64_t>(blockIdx.x) * kWavesPerBlock + wave;
  if (row >= rows) {
    return;
  }

  const std::int64_t base = row * kLayerNormWidth;
  const float first = input[base + lane];
  const float second = input[base + lane + kWaveSize];
  const float sum = wave_sum(first + second);
  const float square_sum = wave_sum(first * first + second * second);
  const float mean = sum * (1.0f / static_cast<float>(kLayerNormWidth));
  const float variance = fmaxf(
      square_sum * (1.0f / static_cast<float>(kLayerNormWidth)) - mean * mean,
      0.0f);
  const float inverse_std = rsqrtf(variance + epsilon);

  const float first_scale = HasScale ? weight[lane] : 1.0f;
  const float second_scale =
      HasScale ? weight[lane + kWaveSize] : 1.0f;
  const float first_offset = HasOffset ? bias[lane] : 0.0f;
  const float second_offset =
      HasOffset ? bias[lane + kWaveSize] : 0.0f;
  output[base + lane] =
      (first - mean) * inverse_std * first_scale + first_offset;
  output[base + lane + kWaveSize] =
      (second - mean) * inverse_std * second_scale + second_offset;
}

__device__ __forceinline__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

__global__ void silu_mul_f32_kernel(
    const float4* __restrict__ activation,
    float4* __restrict__ gate,
    std::int64_t vector_count) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= vector_count) {
    return;
  }
  const float4 a = activation[index];
  const float4 b = gate[index];
  gate[index] = make_float4(
      silu(a.x) * b.x,
      silu(a.y) * b.y,
      silu(a.z) * b.z,
      silu(a.w) * b.w);
}

void check_common(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be FP32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

template <bool HasScale, bool HasOffset>
torch::Tensor launch_layer_norm_f32_128(
    const torch::Tensor& input,
    const float* weight,
    const float* bias,
    double epsilon) {
  auto output = torch::empty_like(input);
  const std::int64_t rows = input.numel() / kLayerNormWidth;
  const dim3 threads(kWaveSize * kWavesPerBlock);
  const dim3 blocks((rows + kWavesPerBlock - 1) / kWavesPerBlock);
  const auto stream = c10::hip::getCurrentHIPStream(input.get_device());
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(layer_norm_f32_128_kernel<HasScale, HasOffset>),
      blocks,
      threads,
      0,
      stream.stream(),
      input.data_ptr<float>(),
      weight,
      bias,
      output.data_ptr<float>(),
      rows,
      static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

torch::Tensor fireprot_layer_norm_f32_128(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double epsilon) {
  check_common(input, "input");
  check_common(weight, "weight");
  check_common(bias, "bias");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  TORCH_CHECK(
      input.size(-1) == kLayerNormWidth,
      "last input dimension must be 128");
  TORCH_CHECK(
      weight.numel() == kLayerNormWidth && bias.numel() == kLayerNormWidth,
      "weight and bias must each contain 128 values");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");

  return launch_layer_norm_f32_128<true, true>(
      input, weight.data_ptr<float>(), bias.data_ptr<float>(), epsilon);
}

torch::Tensor fireprot_layer_norm_f32_128_scale(
    torch::Tensor input,
    torch::Tensor weight,
    double epsilon) {
  check_common(input, "input");
  check_common(weight, "weight");
  TORCH_CHECK(
      input.dim() >= 1 && input.size(-1) == kLayerNormWidth,
      "last input dimension must be 128");
  TORCH_CHECK(weight.numel() == kLayerNormWidth, "weight must contain 128 values");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
  return launch_layer_norm_f32_128<true, false>(
      input, weight.data_ptr<float>(), nullptr, epsilon);
}

torch::Tensor fireprot_layer_norm_f32_128_plain(
    torch::Tensor input,
    double epsilon) {
  check_common(input, "input");
  TORCH_CHECK(
      input.dim() >= 1 && input.size(-1) == kLayerNormWidth,
      "last input dimension must be 128");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
  return launch_layer_norm_f32_128<false, false>(
      input, nullptr, nullptr, epsilon);
}

torch::Tensor fireprot_silu_mul_f32(
    torch::Tensor activation,
    torch::Tensor gate) {
  check_common(activation, "activation");
  check_common(gate, "gate");
  TORCH_CHECK(
      activation.sizes() == gate.sizes(),
      "activation and gate shapes must match");
  TORCH_CHECK(
      activation.numel() % 4 == 0,
      "the fused vector path requires a multiple of four values");
  TORCH_CHECK(
      activation.data_ptr() != gate.data_ptr(),
      "activation and gate must not alias");

  const std::int64_t vector_count = activation.numel() / 4;
  constexpr int threads = 256;
  const dim3 blocks((vector_count + threads - 1) / threads);
  const auto stream = c10::hip::getCurrentHIPStream(activation.get_device());
  hipLaunchKernelGGL(
      silu_mul_f32_kernel,
      blocks,
      dim3(threads),
      0,
      stream.stream(),
      reinterpret_cast<const float4*>(activation.data_ptr<float>()),
      reinterpret_cast<float4*>(gate.data_ptr<float>()),
      vector_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return gate;
}
