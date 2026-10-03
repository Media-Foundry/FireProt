#include <torch/extension.h>

#include <c10/hip/HIPException.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int kWaveSize = 64;
constexpr int kLayerNormWidth = 128;
constexpr int kWavesPerBlock = 4;
constexpr int kChannelFirstRows = 124 * 124;
constexpr int kChannelFirstTileRows = 64;
constexpr int kChannelFirstTileStride = kChannelFirstTileRows + 1;

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

__global__ void layer_norm_f32_128_channel_first_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float epsilon) {
  __shared__ float tile[kLayerNormWidth][kChannelFirstTileStride];
  __shared__ float means[kChannelFirstTileRows];
  __shared__ float inverse_stds[kChannelFirstTileRows];

  const int row_base = blockIdx.x * kChannelFirstTileRows;
  constexpr int tile_elements = kLayerNormWidth * kChannelFirstTileRows;
  for (int index = threadIdx.x; index < tile_elements; index += blockDim.x) {
    const int channel = index / kChannelFirstTileRows;
    const int local_row = index % kChannelFirstTileRows;
    const int row = row_base + local_row;
    tile[channel][local_row] =
        row < kChannelFirstRows
        ? input[static_cast<std::int64_t>(channel) * kChannelFirstRows + row]
        : 0.0f;
  }
  __syncthreads();

  if (threadIdx.x < kChannelFirstTileRows) {
    const int local_row = threadIdx.x;
    const int row = row_base + local_row;
    if (row < kChannelFirstRows) {
      float sum = 0.0f;
      float square_sum = 0.0f;
#pragma unroll
      for (int channel = 0; channel < kLayerNormWidth; ++channel) {
        const float value = tile[channel][local_row];
        sum += value;
        square_sum += value * value;
      }
      const float mean = sum * (1.0f / static_cast<float>(kLayerNormWidth));
      const float variance = fmaxf(
          square_sum * (1.0f / static_cast<float>(kLayerNormWidth)) -
              mean * mean,
          0.0f);
      means[local_row] = mean;
      inverse_stds[local_row] = rsqrtf(variance + epsilon);
    }
  }
  __syncthreads();

  for (int index = threadIdx.x; index < tile_elements; index += blockDim.x) {
    const int local_row = index / kLayerNormWidth;
    const int channel = index % kLayerNormWidth;
    const int row = row_base + local_row;
    if (row < kChannelFirstRows) {
      const float value = tile[channel][local_row];
      output[static_cast<std::int64_t>(row) * kLayerNormWidth + channel] =
          (value - means[local_row]) * inverse_stds[local_row] * weight[channel] +
          bias[channel];
    }
  }
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

__device__ __forceinline__ float sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

template <bool HasMask>
__global__ void sigmoid_mul_f32_kernel(
    const float4* __restrict__ activation,
    float4* __restrict__ output,
    const float* __restrict__ mask,
    std::int64_t vector_count,
    std::int64_t vectors_per_row) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= vector_count) {
    return;
  }
  const float4 a = activation[index];
  const float4 value = output[index];
  const float scale = HasMask ? mask[index / vectors_per_row] : 1.0f;
  output[index] = make_float4(
      (scale * sigmoid(a.x)) * value.x,
      (scale * sigmoid(a.y)) * value.y,
      (scale * sigmoid(a.z)) * value.z,
      (scale * sigmoid(a.w)) * value.w);
}

void check_common(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be FP32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Coalesced reads and writes on either side of a [rows,128] transpose.
// Padding the LDS tile breaks bank conflicts during the layout exchange.
constexpr int kPackedTileRows = 32;

__global__ void sigmoid_pack_f32_128_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ projection,
    float* __restrict__ packed,
    std::int64_t rows) {
  __shared__ float tile[kPackedTileRows][kLayerNormWidth + 1];
  const std::int64_t row_base =
      static_cast<std::int64_t>(blockIdx.x) * kPackedTileRows;
  for (int index = threadIdx.x; index < kPackedTileRows * kLayerNormWidth;
       index += blockDim.x) {
    const int row = index / kLayerNormWidth;
    const int channel = index % kLayerNormWidth;
    const std::int64_t global_row = row_base + row;
    if (global_row < rows) {
      const std::int64_t offset = global_row * kLayerNormWidth + channel;
      tile[row][channel] = sigmoid(gate[offset]) * projection[offset];
    }
  }
  __syncthreads();
  for (int index = threadIdx.x; index < kPackedTileRows * kLayerNormWidth;
       index += blockDim.x) {
    const int channel = index / kPackedTileRows;
    const int row = index % kPackedTileRows;
    if (row_base + row < rows) {
      packed[static_cast<std::int64_t>(channel) * rows + row_base + row] =
          tile[row][channel];
    }
  }
}

__global__ void packed_layer_norm_f32_128_kernel(
    const float* __restrict__ packed,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    std::int64_t rows,
    float epsilon) {
  __shared__ float tile[kLayerNormWidth][kPackedTileRows + 1];
  const std::int64_t row_base =
      static_cast<std::int64_t>(blockIdx.x) * kPackedTileRows;
  for (int index = threadIdx.x; index < kPackedTileRows * kLayerNormWidth;
       index += blockDim.x) {
    const int channel = index / kPackedTileRows;
    const int row = index % kPackedTileRows;
    tile[channel][row] = row_base + row < rows
        ? packed[static_cast<std::int64_t>(channel) * rows + row_base + row]
        : 0.0f;
  }
  __syncthreads();
  const int lane = threadIdx.x % kWaveSize;
  const int wave = threadIdx.x / kWaveSize;
  for (int row = wave; row < kPackedTileRows; row += kWavesPerBlock) {
    if (row_base + row < rows) {
      const float first = tile[lane][row];
      const float second = tile[lane + kWaveSize][row];
      const float mean = wave_sum(first + second) * (1.0f / kLayerNormWidth);
      const float d0 = first - mean;
      const float d1 = second - mean;
      const float variance = wave_sum(d0 * d0 + d1 * d1) / kLayerNormWidth;
      const float inverse_std = rsqrtf(variance + epsilon);
      const std::int64_t base = (row_base + row) * kLayerNormWidth;
      output[base + lane] = d0 * inverse_std * weight[lane] + bias[lane];
      output[base + lane + kWaveSize] =
          d1 * inverse_std * weight[lane + kWaveSize] + bias[lane + kWaveSize];
    }
  }
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

torch::Tensor fireprot_sigmoid_pack_f32_128(
    torch::Tensor gate, torch::Tensor projection) {
  check_common(gate, "gate");
  check_common(projection, "projection");
  TORCH_CHECK(gate.sizes() == projection.sizes(), "gate and projection shapes must match");
  TORCH_CHECK(gate.get_device() == projection.get_device(), "devices must match");
  TORCH_CHECK(gate.dim() == 3 && gate.size(0) == gate.size(1) &&
      gate.size(0) > 0 && gate.size(2) == kLayerNormWidth,
      "input shape must be [N,N,128], N > 0");
  const c10::DeviceGuard device_guard(gate.device());
  const auto rows = gate.numel() / kLayerNormWidth;
  auto output = torch::empty({kLayerNormWidth, gate.size(0), gate.size(1)}, gate.options());
  const auto stream = c10::hip::getCurrentHIPStream(gate.get_device());
  hipLaunchKernelGGL(sigmoid_pack_f32_128_kernel,
      dim3((rows + kPackedTileRows - 1) / kPackedTileRows),
      dim3(kWaveSize * kWavesPerBlock), 0, stream.stream(),
      gate.data_ptr<float>(), projection.data_ptr<float>(), output.data_ptr<float>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fireprot_packed_layer_norm_f32_128(
    torch::Tensor packed, torch::Tensor weight, torch::Tensor bias, double epsilon) {
  check_common(packed, "packed");
  check_common(weight, "weight");
  check_common(bias, "bias");
  TORCH_CHECK(packed.dim() == 3 && packed.size(0) == kLayerNormWidth &&
      packed.size(1) == packed.size(2) && packed.size(1) > 0,
      "packed input shape must be [128,N,N], N > 0");
  TORCH_CHECK(weight.numel() == kLayerNormWidth && bias.numel() == kLayerNormWidth,
      "weight and bias must each contain 128 values");
  TORCH_CHECK(packed.get_device() == weight.get_device() &&
      packed.get_device() == bias.get_device(), "devices must match");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
  const c10::DeviceGuard device_guard(packed.device());
  const auto rows = packed.numel() / kLayerNormWidth;
  auto output = torch::empty({packed.size(1), packed.size(2), kLayerNormWidth}, packed.options());
  const auto stream = c10::hip::getCurrentHIPStream(packed.get_device());
  hipLaunchKernelGGL(packed_layer_norm_f32_128_kernel,
      dim3((rows + kPackedTileRows - 1) / kPackedTileRows),
      dim3(kWaveSize * kWavesPerBlock), 0, stream.stream(),
      packed.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
      output.data_ptr<float>(), rows, static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

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

torch::Tensor fireprot_layer_norm_f32_128_channel_first(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double epsilon) {
  TORCH_CHECK(input.is_cuda(), "input must be a HIP tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be FP32");
  check_common(weight, "weight");
  check_common(bias, "bias");
  TORCH_CHECK(
      input.dim() == 3 && input.size(0) == 124 && input.size(1) == 124 &&
          input.size(2) == kLayerNormWidth,
      "input shape must be [124, 124, 128]");
  TORCH_CHECK(
      input.stride(0) == 124 && input.stride(1) == 1 &&
          input.stride(2) == kChannelFirstRows,
      "input stride must be [124, 1, 15376]");
  TORCH_CHECK(
      weight.numel() == kLayerNormWidth && bias.numel() == kLayerNormWidth,
      "weight and bias must each contain 128 values");
  TORCH_CHECK(
      input.get_device() == weight.get_device() &&
          input.get_device() == bias.get_device(),
      "input, weight, and bias must be on the same device");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");

  auto output = torch::empty(input.sizes(), input.options());
  constexpr int threads = 256;
  const dim3 blocks(
      (kChannelFirstRows + kChannelFirstTileRows - 1) /
      kChannelFirstTileRows);
  const auto stream = c10::hip::getCurrentHIPStream(input.get_device());
  hipLaunchKernelGGL(
      layer_norm_f32_128_channel_first_kernel,
      blocks,
      dim3(threads),
      0,
      stream.stream(),
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
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

torch::Tensor fireprot_sigmoid_mul_f32(
    torch::Tensor activation,
    torch::Tensor output) {
  check_common(activation, "activation");
  check_common(output, "output");
  TORCH_CHECK(
      activation.sizes() == output.sizes(),
      "activation and output shapes must match");
  TORCH_CHECK(
      activation.numel() % 4 == 0,
      "the fused vector path requires a multiple of four values");
  TORCH_CHECK(
      activation.get_device() == output.get_device(),
      "activation and output must be on the same device");
  TORCH_CHECK(
      activation.data_ptr() != output.data_ptr(),
      "activation and output must not alias");

  const std::int64_t vector_count = activation.numel() / 4;
  constexpr int threads = 256;
  const dim3 blocks((vector_count + threads - 1) / threads);
  const auto stream = c10::hip::getCurrentHIPStream(activation.get_device());
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(sigmoid_mul_f32_kernel<false>),
      blocks,
      dim3(threads),
      0,
      stream.stream(),
      reinterpret_cast<const float4*>(activation.data_ptr<float>()),
      reinterpret_cast<float4*>(output.data_ptr<float>()),
      nullptr,
      vector_count,
      1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fireprot_sigmoid_mul_mask_f32(
    torch::Tensor activation,
    torch::Tensor output,
    torch::Tensor mask) {
  check_common(activation, "activation");
  check_common(output, "output");
  check_common(mask, "mask");
  TORCH_CHECK(
      activation.sizes() == output.sizes(),
      "activation and output shapes must match");
  TORCH_CHECK(
      activation.dim() >= 1 && activation.size(-1) % 4 == 0,
      "the last dimension must be a multiple of four");
  TORCH_CHECK(
      mask.numel() == activation.numel() / activation.size(-1),
      "mask must contain one value per activation row");
  TORCH_CHECK(
      activation.get_device() == output.get_device() &&
          activation.get_device() == mask.get_device(),
      "activation, output, and mask must be on the same device");
  TORCH_CHECK(
      activation.data_ptr() != output.data_ptr(),
      "activation and output must not alias");

  const std::int64_t vector_count = activation.numel() / 4;
  const std::int64_t vectors_per_row = activation.size(-1) / 4;
  constexpr int threads = 256;
  const dim3 blocks((vector_count + threads - 1) / threads);
  const auto stream = c10::hip::getCurrentHIPStream(activation.get_device());
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(sigmoid_mul_f32_kernel<true>),
      blocks,
      dim3(threads),
      0,
      stream.stream(),
      reinterpret_cast<const float4*>(activation.data_ptr<float>()),
      reinterpret_cast<float4*>(output.data_ptr<float>()),
      mask.data_ptr<float>(),
      vector_count,
      vectors_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}


namespace {
__global__ void sigmoid_mul_out_kernel(const float4* activation,const float4* value,
                                       float4* output,std::int64_t count) {
  const std::int64_t i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=count) return;
  const float4 g=activation[i],v=value[i];
  output[i]=make_float4(v.x/(1.0f+expf(-g.x)),v.y/(1.0f+expf(-g.y)),
                       v.z/(1.0f+expf(-g.z)),v.w/(1.0f+expf(-g.w)));
}
}

torch::Tensor fireprot_sigmoid_mul_out_f32(torch::Tensor activation,torch::Tensor value) {
  check_common(activation,"activation");
  check_common(value,"value");
  TORCH_CHECK(activation.sizes()==value.sizes(),"shapes must match");
  TORCH_CHECK(activation.device()==value.device(),"devices must match");
  TORCH_CHECK(activation.numel()>0 && activation.numel()%4==0,"requires nonempty float4 vectors");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(activation.data_ptr())%16==0 &&
              reinterpret_cast<std::uintptr_t>(value.data_ptr())%16==0,"requires 16-byte alignment");
  const c10::DeviceGuard guard(activation.device());
  auto output=torch::empty_like(value);
  const auto stream=c10::hip::getCurrentHIPStream(activation.get_device());
  const std::int64_t count=activation.numel()/4;
  hipLaunchKernelGGL(sigmoid_mul_out_kernel,dim3((count+255)/256),dim3(256),0,stream.stream(),
      reinterpret_cast<const float4*>(activation.data_ptr<float>()),
      reinterpret_cast<const float4*>(value.data_ptr<float>()),
      reinterpret_cast<float4*>(output.data_ptr<float>()),count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

namespace {
template<int Rows>
__global__ void triangle_vector_pack_kernel(const float4* gate,const float4* projection,float4* output,int rows) {
    __shared__ float tile[Rows][129];
    int base=blockIdx.x*Rows;
    for(int i=threadIdx.x;i<Rows*32;i+=blockDim.x) {
        int r=i/32,c=(i%32)*4;
        if(base+r<rows) {
            float4 g=gate[(base+r)*32+c/4],p=projection[(base+r)*32+c/4];
            tile[r][c]=(1.0f/(1.0f+expf(-g.x)))*p.x;
            tile[r][c+1]=(1.0f/(1.0f+expf(-g.y)))*p.y;
            tile[r][c+2]=(1.0f/(1.0f+expf(-g.z)))*p.z;
            tile[r][c+3]=(1.0f/(1.0f+expf(-g.w)))*p.w;
        }
    }
    __syncthreads();
    for(int i=threadIdx.x;i<128*(Rows/4);i+=blockDim.x) {
        int c=i/(Rows/4),r=(i%(Rows/4))*4;
        if(base+r<rows) output[(c*rows+base+r)/4]=make_float4(
            tile[r][c],tile[r+1][c],tile[r+2][c],tile[r+3][c]);
    }
}
__device__ float triangle_residual_value(float g, float x, float z, bool add) {
    #pragma clang fp contract(off)
    float value = (1.0f / (1.0f + expf(-g))) * x;
    return add ? z + value : value;
}
__global__ void triangle_residual_kernel(const float4* gate, const float4* value, float4* z,
                         std::int64_t count, bool add) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= count) return;
    float4 g = gate[i], v = value[i], old = add ? z[i] : make_float4(0,0,0,0);
    z[i] = make_float4(triangle_residual_value(g.x,v.x,old.x,add),triangle_residual_value(g.y,v.y,old.y,add),
                      triangle_residual_value(g.z,v.z,old.z,add),triangle_residual_value(g.w,v.w,old.w,add));
}

void check_triangle_vector_tensor(const torch::Tensor& tensor) {
  check_common(tensor, "triangle tensor");
  TORCH_CHECK(tensor.dim()==3 && tensor.size(0)==tensor.size(1) &&
      (tensor.size(0)==1000 || tensor.size(0)==1536) && tensor.size(2)==128,
      "requires exact [1000,1000,128] or [1536,1536,128]");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.data_ptr())%16==0,
      "requires 16-byte alignment");
}
} // namespace

torch::Tensor fireprot_triangle_vector_pack(torch::Tensor gate, torch::Tensor value) {
  check_triangle_vector_tensor(gate); check_triangle_vector_tensor(value);
  TORCH_CHECK(gate.sizes()==value.sizes() && gate.device()==value.device(),
      "shape and device must match");
  const c10::DeviceGuard guard(gate.device());
  const int n=gate.size(0),rows=n*n;
  auto output=torch::empty({128,n,n},gate.options());
  const auto stream=c10::hip::getCurrentHIPStream(gate.get_device());
  hipLaunchKernelGGL(HIP_KERNEL_NAME(triangle_vector_pack_kernel<64>),
      dim3((rows+63)/64),dim3(256),0,stream.stream(),
      reinterpret_cast<const float4*>(gate.data_ptr<float>()),
      reinterpret_cast<const float4*>(value.data_ptr<float>()),
      reinterpret_cast<float4*>(output.data_ptr<float>()),rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fireprot_triangle_residual(torch::Tensor gate,torch::Tensor value,
                                          torch::Tensor z,bool add) {
  check_triangle_vector_tensor(gate); check_triangle_vector_tensor(value);
  check_triangle_vector_tensor(z);
  TORCH_CHECK(gate.sizes()==z.sizes() && value.sizes()==z.sizes() &&
      gate.device()==z.device() && value.device()==z.device(), "shape and device must match");
  TORCH_CHECK(!z.is_alias_of(gate) && !z.is_alias_of(value), "residual target must not alias inputs");
  const c10::DeviceGuard guard(z.device());
  const auto count=z.numel()/4;
  const auto stream=c10::hip::getCurrentHIPStream(z.get_device());
  hipLaunchKernelGGL(triangle_residual_kernel,dim3((count+255)/256),dim3(256),0,stream.stream(),
      reinterpret_cast<const float4*>(gate.data_ptr<float>()),
      reinterpret_cast<const float4*>(value.data_ptr<float>()),
      reinterpret_cast<float4*>(z.data_ptr<float>()),count,add);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return z;
}

namespace {
// Emulate the old wave64 tree with four consecutive channels per lane.
template<bool Square>
__device__ float packed_norm_original_tree(float a,float b,float c,float d) {
    float aa=__shfl_down(a,16,32),bb=__shfl_down(b,16,32);
    float cc=__shfl_down(c,16,32),dd=__shfl_down(d,16,32);
    if constexpr(Square) {a=a*a+aa*aa;b=b*b+bb*bb;c=c*c+cc*cc;d=d*d+dd*dd;}
    else {a+=aa;b+=bb;c+=cc;d+=dd;}
    for(int shift=8;shift;shift/=2) {
        a+=__shfl_down(a,shift,32);b+=__shfl_down(b,shift,32);
        c+=__shfl_down(c,shift,32);d+=__shfl_down(d,shift,32);
    }
    return __shfl((a+c)+(b+d),0,32);
}

__global__ void packed_norm_vector_kernel(const float4* input,const float4* weight,
    const float4* bias,float4* output,int rows,float eps) {
  __shared__ float tile[32][129];
  const int base=blockIdx.x*32;
  for(int i=threadIdx.x;i<128*8;i+=blockDim.x) {
    const int c=i/8,r=(i%8)*4;
    const float4 x=input[(c*rows+base+r)/4];
    tile[r][c]=x.x;tile[r+1][c]=x.y;tile[r+2][c]=x.z;tile[r+3][c]=x.w;
  }
  __syncthreads();
  const int lane=threadIdx.x%32,group=threadIdx.x/32,c=lane*4;
  const float4 w=weight[lane],b=bias[lane];
  for(int r=group;r<32;r+=8) {
    float a=tile[r][c],d=tile[r][c+1],e=tile[r][c+2],f=tile[r][c+3];
    const float mean=packed_norm_original_tree<false>(a,d,e,f)/128;
    a-=mean;d-=mean;e-=mean;f-=mean;
    const float inv=rsqrtf(packed_norm_original_tree<true>(a,d,e,f)/128+eps);
    output[(base+r)*32+lane]=make_float4(a*inv*w.x+b.x,d*inv*w.y+b.y,
                                        e*inv*w.z+b.z,f*inv*w.w+b.w);
  }
}
} // namespace

torch::Tensor fireprot_packed_norm_vector(torch::Tensor input,torch::Tensor weight,
                                         torch::Tensor bias,double epsilon) {
  check_common(input,"input");check_common(weight,"weight");check_common(bias,"bias");
  TORCH_CHECK(input.dim()==3 && input.size(0)==128 && input.size(1)==input.size(2) &&
      (input.size(1)==1000 || input.size(1)==1536),"requires exact [128,N,N], N1000 or N1536");
  TORCH_CHECK(weight.numel()==128 && bias.numel()==128,"requires 128 affine values");
  TORCH_CHECK(input.device()==weight.device() && input.device()==bias.device(),"devices must match");
  for(const auto& tensor : {input,weight,bias}) {
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.data_ptr())%16==0,"requires 16-byte alignment");
  }
  TORCH_CHECK(epsilon>0,"epsilon must be positive");
  const c10::DeviceGuard guard(input.device());
  const int n=input.size(1),rows=n*n;
  auto output=torch::empty({n,n,128},input.options());
  const auto stream=c10::hip::getCurrentHIPStream(input.get_device());
  hipLaunchKernelGGL(packed_norm_vector_kernel,dim3(rows/32),dim3(256),0,stream.stream(),
      reinterpret_cast<const float4*>(input.data_ptr<float>()),
      reinterpret_cast<const float4*>(weight.data_ptr<float>()),
      reinterpret_cast<const float4*>(bias.data_ptr<float>()),
      reinterpret_cast<float4*>(output.data_ptr<float>()),rows,static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

namespace {
using DualV4 = float __attribute__((ext_vector_type(4)));
// C64/K16, four wave64 groups; preserve FP32 accumulation and fuse the gate.
__global__ void dual_projection_kernel(const float* __restrict__ x,
 const float* __restrict__ wg,const float* __restrict__ wp,float* __restrict__ out,int rows) {
#if defined(__gfx90a__)
    __shared__ float input[64][16+4];
    __shared__ float gate_weight[64][16+4],value_weight[64][16+4];
    const int tid=threadIdx.x,lane=tid%64,wave=tid/64;
    const int first=blockIdx.x*64,channel=blockIdx.y*64;
    DualV4 gate[64/16]={},value[64/16]={};
    for(int base=0;base<128;base+=16){
        for(int i=tid;i<64*(16/4);i+=256){
            const int r=i/(16/4),k=(i%(16/4))*4;
            float4 v=make_float4(0,0,0,0);
            if(first+r<rows)v=*reinterpret_cast<const float4*>(x+(first+r)*128+base+k);
            *reinterpret_cast<float4*>(&input[r][k])=v;
        }
        for(int i=tid;i<64*(16/4);i+=256){
            const int r=i/(16/4),k=(i%(16/4))*4,wi=(channel+r)*128+base+k;
            *reinterpret_cast<float4*>(&gate_weight[r][k])=*reinterpret_cast<const float4*>(wg+wi);
            *reinterpret_cast<float4*>(&value_weight[r][k])=*reinterpret_cast<const float4*>(wp+wi);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<16;k+=4){
            const float b=input[wave*16+lane%16][k+lane/16];
            #pragma unroll
            for(int g=0;g<64/16;g++){
                const int r=g*16+lane%16;
                gate[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(gate_weight[r][k+lane/16],b,gate[g],0,0,0);
                value[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(value_weight[r][k+lane/16],b,value[g],0,0,0);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int g=0;g<64/16;g++){
        #pragma unroll
        for(int j=0;j<4;j++){
            const int r=first+wave*16+lane%16;
            if(r<rows)out[(channel+g*16+lane/16*4+j)*rows+r]=(1.f/(1.f+expf(-gate[g][j])))*value[g][j];
        }
    }
#endif
}
} // namespace

torch::Tensor fireprot_dual_projection(torch::Tensor input,torch::Tensor gate_weight,
                                       torch::Tensor value_weight) {
  check_triangle_vector_tensor(input);
  check_common(gate_weight,"gate_weight");check_common(value_weight,"value_weight");
  TORCH_CHECK(!at::GradMode::is_enabled(),"dual projection is inference only");
  for(const auto& weight : {gate_weight,value_weight}) {
    TORCH_CHECK(weight.dim()==2 && weight.size(0)==128 && weight.size(1)==128,
        "requires [128,128] weights");
    TORCH_CHECK(weight.device()==input.device(),"devices must match");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(weight.data_ptr())%16==0,
        "requires 16-byte alignment");
  }
  const c10::DeviceGuard guard(input.device());
  hipDeviceProp_t properties{};
  TORCH_CHECK(hipGetDeviceProperties(&properties,input.get_device())==hipSuccess,
      "cannot query device properties");
  TORCH_CHECK(std::string(properties.gcnArchName).substr(0,6)=="gfx90a","requires gfx90a");
  const int n=input.size(0),rows=n*n;
  auto output=torch::empty({128,n,n},input.options());
  const auto stream=c10::hip::getCurrentHIPStream(input.get_device());
  hipLaunchKernelGGL(dual_projection_kernel,dim3((rows+63)/64,2),dim3(256),0,stream.stream(),
      input.data_ptr<float>(),gate_weight.data_ptr<float>(),value_weight.data_ptr<float>(),
      output.data_ptr<float>(),rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

namespace {
// C128/R128/K16/W8; padded channel stride preserves the logical [128,N,N] shape.
__global__ void dual_projection_tiled_kernel(const float* __restrict__ x,
 const float* __restrict__ wg,const float* __restrict__ wp,float* __restrict__ out,int rows,int stride) {
#if defined(__gfx90a__)
    __shared__ float input[128][16+4];
    __shared__ float gate_weight[128][16+4],value_weight[128][16+4];
    const int tid=threadIdx.x,lane=tid%64,wave=tid/64;
    const int first=blockIdx.x*128,channel=blockIdx.y*128;
    DualV4 gate[128/(16*8)][128/16]={},value[128/(16*8)][128/16]={};
    for(int base=0;base<128;base+=16){
        for(int i=tid;i<128*(16/4);i+=64*8){
            const int r=i/(16/4),k=(i%(16/4))*4;
            float4 v=make_float4(0,0,0,0);
            if(first+r<rows)v=*reinterpret_cast<const float4*>(x+(first+r)*128+base+k);
            *reinterpret_cast<float4*>(&input[r][k])=v;
        }
        for(int i=tid;i<128*(16/4);i+=64*8){
            const int r=i/(16/4),k=(i%(16/4))*4,wi=(channel+r)*128+base+k;
            *reinterpret_cast<float4*>(&gate_weight[r][k])=*reinterpret_cast<const float4*>(wg+wi);
            *reinterpret_cast<float4*>(&value_weight[r][k])=*reinterpret_cast<const float4*>(wp+wi);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<16;k+=4){
            #pragma unroll
            for(int p=0;p<128/(16*8);p++) {
                const float b=input[p*(16*8)+wave*16+lane%16][k+lane/16];
                #pragma unroll
                for(int g=0;g<128/16;g++){
                    const int r=g*16+lane%16;
                    gate[p][g]=__builtin_amdgcn_mfma_f32_16x16x4f32(gate_weight[r][k+lane/16],b,gate[p][g],0,0,0);
                    value[p][g]=__builtin_amdgcn_mfma_f32_16x16x4f32(value_weight[r][k+lane/16],b,value[p][g],0,0,0);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int p=0;p<128/(16*8);p++) {
        #pragma unroll
        for(int g=0;g<128/16;g++){
            #pragma unroll
            for(int j=0;j<4;j++){
                const int r=first+p*(16*8)+wave*16+lane%16;
                if(r<rows)out[(channel+g*16+lane/16*4+j)*stride+r]=(1.f/(1.f+expf(-gate[p][g][j])))*value[p][g][j];
            }
        }
    }
#endif
}
} // namespace

torch::Tensor fireprot_dual_projection_tiled(torch::Tensor input,torch::Tensor gate_weight,
                                       torch::Tensor value_weight) {
  check_triangle_vector_tensor(input);
  check_common(gate_weight,"gate_weight");check_common(value_weight,"value_weight");
  TORCH_CHECK(!at::GradMode::is_enabled(),"dual projection is inference only");
  for(const auto& weight : {gate_weight,value_weight}) {
    TORCH_CHECK(weight.dim()==2 && weight.size(0)==128 && weight.size(1)==128,
        "requires [128,128] weights");
    TORCH_CHECK(weight.device()==input.device(),"devices must match");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(weight.data_ptr())%16==0,
        "requires 16-byte alignment");
  }
  const c10::DeviceGuard guard(input.device());
  hipDeviceProp_t properties{};
  TORCH_CHECK(hipGetDeviceProperties(&properties,input.get_device())==hipSuccess,
      "cannot query device properties");
  TORCH_CHECK(std::string(properties.gcnArchName).substr(0,6)=="gfx90a","requires gfx90a");
  const int n=input.size(0),rows=n*n;
  auto output=torch::empty({128,rows+128},input.options());
  const auto stream=c10::hip::getCurrentHIPStream(input.get_device());
  hipLaunchKernelGGL(dual_projection_tiled_kernel,dim3((rows+127)/128,1),dim3(512),0,stream.stream(),
      input.data_ptr<float>(),gate_weight.data_ptr<float>(),value_weight.data_ptr<float>(),
      output.data_ptr<float>(),rows,rows+128);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output.as_strided({128,n,n},{rows+128,n,1});
}

namespace qkv_detail {
using QkvV4=float __attribute__((ext_vector_type(4)));
constexpr int ChannelWaves=64/16;
constexpr int RowWaves=16/ChannelWaves;
constexpr int RowsPerWave=128/RowWaves;
static_assert(16%ChannelWaves==0 && RowsPerWave%16==0 && 128%64==0);
__global__ __launch_bounds__(16*64) void project_qkv(const float* __restrict__ x,
 const float* __restrict__ wq,const float* __restrict__ wk,const float* __restrict__ wv,
 float* __restrict__ out,int rows) {
#if defined(__gfx90a__)
    __shared__ float input[128][16+4];
    __shared__ float qweight[64][16+4],kweight[64][16+4],vweight[64][16+4];
    const int tid=threadIdx.x,lane=tid%64,wave=tid/64;
    const int first=blockIdx.x*128,channel=blockIdx.y*64;
    const int wave_row=(wave/ChannelWaves)*RowsPerWave,wave_col=(wave%ChannelWaves)*16;
    QkvV4 q[RowsPerWave/16]={},k[RowsPerWave/16]={},v[RowsPerWave/16]={};
    for(int base=0;base<128;base+=16){
        for(int i=tid;i<128*(16/4);i+=16*64){
            const int r=i/(16/4),c=(i%(16/4))*4;
            float4 value=make_float4(0,0,0,0);
            if(first+r<rows)value=*reinterpret_cast<const float4*>(x+(first+r)*128+base+c);
            *reinterpret_cast<float4*>(&input[r][c])=value;
        }
        for(int i=tid;i<64*(16/4);i+=16*64){
            const int r=i/(16/4),c=(i%(16/4))*4,wi=(channel+r)*128+base+c;
            *reinterpret_cast<float4*>(&qweight[r][c])=*reinterpret_cast<const float4*>(wq+wi);
            *reinterpret_cast<float4*>(&kweight[r][c])=*reinterpret_cast<const float4*>(wk+wi);
            *reinterpret_cast<float4*>(&vweight[r][c])=*reinterpret_cast<const float4*>(wv+wi);
        }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<16;kk+=4){
            const int c=kk+lane/16,wr=wave_col+lane%16;
            const float bq=qweight[wr][c],bk=kweight[wr][c],bv=vweight[wr][c];
            #pragma unroll
            for(int g=0;g<RowsPerWave/16;g++){
                const float a=input[wave_row+g*16+lane%16][c];
                q[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(a,bq,q[g],0,0,0);
                k[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(a,bk,k[g],0,0,0);
                v[g]=__builtin_amdgcn_mfma_f32_16x16x4f32(a,bv,v[g],0,0,0);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int g=0;g<RowsPerWave/16;g++){
        #pragma unroll
        for(int j=0;j<4;j++){
            const int r=first+wave_row+g*16+lane/16*4+j,c=channel+wave_col+lane%16;
            if(r<rows){
                out[r*128+c]=q[g][j]*0.1767766952966369f;
                out[(rows+r)*128+c]=k[g][j];
                out[(2*rows+r)*128+c]=v[g][j];
            }
        }
    }
#endif
}
} // namespace qkv_detail

torch::Tensor fireprot_qkv_projection(torch::Tensor input,torch::Tensor wq,
                                      torch::Tensor wk,torch::Tensor wv) {
  TORCH_CHECK(!at::GradMode::is_enabled(), "QKV projection is inference only");
  check_common(input,"input");
  TORCH_CHECK(input.dim()==3 && input.size(2)==128 &&
      ((input.size(0)==1000 && input.size(1)==1000) ||
       (input.size(0)==512 && input.size(1)==1536)), "unsupported QKV input shape");
  for(const auto& tensor : {input,wq,wk,wv}) {
    check_common(tensor,"QKV tensor");
    TORCH_CHECK(tensor.device()==input.device(),"devices must match");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.data_ptr())%16==0,
        "requires 16-byte alignment");
  }
  for(const auto& weight : {wq,wk,wv}) {
    TORCH_CHECK(weight.dim()==2 && weight.size(0)==128 && weight.size(1)==128,
        "requires [128,128] weights");
  }
  const c10::DeviceGuard guard(input.device());
  hipDeviceProp_t properties{};
  TORCH_CHECK(hipGetDeviceProperties(&properties,input.get_device())==hipSuccess,
      "cannot query device properties");
  TORCH_CHECK(std::string(properties.gcnArchName).substr(0,6)=="gfx90a","requires gfx90a");
  const int batch=input.size(0),n=input.size(1),rows=batch*n;
  auto output=torch::empty({3,batch,n,128},input.options());
  const auto stream=c10::hip::getCurrentHIPStream(input.get_device());
  hipLaunchKernelGGL(qkv_detail::project_qkv,dim3((rows+127)/128,2),dim3(1024),0,stream.stream(),
      input.data_ptr<float>(),wq.data_ptr<float>(),wk.data_ptr<float>(),wv.data_ptr<float>(),
      output.data_ptr<float>(),rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

namespace transposed_norm_detail {
struct Moment { float mean, m2, count; };
__device__ Moment merge(Moment a, Moment b) {
    float total=a.count+b.count;
    if(total==0) return {0,0,0};
    float reciprocal=1.0f/total;
    float wa=a.count*reciprocal, wb=b.count*reciprocal;
    float delta=b.mean-a.mean;
    return {wa*a.mean+wb*b.mean,
            a.m2+b.m2+delta*delta*a.count*wb,total};
}
template<int Waves>
__global__ void transpose_norm(const float* x,const float* weight,const float* bias,
                               float* out,int n,float eps) {
    int lane=threadIdx.x%64, wave=threadIdx.x/64;
    int row=blockIdx.x*Waves+wave;
    if(row>=n*n)return;
    int source=(row%n*n+row/n)*128;
    float4 v=make_float4(0,0,0,0);
    Moment m{0,0,0};
    if(lane<32){
        v=*reinterpret_cast<const float4*>(x+source+lane*4);
        float values[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int i=0;i<4;i++){
            float delta=values[i]-m.mean;
            m.count+=1;
            m.mean+=delta*(1.0f/m.count);
            m.m2+=delta*(values[i]-m.mean);
        }
    }
    #pragma unroll
    for(int offset=32;offset>0;offset/=2){
        Moment other{__shfl_down(m.mean,offset,64),__shfl_down(m.m2,offset,64),
                     __shfl_down(m.count,offset,64)};
        m=merge(other,m);
    }
    float mean=__shfl(m.mean,0,64), var=__shfl(m.m2,0,64)*(1.0f/128);
    float inv=rsqrtf(var+eps);
    if(lane<32){
        float4 w=*reinterpret_cast<const float4*>(weight+lane*4);
        float4 b=*reinterpret_cast<const float4*>(bias+lane*4);
        float4 y=make_float4(w.x*(inv*(v.x-mean))+b.x,w.y*(inv*(v.y-mean))+b.y,
                            w.z*(inv*(v.z-mean))+b.z,w.w*(inv*(v.w-mean))+b.w);
        *reinterpret_cast<float4*>(out+row*128+lane*4)=y;
    }
}
}

torch::Tensor fireprot_transposed_layer_norm(torch::Tensor input,torch::Tensor weight,
                                            torch::Tensor bias,double epsilon) {
  TORCH_CHECK(!at::GradMode::is_enabled(),"transposed LayerNorm is inference only");
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==torch::kFloat32 && !input.requires_grad(),
      "requires a CUDA FP32 inference input");
  TORCH_CHECK(input.dim()==3 && (input.size(0)==1000 || input.size(0)==1536) &&
      input.size(1)==input.size(0) && input.size(2)==128,"unsupported shape");
  const int n=input.size(0);
  TORCH_CHECK(input.stride(0)==128 && input.stride(1)==n*128 && input.stride(2)==1,
      "requires transposed pair strides");
  TORCH_CHECK(epsilon==1e-5,"unsupported epsilon");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(input.data_ptr())%16==0,"unaligned input");
  for(const auto& tensor : {weight,bias}) {
    check_common(tensor,"affine tensor");
    TORCH_CHECK(tensor.device()==input.device() && tensor.dim()==1 && tensor.size(0)==128,
        "requires same-device [128] affine tensors");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.data_ptr())%16==0,"unaligned affine tensor");
  }
  const c10::DeviceGuard guard(input.device());
  hipDeviceProp_t properties{};
  TORCH_CHECK(hipGetDeviceProperties(&properties,input.get_device())==hipSuccess,
      "cannot query device properties");
  TORCH_CHECK(std::string(properties.gcnArchName).substr(0,6)=="gfx90a","requires gfx90a");
  auto output=torch::empty({n,n,128},input.options());
  const auto stream=c10::hip::getCurrentHIPStream(input.get_device());
  hipLaunchKernelGGL(transposed_norm_detail::transpose_norm<8>,dim3((n*n+7)/8),dim3(512),0,
      stream.stream(),input.data_ptr<float>(),weight.data_ptr<float>(),bias.data_ptr<float>(),
      output.data_ptr<float>(),n,static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
