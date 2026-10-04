#include <torch/extension.h>

#include <ATen/native/hip/ck_gemm.h>

torch::Tensor fireprot_ck_mm_f32(
    torch::Tensor left,
    torch::Tensor right) {
  TORCH_CHECK(
      left.is_cuda() && right.is_cuda(),
      "left and right must be HIP tensors");
  TORCH_CHECK(
      left.scalar_type() == at::kFloat && right.scalar_type() == at::kFloat,
      "left and right must be FP32");
  TORCH_CHECK(
      left.dim() == 2 && right.dim() == 2 && left.size(1) == right.size(0),
      "left and right must be compatible matrices");
  TORCH_CHECK(
      left.is_contiguous() && right.is_contiguous(),
      "left and right must be contiguous");
  TORCH_CHECK(
      left.get_device() == right.get_device(),
      "left and right must be on the same device");

  const auto m = left.size(0);
  const auto k = left.size(1);
  const auto n = right.size(1);
  auto output = torch::empty({m, n}, left.options());

  // BLAS treats storage as column-major. Reversing the output dimensions
  // computes the row-major expression left @ right without materializing
  // another transpose.
  at::native::gemm_internal_ck<float>(
      'n',
      'n',
      n,
      m,
      k,
      1.0f,
      right.data_ptr<float>(),
      n,
      left.data_ptr<float>(),
      k,
      0.0f,
      output.data_ptr<float>(),
      n);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "mm_f32",
      &fireprot_ck_mm_f32,
      "FireProt direct CK FP32 matrix multiplication");
}
