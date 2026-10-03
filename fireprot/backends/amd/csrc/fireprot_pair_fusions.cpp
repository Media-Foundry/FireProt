#include <torch/extension.h>
torch::Tensor mask_softmax(torch::Tensor, torch::Tensor, int, int, int);
torch::Tensor transpose_bf16(torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mask_softmax", &mask_softmax);
    m.def("transpose_bf16", &transpose_bf16);
}
