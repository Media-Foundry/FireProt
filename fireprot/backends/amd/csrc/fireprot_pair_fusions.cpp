#include <torch/extension.h>
torch::Tensor mask_softmax(torch::Tensor, torch::Tensor, int, int, int);
torch::Tensor transpose_bf16(torch::Tensor);
torch::Tensor batched_transpose(torch::Tensor);
std::vector<torch::Tensor> pack_qkv(torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_qkv", &pack_qkv);
    m.def("mask_softmax", &mask_softmax);
    m.def("transpose_bf16", &transpose_bf16);
    m.def("batched_transpose", &batched_transpose);
}
