#include <torch/extension.h>

torch::Tensor fused_bias_gelu_cuda(torch::Tensor input, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_gelu", &fused_bias_gelu_cuda, "Fused Bias GELU CUDA");
}
