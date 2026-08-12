#include <torch/extension.h>

namespace lmlm {

torch::Tensor fused_bias_gelu_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    bool approximate,
    int block_size
);

torch::Tensor fused_bias_gelu_cpu(
    torch::Tensor input,
    torch::Tensor bias,
    bool approximate
);

}

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
) {
    m.def(
        "fused_bias_gelu_cuda",
        &lmlm::fused_bias_gelu_cuda,
        "LMLM fused Bias + GELU CUDA"
    );

    m.def(
        "fused_bias_gelu_cpu",
        &lmlm::fused_bias_gelu_cpu,
        "LMLM fused Bias + GELU CPU"
    );
}
