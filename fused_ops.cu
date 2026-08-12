#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>


namespace lmlm {

constexpr float kSqrt2OverPi =
    0.7978845608028654f;

constexpr float kInvSqrt2 =
    0.7071067811865475f;

constexpr float kGeluCoeff =
    0.044715f;


// ============================================================================
// GELU
// ============================================================================

__device__ __forceinline__
float gelu_tanh(float x)
{
    const float x2 = x * x;
    const float x3 = x2 * x;

    const float inner =
        kSqrt2OverPi *
        (x + kGeluCoeff * x3);

    return 0.5f *
           x *
           (1.0f + tanhf(inner));
}


__device__ __forceinline__
float gelu_exact(float x)
{
    return 0.5f *
           x *
           (1.0f + erff(x * kInvSqrt2));
}


__device__ __forceinline__
float gelu(float x, bool approximate)
{
    return approximate
        ? gelu_tanh(x)
        : gelu_exact(x);
}


// ============================================================================
// Scalar conversion
// ============================================================================

template <typename T>
__device__ __forceinline__
float to_float(T x)
{
    return static_cast<float>(x);
}


template <>
__device__ __forceinline__
float to_float<at::Half>(at::Half x)
{
    return __half2float(
        reinterpret_cast<const __half&>(x)
    );
}


template <>
__device__ __forceinline__
float to_float<at::BFloat16>(at::BFloat16 x)
{
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16&>(x)
    );
}


template <typename T>
__device__ __forceinline__
T from_float(float x)
{
    return static_cast<T>(x);
}


template <>
__device__ __forceinline__
at::Half from_float(float x)
{
    return reinterpret_cast<at::Half>(
        __float2half(x)
    );
}


template <>
__device__ __forceinline__
at::BFloat16 from_float(float x)
{
    return reinterpret_cast<at::BFloat16>(
        __float2bfloat16(x)
    );
}


// ============================================================================
// Scalar CUDA kernel
// ============================================================================

template <typename scalar_t>
__global__ void fused_bias_gelu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,

    int64_t rows,
    int64_t hidden_dim,

    bool approximate
)
{
    const int64_t row =
        static_cast<int64_t>(blockIdx.x);

    const int64_t col =
        static_cast<int64_t>(blockIdx.y) *
        blockDim.x +
        threadIdx.x;

    if (
        row >= rows ||
        col >= hidden_dim
    ) {
        return;
    }

    const int64_t index =
        row * hidden_dim + col;

    const float x =
        to_float(input[index]) +
        to_float(bias[col]);

    output[index] =
        from_float<scalar_t>(
            gelu(x, approximate)
        );
}


// ============================================================================
// FP16 vectorized kernel
// ============================================================================

__global__ void fused_bias_gelu_half2_kernel(
    const __half2* __restrict__ input,
    const __half2* __restrict__ bias,
    __half2* __restrict__ output,

    int64_t rows,
    int64_t hidden_pairs,

    bool approximate
)
{
    const int64_t row =
        static_cast<int64_t>(blockIdx.x);

    const int64_t pair =
        static_cast<int64_t>(blockIdx.y) *
        blockDim.x +
        threadIdx.x;

    if (
        row >= rows ||
        pair >= hidden_pairs
    ) {
        return;
    }

    const int64_t index =
        row * hidden_pairs + pair;

    const __half2 x =
        __hadd2(
            input[index],
            bias[pair]
        );

    const float x0 =
        __half2float(__low2half(x));

    const float x1 =
        __half2float(__high2half(x));

    const float y0 =
        gelu(x0, approximate);

    const float y1 =
        gelu(x1, approximate);

    output[index] =
        __halves2half2(
            __float2half(y0),
            __float2half(y1)
        );
}


// ============================================================================
// BF16 vectorized kernel
// ============================================================================

__global__ void fused_bias_gelu_bf162_kernel(
    const __nv_bfloat162* __restrict__ input,
    const __nv_bfloat162* __restrict__ bias,
    __nv_bfloat162* __restrict__ output,

    int64_t rows,
    int64_t hidden_pairs,

    bool approximate
)
{
    const int64_t row =
        static_cast<int64_t>(blockIdx.x);

    const int64_t pair =
        static_cast<int64_t>(blockIdx.y) *
        blockDim.x +
        threadIdx.x;

    if (
        row >= rows ||
        pair >= hidden_pairs
    ) {
        return;
    }

    const int64_t index =
        row * hidden_pairs + pair;

    const __nv_bfloat162 x =
        __hadd2(
            input[index],
            bias[pair]
        );

    const float x0 =
        __bfloat162float(x.x);

    const float x1 =
        __bfloat162float(x.y);

    const float y0 =
        gelu(x0, approximate);

    const float y1 =
        gelu(x1, approximate);

    __nv_bfloat162 result;

    result.x =
        __float2bfloat16(y0);

    result.y =
        __float2bfloat16(y1);

    output[index] = result;
}


// ============================================================================
// Alignment
// ============================================================================

bool aligned_4(const torch::Tensor& tensor)
{
    const uintptr_t ptr =
        reinterpret_cast<uintptr_t>(
            tensor.data_ptr()
        );

    return (ptr & 0x3) == 0;
}


// ============================================================================
// CPU implementation
// ============================================================================

template <typename scalar_t>
void fused_bias_gelu_cpu_impl(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,
    bool approximate
)
{
    const int64_t hidden_dim =
        input.size(-1);

    const int64_t total =
        input.numel();

    const scalar_t* input_ptr =
        input.data_ptr<scalar_t>();

    const scalar_t* bias_ptr =
        bias.data_ptr<scalar_t>();

    scalar_t* output_ptr =
        output.data_ptr<scalar_t>();

    at::parallel_for(
        0,
        total,
        4096,
        [&](int64_t begin, int64_t end)
        {
            for (
                int64_t i = begin;
                i < end;
                ++i
            ) {
                const int64_t col =
                    i % hidden_dim;

                const float x =
                    static_cast<float>(
                        input_ptr[i]
                    ) +
                    static_cast<float>(
                        bias_ptr[col]
                    );

                const float y =
                    approximate
                    ?
                    0.5f * x *
                    (
                        1.0f +
                        std::tanh(
                            kSqrt2OverPi *
                            (
                                x +
                                kGeluCoeff *
                                x * x * x
                            )
                        )
                    )
                    :
                    0.5f * x *
                    (
                        1.0f +
                        std::erf(
                            x * kInvSqrt2
                        )
                    );

                output_ptr[i] =
                    static_cast<scalar_t>(y);
            }
        }
    );
}


torch::Tensor fused_bias_gelu_cpu(
    torch::Tensor input,
    torch::Tensor bias,
    bool approximate
)
{
    TORCH_CHECK(
        !input.is_cuda(),
        "input must be CPU"
    );

    TORCH_CHECK(
        !bias.is_cuda(),
        "bias must be CPU"
    );

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous"
    );

    TORCH_CHECK(
        bias.is_contiguous(),
        "bias must be contiguous"
    );

    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension"
    );

    TORCH_CHECK(
        bias.dim() == 1,
        "bias must be one-dimensional"
    );

    TORCH_CHECK(
        input.size(-1) == bias.size(0),
        "input last dimension must equal bias dimension"
    );

    auto output =
        torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        input.scalar_type(),
        "lmlm_fused_bias_gelu_cpu",
        [&]
        {
            fused_bias_gelu_cpu_impl<scalar_t>(
                input,
                bias,
                output,
                approximate
            );
        }
    );

    return output;
}


// ============================================================================
// CUDA launch
// ============================================================================

template <typename scalar_t>
void launch_scalar(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,

    int block_size,
    bool approximate
)
{
    const int64_t hidden_dim =
        input.size(-1);

    const int64_t rows =
        input.numel() / hidden_dim;

    const dim3 block(
        static_cast<unsigned int>(
            block_size
        )
    );

    const dim3 grid(
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(
            (hidden_dim +
             block_size - 1) /
            block_size
        )
    );

    const int device =
        input.device().index();

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            device
        );

    fused_bias_gelu_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),

            rows,
            hidden_dim,

            approximate
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ============================================================================
// FP16 launch
// ============================================================================

void launch_half2(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,

    int block_size,
    bool approximate
)
{
    const int64_t hidden_dim =
        input.size(-1);

    const int64_t rows =
        input.numel() / hidden_dim;

    const int64_t pairs =
        hidden_dim / 2;

    const dim3 block(
        static_cast<unsigned int>(
            block_size
        )
    );

    const dim3 grid(
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(
            (pairs +
             block_size - 1) /
            block_size
        )
    );

    const int device =
        input.device().index();

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            device
        );

    const auto* input_ptr =
        reinterpret_cast<const __half2*>(
            input.data_ptr<at::Half>()
        );

    const auto* bias_ptr =
        reinterpret_cast<const __half2*>(
            bias.data_ptr<at::Half>()
        );

    auto* output_ptr =
        reinterpret_cast<__half2*>(
            output.data_ptr<at::Half>()
        );

    fused_bias_gelu_half2_kernel
        <<<grid, block, 0, stream>>>(
            input_ptr,
            bias_ptr,
            output_ptr,

            rows,
            pairs,

            approximate
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ============================================================================
// BF16 launch
// ============================================================================

void launch_bf162(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,

    int block_size,
    bool approximate
)
{
    const int64_t hidden_dim =
        input.size(-1);

    const int64_t rows =
        input.numel() / hidden_dim;

    const int64_t pairs =
        hidden_dim / 2;

    const dim3 block(
        static_cast<unsigned int>(
            block_size
        )
    );

    const dim3 grid(
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(
            (pairs +
             block_size - 1) /
            block_size
        )
    );

    const int device =
        input.device().index();

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            device
        );

    const auto* input_ptr =
        reinterpret_cast<
            const __nv_bfloat162*
        >(
            input.data_ptr<at::BFloat16>()
        );

    const auto* bias_ptr =
        reinterpret_cast<
            const __nv_bfloat162*
        >(
            bias.data_ptr<at::BFloat16>()
        );

    auto* output_ptr =
        reinterpret_cast<
            __nv_bfloat162*
        >(
            output.data_ptr<at::BFloat16>()
        );

    fused_bias_gelu_bf162_kernel
        <<<grid, block_size, 0, stream>>>(
            input_ptr,
            bias_ptr,
            output_ptr,

            rows,
            pairs,

            approximate
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ============================================================================
// CUDA entry point
//
// block_size:
//     0     -> internal heuristic
//     64    -> explicit
//     128   -> explicit
//     256   -> explicit
//     512   -> explicit
// ============================================================================

torch::Tensor fused_bias_gelu_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    bool approximate,
    int block_size
)
{
    TORCH_CHECK(
        input.is_cuda(),
        "input must be CUDA"
    );

    TORCH_CHECK(
        bias.is_cuda(),
        "bias must be CUDA"
    );

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous"
    );

    TORCH_CHECK(
        bias.is_contiguous(),
        "bias must be contiguous"
    );

    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension"
    );

    TORCH_CHECK(
        bias.dim() == 1,
        "bias must be one-dimensional"
    );

    TORCH_CHECK(
        input.device() == bias.device(),
        "input and bias must be on the same CUDA device"
    );

    TORCH_CHECK(
        input.size(-1) == bias.size(0),
        "input last dimension must equal bias dimension"
    );

    TORCH_CHECK(
        input.numel() > 0,
        "input cannot be empty"
    );

    const int64_t hidden_dim =
        input.size(-1);

    TORCH_CHECK(
        hidden_dim > 0,
        "hidden dimension must be positive"
    );


    // --------------------------------------------------------
    // Block-size validation
    // --------------------------------------------------------

    if (block_size == 0) {

        if (hidden_dim <= 128)
            block_size = 128;

        else if (hidden_dim >= 4096)
            block_size = 512;

        else
            block_size = 256;
    }

    TORCH_CHECK(
        block_size == 64 ||
        block_size == 128 ||
        block_size == 256 ||
        block_size == 512,
        "block_size must be 0, 64, 128, 256, or 512"
    );


    auto output =
        torch::empty_like(input);


    c10::cuda::CUDAGuard guard(
        input.device()
    );


    // --------------------------------------------------------
    // FP16 vectorized path
    // --------------------------------------------------------

    if (
        input.scalar_type() ==
        torch::kFloat16
    ) {
        const bool vectorizable =
            (hidden_dim % 2 == 0) &&
            aligned_4(input) &&
            aligned_4(bias) &&
            aligned_4(output);

        if (vectorizable) {
            launch_half2(
                input,
                bias,
                output,
                block_size,
                approximate
            );

            return output;
        }
    }


    // --------------------------------------------------------
    // BF16 vectorized path
    // --------------------------------------------------------

    if (
        input.scalar_type() ==
        torch::kBFloat16
    ) {
        const bool vectorizable =
            (hidden_dim % 2 == 0) &&
            aligned_4(input) &&
            aligned_4(bias) &&
            aligned_4(output);

        if (vectorizable) {
            launch_bf162(
                input,
                bias,
                output,
                block_size,
                approximate
            );

            return output;
        }
    }


    // --------------------------------------------------------
    // Scalar fallback
    // --------------------------------------------------------

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        input.scalar_type(),
        "lmlm_fused_bias_gelu_cuda",
        [&]
        {
            launch_scalar<scalar_t>(
                input,
                bias,
                output,
                block_size,
                approximate
            );
        }
    );

    return output;
}


// ============================================================================
// torch.library
// ============================================================================

TORCH_LIBRARY(lmlm, m)
{
    m.def(
        "fused_bias_gelu("
        "Tensor input, "
        "Tensor bias, "
        "bool approximate=True, "
        "int block_size=0"
        ") -> Tensor"
    );
}


TORCH_LIBRARY_IMPL(
    lmlm,
    CUDA,
    m
)
{
    m.impl(
        "fused_bias_gelu",
        &fused_bias_gelu_cuda
    );
}


TORCH_LIBRARY_IMPL(
    lmlm,
    CPU,
    m
)
{
    m.impl(
        "fused_bias_gelu",
        &fused_bias_gelu_cpu
    );
}

} // namespace lmlm
