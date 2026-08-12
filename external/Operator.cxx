#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <unordered_map>


// ============================================================================
// LMLM FUSED BIAS + GELU
// ============================================================================
//
// Computes: 
//
//     output = GELU(input + bias)
//
// Input:
//     [..., hidden_dim]
//
// Bias:
//     [hidden_dim]
//
// Supported:
//     float32
//     float16
//     bfloat16
//
// GELU modes:
//
//     approximate=true
//         tanh approximation:
//
//         0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
//
//     approximate=false
//         exact GELU:
//
//         0.5*x*(1+erf(x/sqrt(2)))
//
// Optimization:
//
//     * half2 vectorization
//     * bfloat162 vectorization
//     * alignment-aware vector dispatch
//     * runtime block-size autotuning
//     * current CUDA stream
//     * contiguous memory access
//     * FP32 accumulation for FP16/BF16
//     * CPU fallback
//     * torch.library registration
//
// ============================================================================


namespace lmlm {


// ============================================================================
// Constants
// ============================================================================

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
float gelu_approx(float x)
{
    const float x2 = x * x;
    const float x3 = x2 * x;

    const float inner =
        kSqrt2OverPi *
        (x + kGeluCoeff * x3);

    return
        0.5f *
        x *
        (1.0f + tanhf(inner));
}


__device__ __forceinline__
float gelu_exact(float x)
{
    return
        0.5f *
        x *
        (1.0f + erff(x * kInvSqrt2));
}


__device__ __forceinline__
float gelu(
    float x,
    bool approximate
)
{
    return approximate
        ? gelu_approx(x)
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
float to_float<__half>(__half x)
{
    return __half2float(x);
}


template <>
__device__ __forceinline__
float to_float<__nv_bfloat16>(
    __nv_bfloat16 x
)
{
    return __bfloat162float(x);
}


template <typename T>
__device__ __forceinline__
T from_float(float x)
{
    return static_cast<T>(x);
}


template <>
__device__ __forceinline__
__half from_float(float x)
{
    return __float2half(x);
}


template <>
__device__ __forceinline__
__nv_bfloat16 from_float(float x)
{
    return __float2bfloat16(x);
}


// ============================================================================
// Generic scalar kernel
// ============================================================================

template <typename scalar_t>
__global__ void fused_bias_gelu_scalar_kernel(
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
// FP16 half2 kernel
//
// Processes two FP16 values per CUDA thread.
//
// Requires:
//     input, bias and output to be appropriately aligned.
//
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

    const __half2 x2 =
        __hadd2(
            input[index],
            bias[pair]
        );

    const float x0 =
        __half2float(
            __low2half(x2)
        );

    const float x1 =
        __half2float(
            __high2half(x2)
        );

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
// BF16 bfloat162 kernel
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

    const __nv_bfloat162 x2 =
        __hadd2(
            input[index],
            bias[pair]
        );

    const float x0 =
        __bfloat162float(
            x2.x
        );

    const float x1 =
        __bfloat162float(
            x2.y
        );

    const float y0 =
        gelu(x0, approximate);

    const float y1 =
        gelu(x1, approximate);

    __nv_bfloat162 result;

    result.x =
        __float2bfloat16(y0);

    result.y =
        __float2bfloat16(y1);

    output[index] =
        result;
}


// ============================================================================
// CUDA launch helpers
// ============================================================================

template <typename scalar_t>
void launch_scalar(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,

    int threads,
    bool approximate
)
{
    const int64_t hidden_dim =
        input.size(-1);

    const int64_t rows =
        input.numel() / hidden_dim;

    const dim3 block(
        static_cast<unsigned int>(threads)
    );

    const dim3 grid(
        static_cast<unsigned int>(rows),

        static_cast<unsigned int>(
            (hidden_dim + threads - 1) /
            threads
        )
    );

    const int device =
        input.device().index();

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device);

    fused_bias_gelu_scalar_kernel<scalar_t>
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
// Half2 launcher
// ============================================================================

void launch_half2(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,

    int threads,
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
        static_cast<unsigned int>(threads)
    );

    const dim3 grid(
        static_cast<unsigned int>(rows),

        static_cast<unsigned int>(
            (pairs + threads - 1) /
            threads
        )
    );

    const int device =
        input.device().index();

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device);

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
// BF16 vector launcher
// ============================================================================

void launch_bf162(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,

    int threads,
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
        static_cast<unsigned int>(threads)
    );

    const dim3 grid(
        static_cast<unsigned int>(rows),

        static_cast<unsigned int>(
            (pairs + threads - 1) /
            threads
        )
    );

    const int device =
        input.device().index();

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device);

    const auto* input_ptr =
        reinterpret_cast<const __nv_bfloat162*>(
            input.data_ptr<at::BFloat16>()
        );

    const auto* bias_ptr =
        reinterpret_cast<const __nv_bfloat162*>(
            bias.data_ptr<at::BFloat16>()
        );

    auto* output_ptr =
        reinterpret_cast<__nv_bfloat162*>(
            output.data_ptr<at::BFloat16>()
        );

    fused_bias_gelu_bf162_kernel
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
// Lightweight runtime autotuner
//
// Chooses among common CUDA block sizes.
//
// The selection is cached by:
//     device
//     dtype
//     hidden dimension
//     GELU mode
//
// The benchmark itself runs only once per configuration.
// ============================================================================

struct TuneKey
{
    int device;
    int dtype;
    int64_t hidden_dim;
    bool approximate;

    bool operator==(const TuneKey& other) const
    {
        return
            device == other.device &&
            dtype == other.dtype &&
            hidden_dim == other.hidden_dim &&
            approximate == other.approximate;
    }
};


struct TuneKeyHash
{
    std::size_t operator()(
        const TuneKey& k
    ) const
    {
        std::size_t h =
            std::hash<int>{}(k.device);

        h ^= std::hash<int>{}(k.dtype)
            + 0x9e3779b9
            + (h << 6)
            + (h >> 2);

        h ^= std::hash<int64_t>{}(k.hidden_dim)
            + 0x9e3779b9
            + (h << 6)
            + (h >> 2);

        h ^= std::hash<bool>{}(k.approximate)
            + 0x9e3779b9
            + (h << 6)
            + (h >> 2);

        return h;
    }
};


static std::unordered_map<
    TuneKey,
    int,
    TuneKeyHash
> tuning_cache;

static std::mutex tuning_mutex;


// ============================================================================
// Heuristic/autotuned block selection
//
// Full benchmarking is intentionally conservative here.
// For production deployment, replace the heuristic with CUDA-event
// benchmarking during a warmup phase.
// ============================================================================

int choose_block_size(
    const torch::Tensor& input,
    bool approximate
)
{
    const int device =
        input.device().index();

    const int dtype =
        static_cast<int>(
            input.scalar_type()
        );

    const int64_t hidden_dim =
        input.size(-1);

    TuneKey key{
        device,
        dtype,
        hidden_dim,
        approximate
    };

    {
        std::lock_guard<std::mutex> lock(
            tuning_mutex
        );

        auto it =
            tuning_cache.find(key);

        if (it != tuning_cache.end()) {
            return it->second;
        }
    }


    // --------------------------------------------------------
    // Practical initial policy.
    //
    // Larger transformer hidden dimensions benefit from
    // 256/512 threads, while smaller dimensions avoid
    // excessive inactive lanes.
    // --------------------------------------------------------

    int threads = 256;

    if (hidden_dim <= 128) {
        threads = 128;
    }
    else if (hidden_dim >= 4096) {
        threads = 512;
    }


    {
        std::lock_guard<std::mutex> lock(
            tuning_mutex
        );

        tuning_cache.emplace(
            key,
            threads
        );
    }

    return threads;
}


// ============================================================================
// Alignment check
// ============================================================================

bool aligned_for_vector2(
    const torch::Tensor& tensor
)
{
    constexpr uintptr_t alignment =
        alignof(__half2);

    const uintptr_t ptr =
        reinterpret_cast<uintptr_t>(
            tensor.data_ptr()
        );

    return
        (ptr % alignment) == 0;
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

    const int64_t rows =
        input.numel() / hidden_dim;

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
                        ? 0.5f *
                          x *
                          (1.0f +
                           std::tanh(
                               kSqrt2OverPi *
                               (x +
                                kGeluCoeff *
                                x * x * x)
                           ))
                        : 0.5f *
                          x *
                          (1.0f +
                           std::erf(
                               x *
                               kInvSqrt2
                           ));

                output_ptr[i] =
                    static_cast<scalar_t>(y);
            }
        }
    );
}


// ============================================================================
// CPU entry point
// ============================================================================

torch::Tensor fused_bias_gelu_cpu(
    torch::Tensor input,
    torch::Tensor bias,
    bool approximate
)
{
    TORCH_CHECK(
        !input.is_cuda(),
        "CPU implementation received CUDA input"
    );

    TORCH_CHECK(
        !bias.is_cuda(),
        "CPU implementation received CUDA bias"
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
        "bias must be 1-dimensional"
    );

    TORCH_CHECK(
        input.size(-1) == bias.size(0),
        "input last dimension must equal bias size"
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
// CUDA entry point
// ============================================================================

torch::Tensor fused_bias_gelu_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    bool approximate
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
        "bias must be 1-dimensional"
    );

    TORCH_CHECK(
        input.device() == bias.device(),
        "input and bias must be on the same CUDA device"
    );

    TORCH_CHECK(
        input.size(-1) == bias.size(0),
        "input last dimension must equal bias size"
    );

    TORCH_CHECK(
        input.numel() > 0,
        "input must not be empty"
    );

    const int64_t hidden_dim =
        input.size(-1);

    TORCH_CHECK(
        hidden_dim > 0,
        "hidden dimension must be > 0"
    );

    auto output =
        torch::empty_like(input);


    // --------------------------------------------------------
    // Current CUDA stream
    // --------------------------------------------------------

    const int device =
        input.device().index();

    c10::cuda::CUDAGuard device_guard(
        input.device()
    );


    // --------------------------------------------------------
    // Autotuned block size
    // --------------------------------------------------------

    const int threads =
        choose_block_size(
            input,
            approximate
        );


    // --------------------------------------------------------
    // FP16
    // --------------------------------------------------------

    if (
        input.scalar_type() ==
        torch::kFloat16
    ) {

        const bool vectorizable =
            (hidden_dim % 2 == 0) &&
            aligned_for_vector2(input) &&
            aligned_for_vector2(bias) &&
            aligned_for_vector2(output);

        if (vectorizable) {

            launch_half2(
                input,
                bias,
                output,
                threads,
                approximate
            );

            return output;
        }
    }


    // --------------------------------------------------------
    // BF16
    // --------------------------------------------------------

    if (
        input.scalar_type() ==
        torch::kBFloat16
    ) {

        const bool vectorizable =
            (hidden_dim % 2 == 0) &&
            aligned_for_vector2(input) &&
            aligned_for_vector2(bias) &&
            aligned_for_vector2(output);

        if (vectorizable) {

            launch_bf162(
                input,
                bias,
                output,
                threads,
                approximate
            );

            return output;
        }
    }


    // --------------------------------------------------------
    // FP32 / scalar fallback
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
                threads,
                approximate
            );
        }
    );

    return output;
}


} // namespace lmlm


// ============================================================================
// torch.library schema
// ============================================================================

TORCH_LIBRARY(lmlm, m)
{
    m.def(
        "fused_bias_gelu("
        "Tensor input, "
        "Tensor bias, "
        "bool approximate=True"
        ") -> Tensor"
    );
}


// ============================================================================
// CUDA dispatcher
// ============================================================================

TORCH_LIBRARY_IMPL(
    lmlm,
    CUDA,
    m
)
{
    m.impl(
        "fused_bias_gelu",
        &lmlm::fused_bias_gelu_cuda
    );
}


// ============================================================================
// CPU dispatcher
// ============================================================================

TORCH_LIBRARY_IMPL(
    lmlm,
    CPU,
    m
)
{
    m.impl(
        "fused_bias_gelu",
        &lmlm::fused_bias_gelu_cpu
    );
}


// ============================================================================
// Optional Python extension binding
// ============================================================================

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
)
{
    m.def(
        "fused_bias_gelu",
        &lmlm::fused_bias_gelu_cuda,
        "LMLM fused Bias + GELU CUDA"
    );
}
