from pathlib import Path
import torch
from torch.utils.cpp_extension import load

_cuda_module = None

def get_cuda_ops():
    global _cuda_module
    if _cuda_module is None:
        kernel_dir = Path(__file__).parent
        cu_file = kernel_dir / "fused_ops.cu"
        cpp_file = kernel_dir / "bindings.cpp"
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA accelerator requested but no GPU/CUDA device found.")

        # JIT compilation using Ninja / NVCC
        _cuda_module = load(
            name="lmlm_cuda_ops",
            sources=[str(cpp_file), str(cu_file)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False
        )
    return _cuda_module
