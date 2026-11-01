load("@rules_python//python:pip.bzl", "compile_pip_requirements")

compile_pip_requirements(
    name = "requirements",
    extra_args = ["--allow-unsafe"],
    requirements_in = "//:requirements.in",
    requirements_txt = "//:requirements_lock.txt",
)
cc_library(
    name = "hopper_pipeline",
    hdrs = ["hopper_pipeline.cuh"],
    visibility = ["//visibility:public"],
    deps = [
        # Include CUDA runtime dependencies if using rules_cuda or local CUDA toolchain
    ],
)
