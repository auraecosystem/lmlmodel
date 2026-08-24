module(
    name = "lmlmodel",
    version = "0.1.0",
)

# Bazel Python rules extension
bazel_dep(name = "rules_python", version = "0.36.0")

# 1. Hermetic Python Toolchain Setup (Python 3.11)
python = use_extension("@rules_python//python/extensions:python.bzl", "python")
python.toolchain(
    is_default = True,
    python_version = "3.11",
)

# 2. Parse Third-Party Dependencies from requirements.txt
pip = use_extension("@rules_python//python/extensions:pip.bzl", "pip")
pip.parse(
    hub_name = "pip",
    python_version = "3.11",
    requirements_lock = "//:requirements.txt",
)

# Expose the generated @pip hub repository to your project workspace
use_repo(pip, "pip")
