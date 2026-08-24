load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "lmlmodel",
    srcs = [
        "__init__.py",
        "config.py",
        "inference.py",
        "model.py",
    ],
    imports = [".."],
    visibility = ["//visibility:public"],
    deps = [
        # Reference packages directly from the @pip repository hub
        "@pip//numpy",
        "@pip//torch",
        "@pip//pydantic",
    ],
)
