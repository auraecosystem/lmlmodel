# Build the core library and fetch all dependencies. automatically
bazel build //lmlmodel:lmlmodel

# Run all unit tests
bazel test //tests/...
