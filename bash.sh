# Build the core library and fetch all dependencies. automatically
bazel build //lmlmodel:lmlmodel

# Run all unit tests
bazel test //tests/...
# Generate or update requirements_lock.txt from requirements.in
bazel run //:requirements.update

# Verify that requirements_lock.txt is up-to-date with requirements.in (ideal for CI pipelines)
bazel test //:requirements_test
