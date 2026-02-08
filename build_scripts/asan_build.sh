#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Clean previous build artifacts
# ------------------------------------------------------------
rm -rf build dist *.egg-info

# ------------------------------------------------------------
# Verify toolchain
# ------------------------------------------------------------
which gcc
gcc --version
which ld
ld --version

# ------------------------------------------------------------
# Force setup.py to use the same GCC/G++
# ------------------------------------------------------------
export CC="$(which gcc)"
export CXX="$(which g++)"

# ------------------------------------------------------------
# Enable ASAN instrumentation (compile-time only)
# IMPORTANT: do NOT preload ASAN or enable leak detection here
# ------------------------------------------------------------
export DEBUG_BUILD=1
export CFLAGS="-fsanitize=address -g -O1 -fno-omit-frame-pointer"
export CXXFLAGS="-fsanitize=address -g -O1 -fno-omit-frame-pointer"
export LDFLAGS="-fsanitize=address"

# ------------------------------------------------------------
# Disable LeakSanitizer during the BUILD
# (gcc, ld, cmake, python all leak by design)
# ------------------------------------------------------------
export ASAN_OPTIONS="detect_leaks=0:abort_on_error=1"
export LSAN_OPTIONS="detect_leaks=0"

# Make absolutely sure we are not preloading ASAN here
unset LD_PRELOAD

# ------------------------------------------------------------
# Build wheel
# ------------------------------------------------------------
python setup.py bdist_wheel
