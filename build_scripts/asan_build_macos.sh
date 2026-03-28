#!/usr/bin/env bash
set -euo pipefail

rm -rf build dist *.egg-info

which clang
clang --version
which ld

# Use LLVM, not Apple clang
export CC="/opt/homebrew/opt/llvm/bin/clang"
export CXX="/opt/homebrew/opt/llvm/bin/clang++"

# ASAN
export DEBUG_BUILD=1
export CFLAGS="-fsanitize=address -fno-omit-frame-pointer -g -O1"
export CXXFLAGS="$CFLAGS"
export LDFLAGS="-fsanitize=address"

export ASAN_OPTIONS="detect_leaks=0:abort_on_error=1"
export LSAN_OPTIONS="detect_leaks=0"

unset LD_PRELOAD

python setup.py bdist_wheel
