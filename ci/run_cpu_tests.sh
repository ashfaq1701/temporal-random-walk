#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_DIR="${BUILD_DIR:-build-ci-cpu}"

echo "Using Python interpreter: ${PYTHON_BIN}"
echo "Using build directory: ${BUILD_DIR}"

"${PYTHON_BIN}" -m pip install --break-system-packages --upgrade pip
"${PYTHON_BIN}" -m pip install --break-system-packages -r requirements.txt

cmake -S . -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF
cmake --build "${BUILD_DIR}" --target test_temporal_random_walk --parallel
ctest --test-dir "${BUILD_DIR}" --output-on-failure -R '^test_temporal_random_walk$'

CMAKE_ARGS="-DHAS_CUDA=OFF" "${PYTHON_BIN}" -m pip install --break-system-packages .
"${PYTHON_BIN}" -m pytest py_tests
