#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_DIR="${BUILD_DIR:-build-ci-cpu}"
VENV_DIR="${VENV_DIR:-.venv-ci}"
TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE:-/opt/vcpkg/scripts/buildsystems/vcpkg.cmake}"

echo "Using Python interpreter: ${PYTHON_BIN}"
echo "Using build directory: ${BUILD_DIR}"
echo "Using virtual environment: ${VENV_DIR}"
echo "Using CMake toolchain file: ${TOOLCHAIN_FILE}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install -r requirements.txt

cmake -S . -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DHAS_CUDA=OFF \
    -DPython3_EXECUTABLE="$(command -v python)"
cmake --build "${BUILD_DIR}" --target test_temporal_random_walk --parallel
"${BUILD_DIR}/temporal_random_walk/test/test_temporal_random_walk"

CMAKE_ARGS="-DHAS_CUDA=OFF" python -m pip install .
python -m pytest py_tests
