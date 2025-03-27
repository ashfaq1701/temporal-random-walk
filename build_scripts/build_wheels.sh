#!/bin/bash
set -e
echo "Starting build process..."
# Directory to store repaired wheels
mkdir -p /project/wheelhouse
# Python versions to build for
PYTHON_VERSIONS=("python3.8" "python3.9" "python3.10" "python3.11" "python3.12")
# Set vcpkg environment variables
export VCPKG_ROOT=/opt/vcpkg
export PATH=$VCPKG_ROOT:$PATH
# Function to build wheels for a specific Python version
build_wheel() {
    PYVER=$1
    echo "Building for $PYVER..."
    # Install build dependencies from requirements.txt
    $PYVER -m pip install --upgrade pip
    $PYVER -m pip install -r /project/requirements.txt
    # Build the wheel
    cd /project
    # Set the CXX11 ABI flag for compatibility
    export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
    export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
    # Find actual compilers
    export CC=$(which gcc)
    export CXX=$(which g++)
    echo "Using compilers:"
    echo "  CC: $CC"
    echo "  CXX: $CXX"
    # Set Python paths as environment variables
    export PYTHON_EXECUTABLE=$(which $PYVER)
    export PYTHON_INCLUDE_DIR=$($PYVER -c "from sysconfig import get_paths; print(get_paths()['include'])")
    export PYTHON_LIBRARY=$($PYVER -c "import sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY') or sysconfig.get_config_var('LIBRARY')))")
    echo "Using Python paths:"
    echo "  PYTHON_EXECUTABLE: $PYTHON_EXECUTABLE"
    echo "  PYTHON_INCLUDE_DIR: $PYTHON_INCLUDE_DIR"
    echo "  PYTHON_LIBRARY: $PYTHON_LIBRARY"
    # Add CUDA and vcpkg paths to CMAKE_PREFIX_PATH
    export CMAKE_PREFIX_PATH="$VCPKG_ROOT/installed/x64-linux:/usr/local/cuda:$CMAKE_PREFIX_PATH"
    # Make sure CUDA is properly found
    export CUDACXX="/usr/local/cuda/bin/nvcc"
    export CUDA_HOME="/usr/local/cuda"
    # Build with CUDA enabled
    $PYVER setup.py bdist_wheel
    # Clean up build artifacts but leave dist directory intact
    rm -rf build *.egg-info
}
# Run auditwheel to make wheels compatible
repair_wheels() {
    echo "Repairing wheels from dist/ to wheelhouse/..."
    # Check if there are any wheels to repair
    if [ ! "$(ls -A /project/dist/*.whl 2>/dev/null)" ]; then
        echo "No wheels found in dist/ directory"
        return 1
    fi
    # Install auditwheel if needed
    python3 -m pip install auditwheel
    for wheel in /project/dist/*.whl; do
        echo "Repairing: $wheel"
        auditwheel repair "$wheel" --plat manylinux_2_28_x86_64 -w /project/wheelhouse/
    done
}
# Build wheels for each Python version
for PYVER in "${PYTHON_VERSIONS[@]}"; do
    if command -v $PYVER &> /dev/null; then
        build_wheel "$PYVER"
    else
        echo "Python version $PYVER not found, skipping..."
    fi
done
# Repair wheels to ensure compatibility
repair_wheels
# Check if any wheels were built successfully
if [ "$(ls -A /project/wheelhouse/*.whl 2>/dev/null)" ]; then
    echo "All wheels built successfully!"
    echo "Wheels are available in /project/wheelhouse"
    # List built wheels
    echo "Built wheels:"
    ls -la /project/wheelhouse/*.whl
else
    echo "No wheels were built or repair process failed."
    echo "Check the build logs for errors."
    exit 1
fi
