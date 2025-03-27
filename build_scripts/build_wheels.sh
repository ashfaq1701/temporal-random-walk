#!/bin/bash
set -e

echo "Starting build process..."

# Directory to store repaired wheels
mkdir -p /project/wheelhouse

# Python versions to build for
PYTHON_VERSIONS=("/opt/python/cp38-cp38" "/opt/python/cp39-cp39" "/opt/python/cp310-cp310" "/opt/python/cp311-cp311" "/opt/python/cp312-cp312")

# Set vcpkg environment variables
export VCPKG_ROOT=/opt/vcpkg
export PATH=$VCPKG_ROOT:$PATH

# Function to build wheels for a specific Python version
build_wheel() {
    PYBIN=$1
    PYVER=$2
    echo "Building for $PYBIN ($PYVER)..."

    # Install build dependencies from requirements.txt
    $PYBIN/bin/pip install --upgrade pip
    $PYBIN/bin/pip install -r /project/requirements.txt

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
    export PYTHON_EXECUTABLE=$PYBIN/bin/python
    export PYTHON_INCLUDE_DIR=$($PYTHON_EXECUTABLE -c "from sysconfig import get_paths; print(get_paths()['include'])")
    export PYTHON_LIBRARY=$($PYTHON_EXECUTABLE -c "import sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY') or sysconfig.get_config_var('LIBRARY')))")

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
    $PYBIN/bin/python setup.py bdist_wheel

    # Clean up build artifacts but leave dist directory intact
    rm -rf build *.egg-info
}

# Run auditwheel to make wheels manylinux compatible
repair_wheels() {
    echo "Repairing wheels from dist/ to wheelhouse/..."

    # Check if there are any wheels to repair
    if [ ! "$(ls -A /project/dist/*.whl 2>/dev/null)" ]; then
        echo "No wheels found in dist/ directory"
        return 1
    fi

    for wheel in /project/dist/*.whl; do
        echo "Repairing: $wheel"
        $PYBIN/bin/auditwheel repair "$wheel" --plat manylinux_2_34_x86_64 -w /project/wheelhouse/
    done
}

# Build wheels for each Python version
for PYBIN in "${PYTHON_VERSIONS[@]}"; do
    if [ -d "$PYBIN" ]; then
        PYVER=$(basename $PYBIN)
        build_wheel "$PYBIN" "$PYVER"
    else
        echo "Python version $PYBIN not found, skipping..."
    fi
done

# Use the latest Python for auditwheel
PYBIN=/opt/python/cp310-cp310

# Repair wheels to ensure manylinux compatibility
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
