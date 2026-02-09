#!/bin/bash
set -e
echo "Starting build process..."
# Directory to store repaired wheels
mkdir -p /project/wheelhouse
# Python versions to build for
PYTHON_VERSIONS=("python3.8" "python3.9" "3.10" "python3.11")
# Set vcpkg environment variables
export VCPKG_ROOT=/opt/vcpkg
export PATH=$VCPKG_ROOT:$PATH

# Build wheels for each Python version
for PYVER in "${PYTHON_VERSIONS[@]}"; do
    if command -v $PYVER &> /dev/null; then
        echo "=========================================="
        echo "Building and repairing wheel for $PYVER"
        echo "=========================================="

        # Install build dependencies from requirements.txt
        $PYVER -m pip install --upgrade pip
        $PYVER -m pip install -r /project/requirements.txt
        $PYVER -m pip install auditwheel

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

        # More robust Python library finding
        PYTHON_VERSION=$($PYVER -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYTHON_LIBDIR=$($PYVER -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

        # Look for the Python library in various locations (adjusted for Rocky Linux paths)
        PYTHON_LIBRARY=""
        POSSIBLE_LOCATIONS=(
            "$PYTHON_LIBDIR/libpython${PYTHON_VERSION}.so"
            "$PYTHON_LIBDIR/libpython${PYTHON_VERSION}m.so"
            "/usr/lib64/libpython${PYTHON_VERSION}.so"
            "/usr/lib64/libpython${PYTHON_VERSION}m.so"
            "/usr/lib/libpython${PYTHON_VERSION}.so"
            "/usr/lib/libpython${PYTHON_VERSION}m.so"
            "/usr/local/lib/libpython${PYTHON_VERSION}.so"
            "/usr/local/lib/libpython${PYTHON_VERSION}.so.1.0"  # Add this line
            "/usr/local/lib/libpython${PYTHON_VERSION}m.so"
            "/usr/local/lib64/libpython${PYTHON_VERSION}.so"
            "/usr/local/lib64/libpython${PYTHON_VERSION}m.so"
        )

        for loc in "${POSSIBLE_LOCATIONS[@]}"; do
            if [ -f "$loc" ]; then
                PYTHON_LIBRARY="$loc"
                break
            fi
        done

        # If still not found, try to find any matching library
        if [ -z "$PYTHON_LIBRARY" ]; then
            echo "Searching for Python library in common directories..."
            PYTHON_LIBRARY=$(find /usr -name "libpython${PYTHON_VERSION}*.so*" | head -1)
        fi

        # Exit if still not found
        if [ -z "$PYTHON_LIBRARY" ]; then
            echo "ERROR: Could not find Python library for ${PYVER}"
            continue
        fi

        export PYTHON_LIBRARY

        echo "Using Python paths:"
        echo "  PYTHON_EXECUTABLE: $PYTHON_EXECUTABLE"
        echo "  PYTHON_INCLUDE_DIR: $PYTHON_INCLUDE_DIR"
        echo "  PYTHON_LIBRARY: $PYTHON_LIBRARY"

        # Add CUDA and vcpkg paths to CMAKE_PREFIX_PATH
        export CMAKE_PREFIX_PATH="$VCPKG_ROOT/installed/x64-linux:/usr/local/cuda:$CMAKE_PREFIX_PATH"
        # Make sure CUDA is properly found
        export CUDACXX="/usr/local/cuda/bin/nvcc"
        export CUDA_HOME="/usr/local/cuda"
        export CUDAToolkit_ROOT="/usr/local/cuda"

        # Clean previous builds
        rm -rf build *.egg-info

        # Build with CUDA enabled
        $PYVER setup.py bdist_wheel

        # Get Python version for wheel pattern
        PY_VER_SHORT=$(echo $PYTHON_VERSION | tr -d '.')

        # Find the wheel
        WHEEL_FILES=(/project/dist/*-cp${PY_VER_SHORT}-cp${PY_VER_SHORT}-linux*.whl)
        if [ ${#WHEEL_FILES[@]} -gt 0 ] && [ -f "${WHEEL_FILES[0]}" ]; then
            WHEEL="${WHEEL_FILES[0]}"
            echo "Built wheel: $WHEEL"

            # Repair the wheel with the same Python version
            # Rocky Linux 9 is based on RHEL 9, which corresponds to manylinux_2_34_x86_64
            echo "Repairing wheel..."
            $PYVER -m auditwheel repair "$WHEEL" --plat manylinux_2_34_x86_64 -w /project/wheelhouse/

            echo "Wheel repair complete"
        else
            echo "ERROR: No wheel was built for $PYVER"
        fi

        # Clean up build artifacts but leave dist directory intact
        rm -rf build *.egg-info

        echo "Completed processing for $PYVER"
        echo "=========================================="
    else
        echo "Python version $PYVER not found, skipping..."
    fi
done

# Check if any wheels were built successfully
if [ "$(ls -A /project/wheelhouse/*.whl 2>/dev/null)" ]; then
    echo "All wheels built successfully!"
    echo "Wheels are available in /project/wheelhouse"
    # List built wheels
    ls -la /project/wheelhouse/*.whl
else
    echo "No wheels were built or repair process failed."
    echo "Check the build logs for errors."
    exit 1
fi
