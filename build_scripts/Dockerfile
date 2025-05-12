FROM nvidia/cuda:12.6.0-devel-rockylinux9

ARG PYTHON_VERSIONS="3.8.18 3.9.18 3.10.13 3.11.7"

# Install EPEL and update system
RUN dnf -y install epel-release && \
    dnf -y update && \
    dnf -y --allowerasing install \
        gcc gcc-c++ make automake autoconf \
        git wget curl \
        pkg-config cmake \
        zlib-devel bzip2-devel xz-devel \
        readline-devel libffi-devel sqlite-devel \
        openssl-devel \
        ca-certificates gnupg \
        sudo which \
        dnf-plugins-core \
        libomp-devel \
        tbb tbb-devel && \
    dnf clean all

# Install Python 3.8–3.11 from source
RUN for VERSION in ${PYTHON_VERSIONS}; do \
        cd /usr/src && \
        MAJOR=$(echo $VERSION | cut -d. -f1,2) && \
        wget https://www.python.org/ftp/python/${VERSION}/Python-${VERSION}.tgz && \
        tar xzf Python-${VERSION}.tgz && \
        cd Python-${VERSION} && \
        # Enable shared libraries and set rpath
        ./configure --enable-optimizations --enable-shared \
          LDFLAGS="-Wl,-rpath /usr/local/lib" && \
        make -j$(nproc) && \
        make altinstall && \
        # Create symlinks for libraries
        ln -sf /usr/local/lib/libpython${MAJOR}.so /usr/lib64/ && \
        ldconfig && \
        rm -rf /usr/src/Python-${VERSION}*; \
    done

# Install pip and common tools for all Python versions
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    for py in 3.8 3.9 3.10 3.11; do \
        python${py} /tmp/get-pip.py && \
        python${py} -m pip install --upgrade pip setuptools wheel; \
    done && rm -f /tmp/get-pip.py

# Build and install patchelf from source
RUN git clone https://github.com/NixOS/patchelf.git /tmp/patchelf && \
    cd /tmp/patchelf && \
    ./bootstrap.sh && \
    ./configure && \
    make && \
    make install && \
    rm -rf /tmp/patchelf

# Install vcpkg and gtest
RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg && \
    /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics && \
    /opt/vcpkg/vcpkg install gtest

# Environment setup
ENV VCPKG_ROOT="/opt/vcpkg"
ENV CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
ENV PATH="$VCPKG_ROOT:/root/.local/bin:$PATH"

# Create world-writable /project directory for mounting
RUN mkdir -p /project && chmod -R 0777 /opt/vcpkg /project

# Set working directory
WORKDIR /project

# Default command
CMD ["/project/build_scripts/build_wheels.sh"]
