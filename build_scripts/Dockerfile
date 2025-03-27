FROM nvidia/cuda:12.8.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID=1000
ARG GROUP_ID=1000

# Create non-root user with matching UID/GID
RUN groupadd -g ${GROUP_ID} user && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m user && \
    usermod -aG sudo user

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    git wget curl \
    sudo \
    pkg-config \
    apt-transport-https  \
    ca-certificates  \
    automake \
    autoconf \
    gnupg

# Clone the patchelf repository (latest version)
RUN git clone https://github.com/NixOS/patchelf.git /tmp/patchelf

# Build and install patchelf from source
RUN cd /tmp/patchelf && \
    ./bootstrap.sh && \
    ./configure && \
    make && \
    sudo make install

# Clean up
RUN rm -rf /tmp/patchelf

# Add Kitware's APT repository for newer CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake

# Add deadsnakes PPA for Python versions
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

# Install Python versions and distutils
RUN apt-get install -y \
    python3.8-dev python3.8-distutils \
    python3.9-dev python3.9-distutils \
    python3.10-dev python3.10-distutils \
    python3.11-dev python3.11-distutils \
    python3-distutils

# Note: Python 3.12 might not be available in the deadsnakes PPA for Ubuntu 20.04
# Removed Python 3.12 installation

# Install pip for all versions
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py && \
    python3.9 get-pip.py && \
    python3.10 get-pip.py && \
    python3.11 get-pip.py

# Install vcpkg as root
RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg && \
    cd /opt/vcpkg && \
    ./bootstrap-vcpkg.sh -disableMetrics

# Set up vcpkg environment
ENV VCPKG_ROOT="/opt/vcpkg"
ENV CMAKE_TOOLCHAIN_FILE="/opt/vcpkg/scripts/buildsystems/vcpkg.cmake"
ENV PATH="${VCPKG_ROOT}:${PATH}"

# Install gtest via vcpkg
RUN /opt/vcpkg/vcpkg install gtest

# Give ownership of /opt and /project to the non-root user
RUN mkdir -p /project && \
    chown -R ${USER_ID}:${GROUP_ID} /opt/vcpkg /project

# Verify CMake version
RUN cmake --version

# Switch to the non-root user
USER user

# Set working directory
WORKDIR /project

# Default command
CMD ["/project/build_scripts/build_wheels.sh"]
