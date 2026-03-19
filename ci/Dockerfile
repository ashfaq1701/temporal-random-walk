FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Ensuring ABI compatibility with the project
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
ENV CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libtbb-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        unzip \
        zip && \
    rm -rf /var/lib/apt/lists/*

# Install vcpkg and gtest
RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg && \
    /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics && \
    /opt/vcpkg/vcpkg install gtest:x64-linux

ENV VCPKG_ROOT="/opt/vcpkg"
ENV CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
ENV PATH="$VCPKG_ROOT:$PATH"

# Create a persistent virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip once
RUN pip install --upgrade pip

# Install Python dependencies (cached layer)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /project

CMD ["/bin/bash", "/project/ci/run_cpu_tests.sh"]
