FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libtbb-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv && \
    git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg && \
    /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics && \
    /opt/vcpkg/vcpkg install gtest && \
    rm -rf /var/lib/apt/lists/*

ENV VCPKG_ROOT="/opt/vcpkg"
ENV CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
ENV PATH="$VCPKG_ROOT:$PATH"

WORKDIR /project

CMD ["/bin/bash", "/project/ci/run_cpu_tests.sh"]
