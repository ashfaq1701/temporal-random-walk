FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID
ARG GROUP_ID

RUN groupadd -g ${GROUP_ID:-1000} user && \
    useradd -u ${USER_ID:-1000} -g ${GROUP_ID:-1000} -m user

# Install Python versions and development tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential cmake \
    git wget curl \
    pkg-config

# Add deadsnakes PPA for multiple Python versions
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

# Install Python versions
RUN apt-get install -y \
    python3.8-dev python3.8-distutils \
    python3.9-dev python3.9-distutils \
    python3.10-dev python3.10-distutils \
    python3.11-dev python3.11-distutils \
    python3.12-dev

# Install pip for each Python version
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py && \
    python3.9 get-pip.py && \
    python3.10 get-pip.py && \
    python3.11 get-pip.py && \
    python3.12 get-pip.py

# Install vcpkg
RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg && \
    cd /opt/vcpkg && \
    ./bootstrap-vcpkg.sh -disableMetrics

# Set up vcpkg environment
ENV VCPKG_ROOT="/opt/vcpkg"
ENV CMAKE_TOOLCHAIN_FILE="/opt/vcpkg/scripts/buildsystems/vcpkg.cmake"
ENV PATH="${VCPKG_ROOT}:${PATH}"

# Install just the needed vcpkg packages
RUN /opt/vcpkg/vcpkg install gtest

# Create directory for the project
WORKDIR /project

# Default command - run the build script
CMD ["/project/build_scripts/build_wheels.sh"]
