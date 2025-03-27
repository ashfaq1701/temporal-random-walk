FROM quay.io/pypa/manylinux_2_34_x86_64

# ------------
# Install cuda
# ------------

ARG VER="12-8"
ARG ARCH="x86_64"

RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
RUN dnf -y install cuda-compiler-${VER}.${ARCH} \
                   cuda-libraries-${VER}.${ARCH} \
                   cuda-libraries-devel-${VER}.${ARCH}
RUN dnf clean all
RUN rm -rf /var/cache/dnf/*
RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/999_nvidia_cuda.conf

# -------------------------
# Set environment variables
# -------------------------

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDADIR=/usr/local/cuda

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
