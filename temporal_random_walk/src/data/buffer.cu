#include "buffer.cuh"

#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdint>

#ifdef HAS_CUDA
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#endif

template <typename T>
void Buffer<T>::fill(const T& value) {
    if (size_ == 0 || data_ == nullptr) return;
#ifdef HAS_CUDA
    if (use_gpu_) {
        thrust::fill_n(thrust::device,
                       thrust::device_pointer_cast(data_),
                       size_, value);
        return;
    }
#endif
    std::fill_n(data_, size_, value);
}

template <typename T>
std::vector<T> Buffer<T>::to_host_vector() const {
    std::vector<T> out(size_);
    if (size_ == 0) return out;
#ifdef HAS_CUDA
    if (use_gpu_) {
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            out.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
#endif
    std::memcpy(out.data(), data_, size_ * sizeof(T));
    return out;
}

template class Buffer<int>;
template class Buffer<int64_t>;
template class Buffer<size_t>;
template class Buffer<float>;
template class Buffer<double>;
