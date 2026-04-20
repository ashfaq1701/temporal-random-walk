#include "buffer.cuh"

#include <vector>
#include <cstring>
#include <cstdint>

#include "../common/memory.cuh"

template <typename T>
void Buffer<T>::fill(const T& value) {
    fill_memory(data_, size_, value, use_gpu_);
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
