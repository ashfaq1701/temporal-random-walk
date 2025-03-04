#ifndef TYPES_H
#define TYPES_H

#include "../data/enums.h"

#include <vector>
#ifdef HAS_CUDA
#include <thrust/device_vector.h>
#endif

enum class VectorStorageType {
    STD_VECTOR,
    DEVICE_VECTOR
};

// Base template (default case: CPU only)
template <typename T, GPUUsageMode GPUUsage, typename Enable = void>
struct SelectVectorType;

// Specialization for CPU mode
template <typename T, GPUUsageMode GPUUsage>
struct SelectVectorType<T, GPUUsage, std::enable_if_t<GPUUsage == GPUUsageMode::ON_CPU>> {
    using type = std::vector<T>;

    static constexpr VectorStorageType get_vector_storage_type() {
        return VectorStorageType::STD_VECTOR;
    }
};

#ifdef HAS_CUDA
// Specialization for GPU mode (Device memory)
template <typename T, GPUUsageMode GPUUsage>
struct SelectVectorType<T, GPUUsage, std::enable_if_t<GPUUsage == GPUUsageMode::ON_GPU>> {
    using type = thrust::device_vector<T>;

    static constexpr VectorStorageType get_vector_storage_type() {
        return VectorStorageType::DEVICE_VECTOR;
    }
};
#endif

#endif // TYPES_H
