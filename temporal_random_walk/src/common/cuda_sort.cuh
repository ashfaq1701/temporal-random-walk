#ifndef CUDA_SORT_CUH
#define CUDA_SORT_CUH

#ifdef HAS_CUDA

#include <cub/device/device_radix_sort.cuh>

template <typename KeyType, typename ValueType>
void cub_radix_sort_values_by_keys(
    const KeyType* d_keys,  // Input keys
    ValueType* d_values,    // Input/output values
    size_t num_items        // Number of items to sort
)
{
    // Allocate output buffers
    KeyType* d_keys_out = nullptr;
    ValueType* d_values_out = nullptr;
    cudaMalloc(&d_keys_out, sizeof(KeyType) * num_items);
    cudaMalloc(&d_values_out, sizeof(ValueType) * num_items);

    // Allocate temporary device storage for radix sort
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Get required storage size - note the & before temp_storage_bytes
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys,             // Input keys
        d_keys_out,         // Output keys
        d_values,           // Input values
        d_values_out,       // Output values
        num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys,             // Input keys
        d_keys_out,         // Output keys
        d_values,           // Input values
        d_values_out,       // Output values
        num_items);

    // Copy results back to input arrays - only for values
    cudaMemcpy(d_values, d_values_out, sizeof(ValueType) * num_items, cudaMemcpyDeviceToDevice);

    // Free temporary allocations
    cudaFree(d_temp_storage);
    cudaFree(d_keys_out);
    cudaFree(d_values_out);
}

template <typename KeyType, typename ValueType>
void cub_radix_sort_keys_and_values(
    const KeyType* d_keys,  // Input keys
    ValueType* d_values,    // Input/output values
    size_t num_items        // Number of items to sort
)
{
    // Allocate output buffers
    KeyType* d_keys_out = nullptr;
    ValueType* d_values_out = nullptr;
    cudaMalloc(&d_keys_out, sizeof(KeyType) * num_items);
    cudaMalloc(&d_values_out, sizeof(ValueType) * num_items);

    // Allocate temporary device storage for radix sort
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Get required storage size - note the & before temp_storage_bytes
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys,             // Input keys
        d_keys_out,         // Output keys
        d_values,           // Input values
        d_values_out,       // Output values
        num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys,             // Input keys
        d_keys_out,         // Output keys
        d_values,           // Input values
        d_values_out,       // Output values
        num_items);

    // Copy results back to input arrays - only for values
    cudaMemcpy(d_keys, d_keys_out, sizeof(KeyType) * num_items, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_values, d_values_out, sizeof(ValueType) * num_items, cudaMemcpyDeviceToDevice);

    // Free temporary allocations
    cudaFree(d_temp_storage);
    cudaFree(d_keys_out);
    cudaFree(d_values_out);
}

#endif

#endif //CUDA_SORT_CUH