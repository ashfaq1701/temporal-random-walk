pybind11_add_module(_temporal_random_walk
        _temporal_random_walk.cu
        ${SRC_FILES}
)

set(MODULE_SRC_FILES ${SRC_FILES} _temporal_random_walk.cu)

target_compile_features(_temporal_random_walk PRIVATE cxx_std_17)
set_target_properties(_temporal_random_walk PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(_temporal_random_walk PRIVATE OpenMP::OpenMP_CXX TBB::tbb)

if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(
            ${MODULE_SRC_FILES}
            PROPERTIES
            LANGUAGE CUDA
    )

    set_property(TARGET _temporal_random_walk PROPERTY
            CUDA_HOST_COMPILER_OPTIONS "${OpenMP_CXX_FLAGS}")

    target_link_libraries(_temporal_random_walk PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::curand
    )

    # CUDA_ARCHITECTURES must be quoted: CMAKE_CUDA_ARCHITECTURES is a
    # semicolon list (e.g. "75;80;86;89;90"), and without the quotes CMake
    # splits it into multiple positional tokens inside set_target_properties.
    # That silently pairs "75" as the arch value and turns "80;86;89;90" into
    # bogus property/value pairs, so the .so gets built for sm_75 only. On
    # newer GPUs the PTX-JIT fallback then produces CUB kernels that fail
    # with cudaErrorInvalidValue on the dispatch call.
    set_target_properties(_temporal_random_walk PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            CUDA_VISIBILITY_PRESET "hidden"
            CXX_VISIBILITY_PRESET "hidden"
            VISIBILITY_INLINES_HIDDEN ON
    )

    target_include_directories(_temporal_random_walk PRIVATE
            ${CMAKE_SOURCE_DIR}/temporal_random_walk/src
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
else ()
    set_source_files_properties(
            ${MODULE_SRC_FILES}
            PROPERTIES
            LANGUAGE CXX
    )

    target_include_directories(_temporal_random_walk PRIVATE
            ${CMAKE_SOURCE_DIR}/temporal_random_walk/src
    )
endif()
