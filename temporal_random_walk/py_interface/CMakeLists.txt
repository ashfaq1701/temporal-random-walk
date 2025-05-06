pybind11_add_module(_temporal_random_walk
        _temporal_random_walk.cu
        ${SRC_FILES}
)

set(MODULE_SRC_FILES ${SRC_FILES} _temporal_random_walk.cu)

target_compile_features(_temporal_random_walk PRIVATE cxx_std_17)
set_target_properties(_temporal_random_walk PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(_temporal_random_walk PRIVATE OpenMP::OpenMP_CXX)

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

    set_target_properties(_temporal_random_walk PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
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
