pybind11_add_module(_temporal_random_walk
        _temporal_random_walk.cpp
        ${SRC_FILES}
)

set(MODULE_SRC_FILES ${SRC_FILES} _temporal_random_walk.cpp)

target_compile_features(_temporal_random_walk PRIVATE cxx_std_17)
set_target_properties(_temporal_random_walk PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(
            ${MODULE_SRC_FILES}
            PROPERTIES
            LANGUAGE CUDA
    )

    target_link_libraries(_temporal_random_walk PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
    )

    set_target_properties(_temporal_random_walk PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 89
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
