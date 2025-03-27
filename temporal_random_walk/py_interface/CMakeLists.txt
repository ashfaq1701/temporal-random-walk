add_library(_temporal_random_walk MODULE
        _temporal_random_walk.cpp
        ${SRC_FILES}
)

# Get Python extension suffix (.so, .cpython-*.so, etc.)
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
        OUTPUT_VARIABLE Python3_EXTENSION_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(MODULE_SRC_FILES ${SRC_FILES} _temporal_random_walk.cpp)

set_target_properties(_temporal_random_walk PROPERTIES
        PREFIX ""
        SUFFIX "${Python3_EXTENSION_SUFFIX}"
        POSITION_INDEPENDENT_CODE ON
)

target_compile_features(_temporal_random_walk PRIVATE cxx_std_17)

target_include_directories(_temporal_random_walk PRIVATE
        ${PYBIND11_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src
)

target_link_libraries(_temporal_random_walk PRIVATE
        ${Python3_LIBRARIES}
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
