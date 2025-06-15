
function(add_test_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${SRC_FILES})
    set(MOD_SRC_FILES ${SRC_FILES} ${source_file})

    target_compile_features(${target_name} PRIVATE cxx_std_17)

    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX TBB::tbb)

    if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
        set_source_files_properties(
                ${MOD_SRC_FILES}
                PROPERTIES
                LANGUAGE CUDA
        )

        set_target_properties(${target_name} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

        set_property(TARGET ${target_name} PROPERTY
                CUDA_HOST_COMPILER_OPTIONS "${OpenMP_CXX_FLAGS}")

        target_link_libraries(${target_name} PRIVATE
                OpenMP::OpenMP_CXX
                curand
                CUDA::cudart
                CUDA::cuda_driver
                cuda
                cudart
                cudadevrt
        )

        target_include_directories(${target_name} PRIVATE
                ${CMAKE_SOURCE_DIR}/temporal_random_walk/src
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
    else()
        set_source_files_properties(
                ${MOD_SRC_FILES}
                PROPERTIES
                LANGUAGE CXX
        )

        target_include_directories(${target_name} PRIVATE
                ${CMAKE_SOURCE_DIR}/temporal_random_walk/src
        )
    endif()

    set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

add_test_executable(test_run_temporal_random_walk test_run_temporal_random_walk.cpp)
add_test_executable(test_run_temporal_random_walk_2 test_run_temporal_random_walk_2.cpp)
add_test_executable(test_run_temporal_random_walk_3 test_run_temporal_random_walk_3.cpp)
add_test_executable(test_run_exponential_random_picker test_run_exponential_picker.cpp)
add_test_executable(test_large_data_addition test_large_data_addition.cpp)
