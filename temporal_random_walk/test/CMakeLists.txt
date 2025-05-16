find_package(GTest CONFIG REQUIRED)

set(TEST_FILES
        test_edge_data.cpp
        test_edge_data_weights.cpp
        test_node_edge_index.cpp
        test_node_edge_index_weights.cpp
        #test_random_picker.cpp
        test_temporal_graph.cpp
        test_temporal_graph_get_node_edge_at.cpp
        test_temporal_graph_weights.cpp
        test_temporal_random_walk.cpp
        test_weight_based_random_picker.cpp
)

set(MODULE_SRC_FILES ${SRC_FILES} ${TEST_FILES})

if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(
            ${MODULE_SRC_FILES}
            PROPERTIES
            LANGUAGE CUDA
    )

    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    set(CUDA_LIBS
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::cublas
            CUDA::nvrtc
            CUDA::curand
            cuda
            cudart
    )

    add_executable(test_temporal_random_walk ${TEST_FILES} ${SRC_FILES})
    set_target_properties(test_temporal_random_walk PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_include_directories(test_temporal_random_walk PRIVATE
            ${CMAKE_SOURCE_DIR}/temporal_random_walk/src
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
else()
    set_source_files_properties(
            ${MODULE_SRC_FILES}
            PROPERTIES
            LANGUAGE CXX
    )

    set(CUDA_OPTIONS "")
    set(CUDA_LIBS "")

    add_executable(test_temporal_random_walk ${TEST_FILES} ${SRC_FILES})

    target_include_directories(test_temporal_random_walk PRIVATE
            ${CMAKE_SOURCE_DIR}/temporal_random_walk/src
    )
endif()

target_compile_features(test_temporal_random_walk PRIVATE cxx_std_17)
target_compile_options(test_temporal_random_walk PRIVATE ${CUDA_OPTIONS})

target_link_libraries(test_temporal_random_walk PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        GTest::gmock_main
        OpenMP::OpenMP_CXX
        TBB::tbb
        ${CUDA_LIBS}
)
