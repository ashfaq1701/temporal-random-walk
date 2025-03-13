
cmake_minimum_required(VERSION 3.28)
project(temporal_random_walk)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/python")

execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(f'{pybind11.get_include()}/../share/cmake/pybind11')"
        OUTPUT_VARIABLE PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${PYBIND11_DIR}")
find_package(pybind11 REQUIRED)

option(HAS_CUDA "Enable CUDA support" ON)

if(HAS_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES 89)
        add_definitions(-DHAS_CUDA)
    else()
        set(HAS_CUDA OFF)
    endif()
endif()

set(SRC_FILES
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/common/setup.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/edge_data.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/node_edge_index.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/node_mapping.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/temporal_graph.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/random/pickers.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/core/temporal_random_walk.cu
)

if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
    )

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
            -maxrregcount=64           # Limit register usage
            --default-stream per-thread # Better stream handling
            --ptxas-options=-v         # Verbose PTX assembly
            -Xptxas -dlcm=ca          # Cache all operations
    )

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -rdc=true")
    # For debug builds
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

endif()

add_subdirectory(temporal_random_walk/py_interface)
add_subdirectory(temporal_random_walk/test)
add_subdirectory(temporal_random_walk/test_run)
