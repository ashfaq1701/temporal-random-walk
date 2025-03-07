
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
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/cuda_common/setup.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/random/IndexBasedRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/random/WeightBasedRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/random/ExponentialIndexRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/random/LinearRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/random/UniformRandomPicker.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/interfaces/IEdgeData.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/interfaces/INodeEdgeIndex.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/interfaces/INodeMapping.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/interfaces/ITemporalGraph.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/cpu/EdgeDataCPU.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/cpu/NodeEdgeIndexCPU.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/cpu/NodeMappingCPU.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/cpu/TemporalGraphCPU.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/cuda/EdgeDataCUDA.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/cuda/NodeEdgeIndexCUDA.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/cuda/NodeMappingCUDA.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/cuda/TemporalGraphCUDA.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/proxies/EdgeData.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/proxies/NodeEdgeIndex.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/proxies/NodeMapping.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/proxies/TemporalGraph.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/core/TemporalRandomWalkCPU.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/core/TemporalRandomWalkCUDA.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/core/TemporalRandomWalk.cu
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
