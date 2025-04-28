cmake_minimum_required(VERSION 3.24)
project(temporal_random_walk)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_STANDARD 17)

# Use directly provided Python paths if available
if(DEFINED PYTHON_EXECUTABLE AND DEFINED PYTHON_INCLUDE_DIR AND DEFINED PYTHON_LIBRARY)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
    set(Python3_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR})
    set(Python3_LIBRARIES ${PYTHON_LIBRARY})
    message(STATUS "Using provided Python paths:")
    message(STATUS "  Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
    message(STATUS "  Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS "  Python3_LIBRARIES: ${Python3_LIBRARIES}")
else()
    # Fall back to automatic detection
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
endif()

# Get pybind11 include path using the correct Python executable
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(f'{pybind11.get_include()}/../share/cmake/pybind11')"
        OUTPUT_VARIABLE PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${PYBIND11_DIR}")

# Get Python version dynamically
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        OUTPUT_VARIABLE DETECTED_PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set pybind11 Python version to match detected version
set(PYBIND11_PYTHON_VERSION ${DETECTED_PYTHON_VERSION} CACHE STRING "Python version used for pybind11")
message(STATUS "Setting pybind11 Python version to: ${PYBIND11_PYTHON_VERSION}")

find_package(pybind11 REQUIRED)

find_package(OpenMP REQUIRED)

option(HAS_CUDA "Enable CUDA support" ON)

if(HAS_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90)
        add_definitions(-DHAS_CUDA)
    else()
        set(HAS_CUDA OFF)
    endif()
endif()

set(SRC_FILES
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/common/setup.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/common/random_gen.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/data/WalkSet.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/edge_data.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/node_edge_index.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/stores/temporal_graph.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/random/pickers.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/core/temporal_random_walk.cu

        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/proxies/EdgeData.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/proxies/NodeEdgeIndex.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/proxies/TemporalGraph.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/proxies/RandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_random_walk/src/proxies/TemporalRandomWalk.cu
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
