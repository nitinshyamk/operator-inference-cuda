# CMakeList.txt : CMake project for operator-inference-cuda, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.9 FATAL_ERROR)

project (opinf_cuda
	LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)
enable_language(CUDA)
include(CTest)

file(GLOB EXE_CPP_HEADERS *.h)
file(GLOB EXE_CUDA_HEADERS *.cuh)
file(GLOB EXE_CPP_SRC *.cpp)
file(GLOB EXE_CUDA_SRC *.cu)

# Add source to this project's executable.
add_executable (opinf_cuda ${EXE_CPP_HEADERS} ${EXE_CUDA_HEADERS} ${EXE_CPP_SRC} ${EXE_CUDA_SRC})

target_include_directories(opinf_cuda PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(opinf_cuda
	cudart_static
	cudadevrt
	cublas
	cublasLt
	cusolver
	cusparse )

# TODO: RUNTIME_OUTPUT_DIRECTORY documentation and set

# TODO: Add tests and install targets if needed.

#if(BUILD_TESTING)

#endif()
