# CMakeList.txt : CMake project for operator-inference-cuda, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.12 FATAL_ERROR)

project (opinf_cuda LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)

enable_language(CUDA)
include(CTest)

file(GLOB OPINF_LIB_HEADERS include/*.h include/*.cuh)
file(GLOB OPINF_LIB_SRC src/*.cpp src/*.cu)

set(CUDA_LIBRARIES cudart_static cudadevrt cublas cublasLt curand cusolver cusparse)

add_library(opinf_cuda_lib STATIC ${OPINF_LIB_HEADERS} ${OPINF_LIB_SRC})
target_include_directories(opinf_cuda_lib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(opinf_cuda_lib ${CUDA_LIBRARIES})

# Add source to this project's executable.
add_executable (opinf_cuda_main src/main.cu)
target_include_directories(opinf_cuda_main PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(opinf_cuda_main opinf_cuda_lib ${CUDA_LIBRARIES})


# TODO: RUNTIME_OUTPUT_DIRECTORY documentation and set

# TODO: Add tests and install targets if needed.

option(PACKAGE_TESTS "Build Tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    include(GoogleTest)
	add_subdirectory("../extern/googletest" "extern/googletest")
    set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
    configure_file(${CMAKE_SOURCE_DIR}/TestingConfig.h.in ${CMAKE_SOURCE_DIR}/TestingConfig.h)
	macro(package_add_test TESTNAME)
        # create an exectuable in which the tests will be stored
        set(TEST_FNAMES ${ARGN})
        list(TRANSFORM TEST_FNAMES PREPEND "tests/")
        add_executable(${TESTNAME} ${TEST_FNAMES} tests/test_utilities.h tests/test_utilities.cpp)
        # link the Google test infrastructure, mocking library, and a default main fuction to
        # the test executable.  Remove g_test_main if writing your own main function.
        target_link_libraries(${TESTNAME} gtest gmock gtest_main opinf_cuda_lib ${CUDA_LIBRARIES})
        # gtest_discover_tests replaces gtest_add_tests,
        # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
        gtest_discover_tests(${TESTNAME}
            # set a working directory so your project root so that you can find test data via paths relative to the project root
            WORKING_DIRECTORY ${PROJECT_DIR}
            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
        )
        set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    endmacro()

    package_add_test(linear_algebra_test "linear_algebra_test.cpp")
endif()

