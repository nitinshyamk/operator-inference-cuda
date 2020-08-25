
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_host_matrix.h"
#include "cuda_gpu_matrix.h"
#include "cuda_libraries.h"
#include "ddt.cuh"
#include "gpu_timer.h"
#include "linear_algebra.cuh"

#include <stdio.h>
#include <iostream>

int main()
{
    gpu_timer<true> block_timer("int main()");
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    using std::cout;
    using std::endl;

    cuda_libraries cuLibraries;
    linear_algebra linalg(cuLibraries);
    Ddt<ForwardDifference4> ddt(0.1);

    gpu_timer<false> timer("main timer");

    timer.start();
    cuda_host_matrix A(10, 10, cuda_host_matrix::MatrixType::CM_DENSE);
    for (int r = 0; r < 10 * 10; ++r)
        A.data.get()[r] = r/ 10.0;

    cuda_gpu_matrix Agpu(10, 10);
    A.copyToGpuMemory(Agpu);

    timer.stop();
    timer.elapsed();

    cuda_gpu_matrix ddt_ans = ddt(Agpu);

    timer.start();
    auto Cgpu = linalg.multiply(Agpu, false, Agpu, false);
    cuda_host_matrix C(Cgpu.M(), Cgpu.N(), cuda_host_matrix::MatrixType::CM_DENSE);
    timer.stop();
    timer.elapsed();

    C.copyFromGpuMemory(Cgpu);

    A.print();
    C.print();
    return 0;
}
