#include "../include/cuda_random.h"

cuda_random::cuda_random(cuda_libraries& libraries) : _libraries(libraries)
{
	checkCudaStatus<cuda_general_error>(
		curandCreateGenerator(&_generator, CURAND_RNG_PSEUDO_DEFAULT));

	checkCudaStatus<cuda_general_error>(
		curandSetPseudoRandomGeneratorSeed(_generator, 1234ULL));
}

cuda_random::~cuda_random()
{
	checkCudaStatus<cuda_general_error>(curandDestroyGenerator(_generator));
}

void cuda_random::random_sample_normal(double* c_ptr, size_t n, double mean, double stdev)
{
	checkCudaStatus<cuda_general_error>(
		curandGenerateNormalDouble(_generator, c_ptr, n, mean, stdev));
}

void cuda_random::random_sample_normal(cuda_gpu_matrix& A, double mean, double stdev)
{
	cuda_random::random_sample_normal(A.c_ptr(), A.M() * A.N(), mean, stdev);
}

