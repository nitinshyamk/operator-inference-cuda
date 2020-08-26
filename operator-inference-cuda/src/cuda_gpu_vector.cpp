#include "../include/cuda_gpu_vector.h"

cuda_gpu_vector::cuda_gpu_vector(size_t n) : cuda_gpu_matrix(n, 1) {};

cuda_host_vector::cuda_host_vector(size_t n) : cuda_host_matrix(n, 1) {};

double& cuda_host_vector::operator[](size_t i)
{
	return (cuda_host_matrix::c_ptr())[i];
};