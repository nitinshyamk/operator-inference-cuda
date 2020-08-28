#pragma once
#ifndef OPINF_CUDA_RANDOM_H
#define OPINF_CUDA_RANDOM_H

#include "cuda_libraries.h"
#include "cuda_gpu_matrix.h"
#include "cuda_gpu_vector.h"
#include <curand.h>
#include "utilities.h"

class cuda_random
{
public:
	cuda_random(cuda_libraries& cuda_libraries);
	~cuda_random();


	void random_sample_standard_normal(cuda_gpu_matrix& A)
	{
		random_sample_normal(A, 0, 1);
	};

	/// <summary>
	/// Fills the matrix (overwriting old entries)
	///	with randomly sampled entries from the distribution N(mean, stdev)
	/// </summary>
	/// <param name="A"></param>
	void random_sample_normal(cuda_gpu_matrix& A, double mean, double stdev);

	/// <summary>
	/// Not recommended. Overwrites N * sizeof(double) bytes with N random samples from
	///		N(mean, stdev).
	/// </summary>
	/// <param name="c_ptr"></param>
	/// <param name="n"></param>
	/// <param name="mean"></param>
	/// <param name="stdev"></param>
	void random_sample_normal(double* c_ptr, size_t N, double mean, double stdev);

private:
	cuda_libraries& _libraries;
	curandGenerator_t _generator;
};

#endif OPINF_CUDA_RANDOM_H