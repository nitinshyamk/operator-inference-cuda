#pragma once
#ifndef OPINF_CUDAVECTOR_H
#define OPINF_CUDAVECTOR_H

#include <memory>
#include <cublasLt.h>

#include "cuda_host_matrix.h"
#include "cuda_gpu_matrix.h"
#include "cuda_libraries.h"

class cuda_host_vector : public cuda_host_matrix
{
public:
	cuda_host_vector() : cuda_host_matrix() {};
	cuda_host_vector(size_t n) : cuda_host_matrix(n, 1) {};
	cuda_host_vector(const cuda_host_matrix& mat) : cuda_host_matrix(mat)
	{
		if (mat.N() != 1)
			throw std::invalid_argument("cannot convert matrix with multiple columns to vector");
	}

	double& operator[](size_t i) 
	{
		return (cuda_host_matrix::c_ptr())[i];
	};
};

class cuda_gpu_vector : public cuda_gpu_matrix
{
public:
	cuda_gpu_vector() : cuda_gpu_matrix() {};
	cuda_gpu_vector(size_t n) : cuda_gpu_matrix(n, 1) {};
	cuda_gpu_vector(const cuda_gpu_matrix& mat) : cuda_gpu_matrix(mat)
	{
		if (mat.N() != 1)
			throw std::invalid_argument("cannot convert matrix with multiple columns to vector");
	}
};

#endif OPINF_CUDAVECTOR_H

