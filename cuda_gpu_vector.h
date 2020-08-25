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
	cuda_host_vector(size_t n);
	double& operator[](size_t i);
};

class cuda_gpu_vector : public cuda_gpu_matrix
{
public:
	cuda_gpu_vector(size_t n);
};

#endif OPINF_CUDAVECTOR_H

