#pragma once
#ifndef OPINF_CUDAMATRIX_H
#define OPINF_CUDAMATRIX_H

#include <memory>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdexcept>
#include <exception>
#include <tuple>

#include "cuda_gpu_matrix.h"
#include "cuda_libraries.h"
#include "utilities.h"


class cuda_host_matrix
{
	using const_proxy_t = matrix_bracket_proxy<const cuda_host_matrix, const double>;
	using matrix_proxy_t = matrix_bracket_proxy<cuda_host_matrix, double>;
	friend class cuda_gpu_matrix;



public:
	enum class MatrixType
	{
		CM_SPARSE,
		CM_DENSE
	};

	// Construct empty matrix
	cuda_host_matrix(size_t m, size_t n, MatrixType matrixType = MatrixType::CM_DENSE);
	// Construct from existing array (CPU memory), assuming data is zero indexed by column major format
	cuda_host_matrix(size_t m, size_t n, MatrixType matrixType, double* data); 

	// Memory management utilities
	void copyToGpuMemory(const cuda_gpu_matrix& gpuMatrix) const;
	void copyFromGpuMemory(const cuda_gpu_matrix& gpuMatrix);

	// Prints matrix contents
	void print();

	// Indexing operators
	double& operator()(int row, int col);
	matrix_proxy_t operator[](size_t row);

	// Get size information
	std::pair<size_t, size_t> getMatrixSize() const;

	std::shared_ptr<double> data;
	const size_t M;
	const size_t N;

private:
	MatrixType matrixType;
};

#endif OPINF_CUDAMATRIX_H

