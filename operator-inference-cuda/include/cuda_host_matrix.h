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

	/// <summary>
	/// Default constructor creates placeholder
	/// </summary>
	/// <returns></returns>
	cuda_host_matrix() : _m(0), _n(0), data() {}

	// Construct empty matrix
	cuda_host_matrix(size_t m, size_t n, MatrixType matrixType = MatrixType::CM_DENSE);
	
	// copy and assignment constructors
	cuda_host_matrix(const cuda_host_matrix& matrix);
	cuda_host_matrix& operator=(const cuda_host_matrix& matrix);

	// Memory management utilities
	void copy_to_gpu_memory(const cuda_gpu_matrix& gpuMatrix) const;
	void copy_to_host_memory(const cuda_gpu_matrix& gpuMatrix);

	// Prints matrix contents
	void print();

	// Indexing operators
	double& operator()(int row, int col);
	matrix_proxy_t operator[](size_t row);

	// Get size information
	std::pair<size_t, size_t> getMatrixSize() const;

	/// <summary>
	/// Returns a C style pointer to the underlying contents of the GPU matrix.
	///		Data is stored in zero indexed column major format
	///	DANGEROUS: do NOT use this method for memory allocation/deallocation
	/// </summary>
	/// <returns></returns>
	double* c_ptr() { return data.get(); }

	/// <summary>
	/// Number of rows - A has dimension M x N
	/// </summary>
	/// <returns></returns>
	inline size_t M() const { return this->_m; }

	/// <summary>
	/// Number of columns - A has dimension M x N
	/// </summary>
	/// <returns></returns>
	inline size_t N() const { return this->_n; }


protected:
	std::shared_ptr<double> data;
	size_t _m;
	size_t _n;
	MatrixType matrixType;
};

#endif OPINF_CUDAMATRIX_H

