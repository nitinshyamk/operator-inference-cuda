#pragma once
#ifndef OPINF_UTILITIES_H
#define OPINF_UTILITIES_H

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusolver_common.h>
#include <curand.h>
#include <iostream>
#include <iomanip>
#include <string>

// Utility methods that throw an error if code does not contain success value

template <typename exception_t, typename statusCode_t, int successCode_v>
bool checkCudaStatusT(statusCode_t code)
{
	if (code != statusCode_t(successCode_v))
	{
		std::cout << "CUDA Failure with code " << code << std::endl;
		throw exception_t();
	}
	return true;
};

template <typename exception_t>
bool checkCudaStatus(cublasStatus_t code)
{
	return checkCudaStatusT<exception_t, cublasStatus_t, CUBLAS_STATUS_SUCCESS>(code);
};

template <typename exception_t>
bool checkCudaStatus(cusparseStatus_t code)
{
	return checkCudaStatusT<exception_t, cusparseStatus_t, CUSPARSE_STATUS_SUCCESS>(code);
};

template <typename exception_t>
bool checkCudaStatus(cusolverStatus_t code)
{
	return checkCudaStatusT<exception_t, cusolverStatus_t, CUSOLVER_STATUS_SUCCESS>(code);
};

template <typename exception_t>
bool checkCudaStatus(curandStatus_t code)
{
	return checkCudaStatusT<exception_t, curandStatus_t, CURAND_STATUS_SUCCESS>(code);
};

template <typename exception_t>
bool checkCudaError(cudaError_t code)
{
	return checkCudaStatusT<exception_t, cudaError_t, cudaSuccess>(code);
};

inline __device__ __host__ size_t columnMajorZeroIndex(size_t row, size_t col, size_t M, size_t N)
{
	return col * M + row;
};

// print a matrix (assuming column major and zero indexing) of size M x N.
void print_mat(const double* arr, int M, int N);

template<typename Matrix, typename Result>
class matrix_bracket_proxy
{
public:
	matrix_bracket_proxy(Matrix& A, size_t row) : A(A), row(row) {}
	Result& operator[](size_t col) { return A(row, col); }

private:
	Matrix& A;
	size_t row;
};

/// <summary>
/// helper function allocates memory on GPU and returns a shared_ptr of type allocate_t
///	Size specified must be in bytes, i.e. for a vector of size N, you must specify N * sizeof(vector_t). 
///	N * sizeof(double) , for example.
/// </summary>
/// <typeparam name="allocate_t"></typeparam>
/// <param name="size"></param>
/// <returns>A shared_ptr of template type</returns>
template<typename allocate_t>
std::shared_ptr<allocate_t> allocate_on_device(size_t size_in_bytes)
{
	allocate_t* gpuPtrLocal;
	checkCudaError<cuda_memory_error>(cudaMalloc(reinterpret_cast<void**>(&gpuPtrLocal), size_in_bytes));
	return std::shared_ptr<allocate_t>(
		gpuPtrLocal,
		[](allocate_t* ptr) { checkCudaError<cuda_memory_error>(cudaFree(reinterpret_cast<void*>(ptr))); }
	);
};

class cuda_general_error : public std::runtime_error
{
public:
	cuda_general_error() : runtime_error("General CUDA configuration error") {}
	virtual const char* what() const throw()
	{
		return "General CUDA configuration error";
	}
};


class cuda_memory_error : public std::runtime_error
{
public:
	cuda_memory_error() : runtime_error("CUDA memory allocation or transfer failure") {}
	virtual const char* what() const throw()
	{
		return "CUDA memory allocation or transfer failure";
	}
};

class cublas_matrix_operation_error : public std::runtime_error
{
public:
	cublas_matrix_operation_error() : runtime_error("cuBlas matrix operation failed") {}
	virtual const char* what() const throw()
	{
		return "cuBlas matrix operation failed";
	}
};

class incompatible_dimensions_error : public std::invalid_argument
{
public:
	incompatible_dimensions_error(int Am, int An, int Bm, int Bn) :
		invalid_argument(helper_format(Am, An, Bm, Bn)),
		_phrase(helper_format(Am, An, Bm, Bn)) {}

	virtual const char* what() const throw()
	{
		return _phrase.c_str();
	}

	std::string helper_format(int Am, int An, int Bm, int Bn)
	{
		std::string phrase = "Incompatible matrix dimensions error. Cannot multiply matrices of dimensions ";
		phrase = phrase + std::to_string(Am) + " x " + std::to_string(An) + " and " + std::to_string(Bm) + " x " + std::to_string(Bn);
		return phrase;
	}

private:
	std::string _phrase;
};
#endif OPINF_UTILITIES_H