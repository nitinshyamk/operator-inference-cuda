#pragma once
#ifndef OPINF_CUDAGPUMATRIX_H
#define OPINF_CUDAGPUMATRIX_H

#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_host_matrix.h"
#include "utilities.h"

class cuda_gpu_matrix
{
public:
	cuda_gpu_matrix() : _m(0), _n(0), gpuPtr() {}

	cuda_gpu_matrix(size_t m, size_t n) : _m(m), _n(n), gpuPtr(allocate_on_device<double>(m * n * sizeof(double))) 
	{
		cudaMemset(c_ptr(), 0, sizeof(double) * m * n);
	}

	cuda_gpu_matrix(const cuda_gpu_matrix& matrix) : _m(matrix._m), _n(matrix._n), gpuPtr(matrix.gpuPtr) {}

	cuda_gpu_matrix& operator=(const cuda_gpu_matrix& matrix)
	{
		this->_m = matrix._m;
		this->_n = matrix._n;
		this->gpuPtr = matrix.gpuPtr;
		return *this;
	}

	cuda_gpu_matrix deep_copy() const
	{
		if (this->_m * this->_n < 1)
			return cuda_gpu_matrix();

		cuda_gpu_matrix ans(this->_m, this->_n);
		cudaMemcpy(ans.c_ptr(), this->c_ptr(), (this->_m * this->_n) * sizeof(this->c_ptr()), cudaMemcpyDeviceToDevice);
		return ans;
	}

	/// <summary>
	/// Returns a C style pointer to the underlying contents of the GPU matrix.
	///		Data is stored in zero indexed column major format
	///	DANGEROUS: do NOT use this method for memory allocation/deallocation
	/// </summary>
	/// <returns></returns>
	inline double* c_ptr() const { return this->gpuPtr.get(); }

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

private:
	size_t _m;
	size_t _n;
	std::shared_ptr<double> gpuPtr;
};

#endif OPINF_CUDAGPUMATRIX_H