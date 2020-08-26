#include "../include/cuda_host_matrix.h"


cuda_host_matrix::cuda_host_matrix(size_t m, size_t n, MatrixType matrixType) : _m(m), _n(n), matrixType(matrixType)
{
	// currently matrixType doesn't do anything, but can be used later to inform the relevant storage choices
	// CM_DENSE matrix storage
	size_t sz = m * n;
	this->data = std::shared_ptr<double>(new double[sz], std::default_delete<double[]>());
	memset(data.get(), 0, sizeof(double) * sz);
};


cuda_host_matrix::cuda_host_matrix(const cuda_host_matrix& matrix) : _m(matrix._m), _n(matrix._n), data(matrix.data), matrixType(matrix.matrixType)
{

}

cuda_host_matrix& 
cuda_host_matrix::operator=(const cuda_host_matrix& matrix)
{
	this->_m = matrix.M();
	this->_n = matrix.N();
	this->data = matrix.data;
	this->matrixType = matrixType;
	return *this;
}

void cuda_host_matrix::copy_to_gpu_memory(const cuda_gpu_matrix& gpuMatrix) const
{
	checkCudaStatus<cuda_memory_error>(
		cublasSetMatrix(
			this->M(),
			this->N(),
			sizeof(double),
			(this->data).get(),
			this->M(),
			gpuMatrix.c_ptr(),
			this->M()));
};

void cuda_host_matrix::copy_to_host_memory(const cuda_gpu_matrix& gpuMatrix)
{
	checkCudaStatus<cuda_memory_error>(
		cublasGetMatrix(
			this->M(),
			this->N(),
			sizeof(double),
			gpuMatrix.c_ptr(),
			this->M(),
			(this->data).get(),
			this->M()));
};

void cuda_host_matrix::print()
{
	print_mat(data.get(), _m, _n);
};

// Basic operators
double& cuda_host_matrix::operator()(int row, int col)
{
	// column major order  is key //
	return (this->data).get()[columnMajorZeroIndex(row, col, this->M(), this->N())];
};

cuda_host_matrix::matrix_proxy_t cuda_host_matrix::operator[](size_t row)
{
	return cuda_host_matrix::matrix_proxy_t(*this, row);
};

std::pair<size_t, size_t> cuda_host_matrix::getMatrixSize() const
{
	return std::make_pair(this->M(), this->N());
};