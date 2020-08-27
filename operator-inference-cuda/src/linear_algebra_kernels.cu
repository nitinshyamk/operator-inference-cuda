#include "../include/linear_algebra_kernels.cuh"

__global__ void transpose_kernel(double* src, double* dest, size_t src_M, size_t src_N)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;

	if (r < src_M && c < src_N) {
		auto srcInd = columnMajorZeroIndex(r, c, src_M, src_N);
		auto destInd = columnMajorZeroIndex(c, r, src_N, src_M);
		dest[destInd] = src[srcInd];
	}
};

__global__ void find_column_abs_maxes_kernel(double* src, double* dest, size_t src_M, size_t src_N)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < src_N)
	{
		size_t dest_ind = columnMajorZeroIndex(0, c, 1, src_N);
		dest[dest_ind] = 1.0;
		for (size_t r = 0; r < src_M; ++r)
		{
			size_t src_ind = columnMajorZeroIndex(r, c, src_M, src_N);
			dest[dest_ind] = fmax(fabs(src[src_ind]), fabs(dest[dest_ind]));
		}
	}
};

__global__ void column_normalize_kernel(double* matrix, double* scaling, size_t mat_M, size_t mat_N)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;

	if (r < mat_M && c < mat_N) {
		auto mat_ind = columnMajorZeroIndex(r, c, mat_M, mat_N);
		auto scale_ind = columnMajorZeroIndex(0, c, 1, mat_N);
		matrix[mat_ind] = matrix[mat_ind] / scaling[scale_ind];
	}
};


__global__ void set_ones_kernel(double* src, size_t N)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < N)
	{
		src[c] = 1.0;
	}
};


// __host__ added for testing
__device__ __host__ size_t lookup_ind(size_t* src, size_t sz, size_t lookup_ind)
{
	size_t start = 0, end = sz - 1;
	while (start < end)
	{
		size_t mid = (start + end) / 2;
		if (src[mid] < lookup_ind)
		{
			start = mid + 1;
		}
		else if (src[mid] > lookup_ind)
		{
			end = mid;
		}
		else {
			return mid;
		}
	}
	return end;
};

__global__ void get_matrix_squared_kernel(double* matrix, size_t M, size_t N, double* matrix_squared, size_t* lookup)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;
	int width = (N * N + N) / 2;
	if (r < M && c < width)
	{
		size_t primary_col_ind = lookup_ind(lookup, N, c);
		size_t secondary_col_ind = c - (primary_col_ind == 0 ? 0 : lookup[primary_col_ind - 1] + 1) + primary_col_ind;

		size_t mat_sq_ind = columnMajorZeroIndex(r, c, M, width),
			primary_ind = columnMajorZeroIndex(r, primary_col_ind, M, N),
			secondary_ind = columnMajorZeroIndex(r, secondary_col_ind, M, N);

		matrix_squared[mat_sq_ind] = matrix[primary_ind] * matrix[secondary_ind];
	}
};

__global__ void invert_rectangular_diagonal_kernel(double* matrix, size_t M, size_t N, double epsilon)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	double* cell = matrix + columnMajorZeroIndex(r, r, M, N);
	*cell = 1.0 / *cell;
};
