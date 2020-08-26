#pragma once
#ifndef OPINF_HEAT1D_H
#define OPINF_HEAT1D_H

#include "linear_algebra.cuh"
#include "operator_inference.cuh"

class Heat1D
{
public:
	Heat1D(linear_algebra& linalg, OperatorInference<BackwardEuler, false>& opinf);

	std::pair<cuda_gpu_matrix, cuda_gpu_vector> get_heat_matrix_operators(size_t N, double dx, double mu);

	cuda_gpu_matrix backward_euler(cuda_gpu_vector x0, cuda_gpu_matrix A, cuda_gpu_matrix b, double bc, size_t timesteps, double dt);
	void run_reduced_model_sim(size_t reduced_basis_sz, double T, double dt, double dx, double mu);

private:
	OperatorInference<BackwardEuler, false> operatorInference;
	linear_algebra& linalg;
};

#endif OPINF_HEAT1D_H

