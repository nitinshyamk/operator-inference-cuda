#include "heat_1d.cuh"

Heat1D::Heat1D(linear_algebra& linalg, OperatorInference<BackwardEuler, false>& opinf) : linalg(linalg), operatorInference(opinf)
{
}

std::pair<cuda_gpu_matrix, cuda_gpu_vector> 
Heat1D::get_heat_matrix_operators(size_t N, double dx, double mu)
{
	if (N < 2)
		throw std::invalid_argument("invalid requested matrix operator size");

	cuda_host_matrix A(N, N);
	double alpha = mu / (dx * dx);
	for (size_t i = 1; i < N - 1; ++i)
	{
		A[i - 1][i] = alpha;
		A[i][i] = -2 * alpha;
		A[i + 1][i] = alpha;
	}
	A[0][0] = -2.0;
	A[1][0] = 1.0;
	A[N - 1][N - 1] = -2.0;
	A[N - 2][N - 1] = 1.0;

	cuda_gpu_matrix Agpu(A.N, A.N);
	A.copyToGpuMemory(Agpu);

	cuda_host_vector B(A.N);
	B[0] = alpha;
	B[N - 1] = alpha;

	cuda_gpu_vector Bgpu(A.N);
	B.copyToGpuMemory(Bgpu);



	return std::make_pair(Agpu, Bgpu);
}

cuda_gpu_matrix
Heat1D::backward_euler(
	cuda_gpu_vector x0,
	cuda_gpu_matrix A,
	cuda_gpu_matrix b,
	double bc,
	size_t timesteps,
	double dt)
{
	// Configure operator (apply pinverse on A with modded diagonal)
	cuda_host_matrix A_host_op(A.M(), A.N());
	A_host_op.copyFromGpuMemory(A);
	for (size_t i = 0; i < A.N(); ++i)
	{
		A_host_op[i][i] -= 1.0 / dt;
	}
	
	cuda_gpu_matrix Aop_partial(A.M(), A.N());
	A_host_op.copyToGpuMemory(Aop_partial);

	cuda_gpu_matrix Aop = linalg.pinv(Aop_partial);


	// Now configure state matrices to record trajectory
	cuda_gpu_vector prev_state(x0.M()), curr_state(x0.M());
	cuda_gpu_matrix trajectory(x0.M(), timesteps);
	auto getColumnStart = [&trajectory](size_t col) -> double*
	{
		return (trajectory.c_ptr() + trajectory.M() * col);
	};
	cudaMemcpy(getColumnStart(0), x0.c_ptr(), x0.M() * sizeof(double), cudaMemcpyDeviceToDevice);

	// Iterate over time, plotting out trajectory
	for (size_t tstep = 2; tstep < timesteps; ++tstep)
	{
		cudaMemcpy(prev_state.c_ptr(), getColumnStart(tstep - 1), trajectory.M() * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(
			getColumnStart(tstep),
			linalg.multiply(
				Aop, false,
				linalg.add(prev_state, -1.0 / dt, bc, 1), false).c_ptr(),
			trajectory.M() * sizeof(double),
			cudaMemcpyDeviceToDevice);
	}

	return trajectory;

}

void
Heat1D::run_reduced_model_sim(size_t reduced_basis_sz, double T, double dt, double dx, double mu)
{
	size_t timesteps = ceil(T / dt);
	size_t mesh = ceil(1 / dx);

	auto initial_condition = cuda_gpu_vector(mesh);
	auto operators = get_heat_matrix_operators(mesh, dx, mu);

	auto trajectory = backward_euler(initial_condition, operators.first, operators.second, 1 /* boundary condition */, timesteps, dt);

	// compute POD basis for reduced model using svd

	svd decomposition = linalg.svd_decomposition(trajectory);

	auto Vr = linalg.subset(
		decomposition.S,
		std::make_pair(0, decomposition.S.M() - 1),
		std::make_pair(0, reduced_basis_sz - 1));

	model_form modelform;
	modelform.Linear = true;
	modelform.Input = true;
	modelform.Bilinear = false;
	modelform.Constant = false;
	modelform.Quadratic = false;

	auto inputs = linalg.get_ones(timesteps);
	auto inferred_operators = operatorInference.infer(modelform, trajectory, inputs, Vr, dt, 0);

}