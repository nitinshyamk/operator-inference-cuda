#include "test_utilities.h"
bool 
test_utilities::is_approx_equal(cuda_host_matrix& A, cuda_host_matrix& B, double epsilon)
{
	if (A.M() != B.M() || A.N() != B.N())
		return false;

	for (size_t r = 0; r < A.M(); ++r)
	{
		for (size_t c = 0; c < A.N(); ++c)
		{
			if (abs(A[r][c] - B[r][c]) > epsilon)
				return false;
		}
	}
	return true;
}
bool
test_utilities::is_approx_equal(double a, double b, double epsilon)
{
	return abs(a - b) <= epsilon;
}


bool
test_utilities::file_exists(const std::string& fname)
{
	struct stat buf;
	return stat(fname.c_str(), &buf) != -1;
}


cuda_host_matrix 
test_utilities::read_matrix_from_file(std::string fname)
{
	std::ifstream file(fname);
	size_t m, n;
	file >> m >> n;

	cuda_host_matrix ans(m, n);
	for (size_t r = 0; r < m; ++r)
	{
		for (size_t c = 0; c < n; ++c)
		{
			file >> ans[r][c];
		}
	}
	file.close();

	return ans;
}

void test_utilities::write_matrix_to_file(cuda_host_matrix& A, std::string fname)
{
	std::ofstream file(fname);
	file << A.M() << " " << A.N() << "\n";

	for (size_t r = 0; r < A.M(); ++r)
	{
		for (size_t c = 0; c < A.N(); ++c)
		{
			file << A[r][c] << " ";
		}
		file << "\n";
	}
	file << std::endl;
}

std::string 
test_utilities::get_baselines_directory()
{
	return std::string(SOURCE_CODE_DIR) + "/tests/baselines";
}


bool
test_utilities::check_baseline(cuda_host_matrix& A, std::string fname)
{
	if (!test_utilities::file_exists(fname))
	{
		test_utilities::write_matrix_to_file(A, fname);
		std::cout << "WARNING: no baselines found. Writing to file " << fname << std::endl;
	}

	auto baseline = test_utilities::read_matrix_from_file(fname);
	return test_utilities::is_approx_equal(baseline, A);

}

