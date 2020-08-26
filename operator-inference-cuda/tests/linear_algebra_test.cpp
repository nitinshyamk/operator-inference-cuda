#include "gtest/gtest.h"
#include "test_utilities.h"
#include "../include/cuda_host_matrix.h"
#include "../include/cuda_gpu_matrix.h"
#include "../include/cuda_libraries.h"
#include "../include/linear_algebra.cuh"


class linear_algebra_test : public ::testing::Test
{
public:
	linear_algebra_test() : _cudaLibraries(), _linalg(_cudaLibraries) {}
	~linear_algebra_test() {}

	virtual void SetUp()
	{
		this->allZeroesMat = cuda_host_matrix(10, 20);
		this->matrix1 = getCudaHostMatrix1();
		this->matrix2 = getCudaHostMatrix2();
	}

	virtual void TearDown() {}

	cuda_host_matrix getCudaHostMatrix1()
	{
		cuda_host_matrix ans(10, 20);
		size_t ind = 0;
		for (size_t r = 0; r < ans.M(); ++r)
		{
			for (size_t c = 0; c < ans.N(); ++c)
			{
				ans[r][c] = ++ind;
			}
		}
		return ans;
	}

	cuda_host_matrix getCudaHostMatrix2()
	{
		cuda_host_matrix ans(100, 200);
		long ind = 0;
		for (size_t r = 0; r < ans.M(); ++r)
		{
			for (size_t c = 0; c < ans.N(); ++c)
			{
				ans[r][c] = ++ind * (ind % 2 == 0 ? 1 : -1);
			}
		}
		return ans;
	}

	void ASSERT_MATRIX_ZERO(cuda_host_matrix& matrix)
	{
		for (size_t r = 0; r < matrix.M(); ++r)
		{
			for (size_t c = 0; c < matrix.N(); ++c)
			{
				EXPECT_TRUE(test_utilities::is_approx_equal(matrix[r][c], 0));
			}
		}
	}
	
	cuda_host_matrix allZeroesMat;
	cuda_host_matrix matrix1;
	cuda_host_matrix matrix2;

	cuda_libraries _cudaLibraries;
	linear_algebra _linalg;
};


TEST_F(linear_algebra_test, cuda_matrices_initialized_to_all_zeroes)
{
	ASSERT_MATRIX_ZERO(allZeroesMat);
	cuda_gpu_matrix testans = create_gpu_matrix_from_host(allZeroesMat);
	allZeroesMat[0][0] = 1.0;
	EXPECT_TRUE(test_utilities::is_approx_equal(allZeroesMat[0][0], 1.0));
	allZeroesMat.copy_to_host_memory(testans);
	ASSERT_MATRIX_ZERO(allZeroesMat);
}

TEST_F(linear_algebra_test, cuda_matrices_column_major_zero_indexed_memory)
{
	ASSERT_GE(10, allZeroesMat.M());
	ASSERT_GE(20, allZeroesMat.N());
	allZeroesMat[0][4] = 5.0;
	allZeroesMat[0][0] = 4.0;
	allZeroesMat[3][0] = 8.0;
	allZeroesMat[allZeroesMat.M() - 1][allZeroesMat.N() - 1] = 3.0;
	EXPECT_TRUE(test_utilities::is_approx_equal(allZeroesMat.c_ptr()[0], 4.0));
	EXPECT_TRUE(test_utilities::is_approx_equal(allZeroesMat.c_ptr()[3], 8.0));
	EXPECT_TRUE(test_utilities::is_approx_equal(allZeroesMat.c_ptr()[allZeroesMat.M() * 4], 5.0));
	EXPECT_TRUE(test_utilities::is_approx_equal(allZeroesMat.c_ptr()[(allZeroesMat.N() * allZeroesMat.M() - 1)], 3.0));
}

TEST_F(linear_algebra_test, cuda_matrices_memory_transfer_correct)
{
	cuda_gpu_matrix testmat = create_gpu_matrix_from_host(matrix1);
	cuda_host_matrix testmathost = create_host_matrix_from_gpu(testmat);

	EXPECT_TRUE(test_utilities::is_approx_equal(matrix1, testmathost));
}

TEST_F(linear_algebra_test, linear_algebra_transpose)
{
	cuda_gpu_matrix matrix1Gpu = create_gpu_matrix_from_host(matrix1); (matrix1.M(), matrix1.N());
	cuda_gpu_matrix ansGpu = _linalg.transpose(matrix1Gpu);
	cuda_host_matrix ans = create_host_matrix_from_gpu(ansGpu);

	std::string base_fname = test_utilities::get_baselines_directory() + "/" + ::testing::UnitTest::GetInstance()->current_test_info()->name();
	auto transpose1ansfname = base_fname + "_matrixAns.txt";
	auto transpose2ansfname = base_fname + "_matrixAnsT2.txt";

	cuda_gpu_matrix doubleTranspose = _linalg.transpose(ansGpu);
	cuda_host_matrix ans2 = create_host_matrix_from_gpu(doubleTranspose);

	ASSERT_TRUE(test_utilities::check_baseline(ans, transpose1ansfname));
	ASSERT_TRUE(test_utilities::check_baseline(ans2, transpose2ansfname));
}

TEST_F(linear_algebra_test, linear_algebra_subset)
{
	auto getSubsetForTest = [&](cuda_host_matrix host, std::pair<size_t, size_t> rowrange, std::pair<size_t, size_t> colrange) -> cuda_host_matrix
	{
		cuda_gpu_matrix cpy = create_gpu_matrix_from_host(host);
		cuda_gpu_matrix ans = _linalg.subset(cpy, rowrange, colrange);
		cuda_host_matrix ans_host = create_host_matrix_from_gpu(ans);
		return ans_host;
	};

	std::string base_fname = test_utilities::get_baselines_directory() + "/" + ::testing::UnitTest::GetInstance()->current_test_info()->name();
	using std::make_pair;

	auto identity1 = getSubsetForTest(matrix1, make_pair(0, matrix1.M() - 1), make_pair(0, matrix1.N() - 1));
	ASSERT_TRUE(test_utilities::is_approx_equal(matrix1, identity1));
	auto identity2 = getSubsetForTest(matrix2, make_pair(0, matrix2.M() - 1), make_pair(0, matrix2.N() - 1));
	ASSERT_TRUE(test_utilities::is_approx_equal(matrix2, identity2));
	
	auto apply_and_check_subset = [&](cuda_host_matrix matrix, std::string matrixname) -> void
	{
		auto subset_1 = getSubsetForTest(matrix, make_pair(1, matrix.M() - 2), make_pair(1, matrix.N() - 2));
		ASSERT_TRUE(test_utilities::check_baseline(subset_1, base_fname + "_" + matrixname + "_1.txt"));

		auto subset_2 = getSubsetForTest(matrix, make_pair(0, matrix.M() - 2), make_pair(matrix.N() - 1, matrix.N() - 1));
		ASSERT_TRUE(test_utilities::check_baseline(subset_2, base_fname + "_" + matrixname + "_2.txt"));

		auto subset_3 = getSubsetForTest(matrix, make_pair(matrix.M() - 1, matrix.M() - 1), make_pair(0, matrix.N() - 1));
		ASSERT_TRUE(test_utilities::check_baseline(subset_3, base_fname + "_" + matrixname + "_3.txt"));

		auto subset_4 = getSubsetForTest(matrix, make_pair(matrix.M() - 1, matrix.M() - 1), make_pair(matrix.N() - 1, matrix.N() - 1));
		ASSERT_TRUE(test_utilities::check_baseline(subset_4, base_fname + "_" + matrixname + "_4.txt"));
	};

	apply_and_check_subset(matrix1, "subset1");
	apply_and_check_subset(matrix2, "subset2");

}


