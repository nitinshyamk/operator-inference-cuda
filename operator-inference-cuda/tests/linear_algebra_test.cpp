#include "gtest/gtest.h"
#include "test_utilities.h"
#include <memory>
#include "../include/cuda_host_matrix.h"
#include "../include/cuda_gpu_matrix.h"
#include "../include/cuda_libraries.h"
#include "../include/cuda_random.h"
#include "../include/linear_algebra.cuh"


class linear_algebra_test : public ::testing::Test
{
public:
	linear_algebra_test() : _cudaLibraries(), _linalg(_cudaLibraries), _cuda_random(_cudaLibraries) {}
	~linear_algebra_test() {}

	virtual void SetUp()
	{
		this->allZeroesMat = cuda_host_matrix(10, 20);
		this->matrix1 = getCudaHostMatrix1();
		this->matrix2 = getCudaHostMatrix2();
		this->matrix3 = getCudaHostMatrix3();
		this->matrix4 = getCudaHostMatrix4();
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

	cuda_host_matrix getCudaHostMatrix3()
	{
		cuda_host_matrix ans(50, 5);
		for (size_t r = 0; r < ans.M(); ++r)
		{
			for (size_t c = 0; c < ans.N(); ++c)
			{
				ans[r][c] = (double)c + 1.0;
			}
		}
		return ans;
	}

	cuda_host_matrix getCudaHostMatrix4()
	{
		cuda_host_matrix ans(40, 4);
		for (size_t r = 0; r < ans.M(); ++r)
		{
			for (size_t c = 0; c < ans.N(); ++c)
			{
				ans[r][c] = (double)r + 1.0;
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
				ASSERT_DOUBLE_EQ(matrix[r][c], 0);
			}
		}
	}

	std::shared_ptr<size_t> get_lookup_table(size_t N)
	{
		std::shared_ptr<size_t> ans(new size_t[N], std::default_delete<size_t[]>());
		ans.get()[0] = N - 1;
		for (int i = N - 2; i >= 0; --i)
		{
			ans.get()[N - i - 1] = (size_t)(i + 1) + ans.get()[N - i - 2];
		}
		return ans;
	}
	
	cuda_host_matrix allZeroesMat;
	cuda_host_matrix matrix1;
	cuda_host_matrix matrix2;
	cuda_host_matrix matrix3;
	cuda_host_matrix matrix4;
	cuda_gpu_vector gpuVec1;
	cuda_gpu_vector gpuVec2;

	cuda_libraries _cudaLibraries;
	linear_algebra _linalg;
	cuda_random _cuda_random;
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

	ASSERT_THROW(getSubsetForTest(matrix1, make_pair(0, matrix1.M()), make_pair(matrix1.N() - 1, matrix1.N() - 2)), std::invalid_argument);
}

TEST_F(linear_algebra_test, linear_algebra_ones)
{
	cuda_gpu_vector test = _linalg.get_ones(50);
	cuda_host_matrix ans = create_host_matrix_from_gpu(test);

	for (size_t r = 0; r < ans.M(); ++r)
	{
		ASSERT_DOUBLE_EQ(ans[r][0], 1.0);
	}
}

TEST_F(linear_algebra_test, linear_algebra_find_column_maxes)
{
	cuda_gpu_vector test = _linalg.get_ones(50);
	cuda_host_matrix ans = create_host_matrix_from_gpu(test);

	for (size_t r = 0; r < ans.M(); ++r)
	{
		ASSERT_DOUBLE_EQ(ans[r][0], 1.0);
	}
}

TEST_F(linear_algebra_test, linear_algebra_lookup_index)
{
	// 0 2 5
	size_t N = 5;

	auto lookup_and_check = [&](size_t col, size_t N, size_t* tbl, size_t tbl_sz, size_t ans) -> void
	{
		if (N == tbl_sz)
		{
			EXPECT_EQ(ans, lookup_ind(tbl, tbl_sz, col));
		}
	};

	for (size_t N = 1; N < 10; ++N)
	{
		std::shared_ptr<size_t> tbl = get_lookup_table(N);

		auto test_lookup = [&](size_t col, size_t tc_N, size_t ans) -> void
		{
			lookup_and_check(col, tc_N, tbl.get(), N, ans);
		};

		test_lookup(0, N, 0);
		test_lookup(N * N + N - 1, N, N - 1);
		test_lookup(3, 3, 1);
		test_lookup(3, 4, 0);
		test_lookup(1, 2, 0);
		test_lookup(11, 5, 2);
		test_lookup(13, 5, 3);
		test_lookup(14, 6, 2);
		test_lookup(6, 9, 0);
		test_lookup(37, 9, 5);
		test_lookup(44, 9, 8);
	}
}

TEST_F(linear_algebra_test, get_matrix_squared)
{
	cuda_gpu_matrix matrix3gpu = create_gpu_matrix_from_host(matrix3);
	cuda_gpu_matrix ansgpu = _linalg.get_matrix_squared(matrix3gpu);
	cuda_host_matrix ans = create_host_matrix_from_gpu(ansgpu);

	std::string base_fname = test_utilities::get_baselines_directory() + "/" + ::testing::UnitTest::GetInstance()->current_test_info()->name();

	ASSERT_TRUE(test_utilities::check_baseline(ans, base_fname + "_1.txt"));

	cuda_gpu_vector allones = _linalg.get_ones(500);
	cuda_gpu_matrix allonessq = _linalg.get_matrix_squared(allones);
	ASSERT_TRUE(test_utilities::check_baseline(create_host_matrix_from_gpu(allonessq), base_fname + "_2.txt"));
	using std::make_pair;
	
	auto subset = _linalg.subset(matrix3gpu, make_pair(0, matrix3gpu.M() - 1), make_pair(0, 3));
	auto subsetsq = _linalg.get_matrix_squared(subset);
	ASSERT_TRUE(test_utilities::check_baseline(create_host_matrix_from_gpu(subsetsq), base_fname + "_3.txt"));
}

TEST_F(linear_algebra_test, concatenate)
{
	cuda_gpu_matrix matrix3gpu = create_gpu_matrix_from_host(matrix3);
	auto m3concat = _linalg.concatenate(matrix3gpu, matrix3gpu, false);
	auto ans = create_host_matrix_from_gpu(m3concat);

	std::string base_fname = test_utilities::get_baselines_directory() + "/" + ::testing::UnitTest::GetInstance()->current_test_info()->name();
	ASSERT_TRUE(test_utilities::check_baseline(ans, base_fname + "_1.txt"));

	auto m4gpu = create_gpu_matrix_from_host(matrix4);
	auto ans2 = create_host_matrix_from_gpu(_linalg.concatenate(m4gpu, m4gpu, true));
	ASSERT_TRUE(test_utilities::check_baseline(ans2, base_fname + "_2.txt"));

	cuda_gpu_vector allones = _linalg.get_ones(500);
	auto allonesconcat = create_host_matrix_from_gpu(_linalg.concatenate(allones, allones, true));
	ASSERT_TRUE(test_utilities::check_baseline(allonesconcat, base_fname + "_3.txt"));

	auto stackright = create_host_matrix_from_gpu(_linalg.concatenate(allones, allones, false));
	ASSERT_TRUE(test_utilities::check_baseline(stackright, base_fname + "_4.txt"));
}

TEST_F(linear_algebra_test, find_column_maxes)
{
	auto m4gpu = create_gpu_matrix_from_host(matrix4);
	auto m4maxes = _linalg.find_column_abs_maxes(m4gpu);
	auto m4maxans = create_host_matrix_from_gpu(m4maxes);

	std::string base_fname = test_utilities::get_baselines_directory() + "/" + ::testing::UnitTest::GetInstance()->current_test_info()->name();
	ASSERT_TRUE(test_utilities::check_baseline(m4maxans, base_fname + "_1.txt"));


	auto m3maxans = create_host_matrix_from_gpu(_linalg.find_column_abs_maxes(create_gpu_matrix_from_host(matrix3)));
	ASSERT_TRUE(test_utilities::check_baseline(m3maxans, base_fname + "_2.txt"));
}

TEST_F(linear_algebra_test, column_normalize)
{
	auto m4gpu = create_gpu_matrix_from_host(matrix4);
	auto m4maxes = _linalg.find_column_abs_maxes(m4gpu);
	_linalg.column_normalize(m4gpu, m4maxes);
	auto m4host = create_host_matrix_from_gpu(m4gpu);

	std::string base_fname = test_utilities::get_baselines_directory() + "/" + ::testing::UnitTest::GetInstance()->current_test_info()->name();
	ASSERT_TRUE(test_utilities::check_baseline(m4host, base_fname + "_1.txt"));

	auto m3gpu = create_gpu_matrix_from_host(matrix3);
	auto m3maxes = _linalg.find_column_abs_maxes(m3gpu);
	_linalg.column_normalize(m3gpu, m3maxes);
	ASSERT_TRUE(test_utilities::check_baseline(create_host_matrix_from_gpu(m3gpu), base_fname + "_2.txt"));
}

TEST_F(linear_algebra_test, add_vectors)
{
	std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
	std::string base_input = test_utilities::get_inputs_directory() + "/" + test_name;
	std::string base_baseline = test_utilities::get_baselines_directory() + "/" + test_name;

	cuda_host_vector v1 = test_utilities::read_matrix_from_file(base_input + "_1_1.txt");
	cuda_host_vector v2 = test_utilities::read_matrix_from_file(base_input + "_1_2.txt");
	auto v1gpu = create_gpu_matrix_from_host(v1);
	auto v2gpu = create_gpu_matrix_from_host(v2);

	auto ans1 = _linalg.add(v1gpu, 0, v2gpu, 0);
	ASSERT_MATRIX_ZERO(create_host_matrix_from_gpu(ans1));

	auto ans2 = _linalg.add(v1gpu, 0, v2gpu, 1);
	ASSERT_TRUE(test_utilities::is_approx_equal(v2, create_host_matrix_from_gpu(ans2)));

	auto ans3 = _linalg.add(v1gpu, 1, v2gpu, 0);
	ASSERT_TRUE(test_utilities::is_approx_equal(v1, create_host_matrix_from_gpu(ans3)));

	auto ans4 = _linalg.add(v1gpu, 1, v2gpu, 1);
	ASSERT_TRUE(test_utilities::check_baseline(create_host_matrix_from_gpu(ans4), base_baseline + "_4.txt"));

	auto ans5 = _linalg.add(v1gpu, 3, v2gpu, -0.4);
	ASSERT_TRUE(test_utilities::check_baseline(create_host_matrix_from_gpu(ans5), base_baseline + "_5.txt"));
}

TEST_F(linear_algebra_test, multiply_matrices)
{
	std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
	std::string base_baseline = test_utilities::get_baselines_directory() + "/" + test_name;
	std::string base_input = test_utilities::get_inputs_directory() + "/" + test_name;

	auto m1gpu = create_gpu_matrix_from_host(matrix1);
	auto ans = _linalg.multiply(m1gpu, true, m1gpu, false);
	auto anshost = create_host_matrix_from_gpu(ans);
	ASSERT_TRUE(test_utilities::check_baseline(anshost, base_baseline + "_1.txt"));

	auto m1 = create_gpu_matrix_from_host(test_utilities::read_matrix_from_file(base_input + "_1.txt"));
	auto m2 = create_gpu_matrix_from_host(test_utilities::read_matrix_from_file(base_input + "_2.txt"));

	auto ans2 = _linalg.multiply(m1, false, m2, false);
	auto ans2host = create_host_matrix_from_gpu(ans2);
	auto ans3host = create_host_matrix_from_gpu(_linalg.multiply(m2, true, m1, true));
	ASSERT_TRUE(test_utilities::check_baseline(ans2host, base_baseline + "_2.txt"));
	ASSERT_TRUE(test_utilities::check_baseline(ans3host, base_baseline + "_3.txt"));

	ASSERT_THROW(_linalg.multiply(m1, false, m2, true), incompatible_dimensions_error);
	ASSERT_THROW(_linalg.multiply(m1, true, m2, true), incompatible_dimensions_error);
	ASSERT_THROW(_linalg.multiply(m2, false, m1, true), incompatible_dimensions_error);
	ASSERT_THROW(_linalg.multiply(m1, false, m2, true), std::invalid_argument);
	ASSERT_THROW(_linalg.multiply(m1, true, m2, true), std::invalid_argument);
	ASSERT_THROW(_linalg.multiply(m2, false, m1, true), std::invalid_argument);
}

TEST_F(linear_algebra_test, singular_value_decomposition)
{
}