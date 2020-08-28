#pragma once
#ifndef OPINF_TEST_UTILITIES_H
#define OPINF_TEST_UTILITIES_H

#include "../include/cuda_host_matrix.h"
#include "../TestingConfig.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/stat.h>
#include <string>


class test_utilities
{
public:
	/// <summary>
	/// Returns true iff | A[i][j] - B[i][j] |  < epsilon for all i, j
	/// </summary>
	/// <param name="A"></param>
	/// <param name="B"></param>
	/// <param name="epsilon"></param>
	/// <returns></returns>
	static bool is_approx_equal(
		cuda_host_matrix& A,
		cuda_host_matrix& B,
		double epsilon = std::numeric_limits<double>::epsilon());

	/// <summary>
	/// Returns true iff | a - b | < epsilon
	/// </summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <param name="epsilon"></param>
	/// <returns></returns>
	static bool is_approx_equal(double a, double b, double epsilon = std::numeric_limits<double>::epsilon());

	/// <summary>
	/// Reads a matrix from file. assumes each row is a newline, and the first line contains number of rows
	///	followed by number of columns
	/// </summary>
	/// <param name="fname"></param>
	/// <returns></returns>
	static cuda_host_matrix read_matrix_from_file(std::string fname);

	/// <summary>
	/// Writes a matrix to specified file. Writes each row, with elements separated by a space. 
	///	First line contains matrix dimensions M x N
	/// </summary>
	/// <param name="A"></param>
	/// <param name="fname"></param>
	static void write_matrix_to_file(cuda_host_matrix& A, std::string fname);

	/// <summary>
	/// Returns true iff the file with name fname exists
	/// </summary>
	/// <param name="fname"></param>
	/// <returns></returns>
	static bool file_exists(const std::string& fname);

	/// <summary>
	/// Gets the default directory for test baselines. Relies on CMAKE build system integration
	/// </summary>
	/// <returns></returns>
	static std::string get_baselines_directory();

	/// <summary>
	/// Gets the default directory for test inputs. Relies on CMAKE build system integration
	/// </summary>
	static std::string get_inputs_directory();

	/// <summary>
	/// Compare A against the baseline (using the appropriate implementation of is_approx_equal)
	/// </summary>
	/// <param name="A"></param>
	/// <param name="fname"></param>
	/// <returns></returns>
	static bool check_baseline(cuda_host_matrix& A, std::string fname);
};

#endif OPINF_TEST_UTILITIES_H