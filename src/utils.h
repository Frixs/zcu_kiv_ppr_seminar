#pragma once

#include <iostream>
#include <cstdlib>
#include<stdio.h>
#include <vector>
#include "constants.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#pragma comment(lib, "OpenCL.lib") // evade MS Studio 2019 error LNK2019
#include "CL/cl.h"
#include "CL/cl.hpp"

#define DEBUG
#ifdef DEBUG
#define DEBUG_MSG(str) do { std::cout << str; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif

namespace utils
{
	/// <summary>
	/// Check if double value is valid for processing
	/// </summary>
	bool is_double_valid(double d);

	/// <summary>
	/// Select random item from vector
	/// </summary>
	double select_r_item(std::vector<double> stream, int n);

	/// <summary>
	/// Generate random number from 0 to max.
	/// </summary>
	size_t generate_rand(size_t max);

	/// <summary>
	/// Gets PTR to copy of OpenCL GPU Device
	/// Throws error if something goes wrong.
	/// </summary>
	cl::Device cl_get_gpu_device();

	/// <summary>
	/// Create program for inputing source code to process.
	/// Throws error if something goes wrong.
	/// </summary>
	cl::Program cl_create_program(const std::string& src);
}
