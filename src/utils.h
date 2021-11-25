#pragma once

#include <iostream>
#include <cstdlib>
#include<stdio.h>
#include <vector>

#include "mymem.h"
#include "constants.h"

#include "libs.h"

// Define debug mode
#define DEBUG
// Define debug print function
#ifdef DEBUG
#define DEBUG_MSG(str) do { std::cout << str; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif

namespace utils
{
	/// <summary>
	/// Set default state of cout
	/// </summary>
	void cout_toggle_set_default(bool def);

	/// <summary>
	/// Toggle cout output to turn on or off
	/// </summary>
	void cout_toggle(bool toggle);

	/// <summary>
	/// Set cout to default state
	/// </summary>
	void cout_toggle_to_default();

	/// <summary>
	/// Frees buffer, if any.
	/// </summary>
	void fi_try_free_buffer(char** buffer);

	/// <summary>
	/// Sets buffer based on iterator going through the data file (it frees the memory if the buffer is not nullptr).
	/// </summary>
	/// <param name="buffer">The buffer</param>
	/// <param name="buffer_size">Size of the set buffer.</param>
	/// <param name="fi_fsize_remaining">Iterator remaining file size to go through based on which is set buffer size.</param>
	/// <param name="memory_limit">Memory limit.</param>
	void fi_set_buffer(char** buffer, size_t* buffer_size, size_t* fi_fsize_remaining, unsigned int memory_limit);

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
