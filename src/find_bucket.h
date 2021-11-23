#pragma once

#include <iostream>
#include <fstream>

#include "libs.h"

#include "constants.h"
#include "utils.h"
#include "worker_values.h"

namespace worker
{
	namespace bucket
	{
		/// <summary>
		/// 1. part of the algorithm - find limit upper and lower value to specify range for the final bucket that can be read in limited memory.
		/// </summary>
		void find(std::ifstream* file, size_t* fsize, int percentil,
			size_t* total_values, double* bucket_lower_val, double* bucket_upper_val, size_t* bucket_value_offset, size_t* bucket_total_found);
	}
}