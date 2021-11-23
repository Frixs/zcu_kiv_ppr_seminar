#pragma once

#include <iostream>
#include <fstream>

#include "libs.h"

#include "constants.h"
#include "utils.h"
#include "worker_values.h"

namespace worker
{
	namespace percentil
	{
		/// <summary>
		/// 2. part of the algorithm - find percentil value in the data file based on limit upper and lower value.
		/// </summary>
		void find(std::ifstream* file, size_t* fsize, size_t total_values, int percentil, double bucket_lower_val, double bucket_upper_val, size_t bucket_value_offset, size_t bucket_total_found,
			double* percentil_value);
	}
}