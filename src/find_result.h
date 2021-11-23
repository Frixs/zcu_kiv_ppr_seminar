#pragma once

#include <iostream>
#include <fstream>

#include "libs.h"

#include "constants.h"
#include "utils.h"
#include "worker_values.h"

namespace worker
{
	namespace result
	{
		/// <summary>
		/// 3. part of the algorithm - find result values in the data file.
		/// </summary>
		void find(std::ifstream* file, size_t* fsize, double percentil_value,
			size_t* first_occurance_index, size_t* last_occurance_index);
	}
}