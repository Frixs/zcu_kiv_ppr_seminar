#pragma once

#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <chrono>

#include "libs.h"

#include "constants.h"
#include "utils.h"
#include "find_bucket.h"
#include "find_percentil.h"
#include "find_result.h"
#include "worker_values.h"
#include "worker.h"

namespace worker
{
	/// <summary>
	/// Run the worker process
	/// </summary>
	void run(std::string filePath, int percentil);
}