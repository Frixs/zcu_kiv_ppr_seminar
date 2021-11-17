#pragma once

#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <chrono>
#include "mymem.h"
#include "constants.h"
#include "utils.h"

#include "tbb/parallel_for.h"
#include "tbb/combinable.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#pragma comment(lib, "OpenCL.lib") // evade MS Studio 2019 error LNK2019
#include "CL/cl.h"
#include "CL/cl.hpp"

namespace worker
{
	/// <summary>
	/// Processing type of the algorithm
	/// </summary>
	enum class ProcessingType
	{
		SingleThread,
		MultiThread,
		OpenCL
	};

	/// <summary>
	/// State values of the algorithm
	/// </summary>
	class State
	{
	public:
		bool terminated; // if the process is not running (terminated)

		bool terminate_process_requested; // if termination is requested
		bool recovery_requested; // if the process should try to recover (try again)

		bool file_loaded; // if initially file loaded successfully

		size_t analyzing_task; // total tasks done (currently) in the analyzing phase
		bool analyzing_done; // indication if theanalyzing is done

		size_t bucket_task_sub; // total tasks done (currently) in the currently processed bucket
		size_t bucket_task; // total tasks done (total searched buckets), currently
		bool bucket_found; // indication if the final bucket has been found

		size_t percentil_search_task; // currently searched tasks for percentil in the final bucket
		bool percentil_search_done;  // indication if the percentil search is done

		bool waiting_for_percentil_pickup; // indication the process is waiting for process pickup (std nth element)

		size_t result_search_task; // currently searched tasks for the final result
		bool result_search_done;  // indication if the result search is done

		State();
		void set_defaults();
	};

	/// <summary>
	/// Run the worker process
	/// </summary>
	void run(State* state, std::string filePath, int percentil, ProcessingType* processing_type);
}