#pragma once

#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <chrono>
#include "mymem.h"
#include "constants.h"
#include "utils.h"

namespace worker
{
	enum class ProcessingType
	{
		SingleThread,
		MultiThread,
		OpenCL
	};

	/// <summary>
	/// 
	/// </summary>
	class State
	{
	public:
		bool terminate_process_requested = false; // if termination is requested
		bool recover_requested = false; // if the process should try to recover (try again)
		bool terminated = true; // if the process is not running (terminated)

		bool file_loaded = false; // if initially file loaded successfully
		
		size_t analyzing_task = 0; // total tasks done (currently) in the analyzing phase
		bool analyzing_done = false; // indication if theanalyzing is done

		size_t bucket_task_sub = 0; // total tasks done (currently) in the currently processed bucket
		size_t bucket_task = 0; // total tasks done (total searched buckets), currently
		bool bucket_found = false; // indication if the final bucket has been found

		size_t percentil_search_task = 0; // currently searched tasks for percentil in the final bucket
		bool percentil_search_done = false;  // indication if the percentil search is done

		bool waiting_for_percentil_pickup = false; // indication the process is waiting for process pickup (std nth element)

		size_t result_search_task = 0; // currently searched tasks for the final result
		bool result_search_done = false;  // indication if the result search is done
	};

	/// <summary>
	/// UNDONE
	/// </summary>
	void run(State state, std::string filePath, int percentil, ProcessingType processingType);
}