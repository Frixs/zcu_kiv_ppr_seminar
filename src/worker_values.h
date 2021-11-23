#pragma once

#include <mutex>

#include "libs.h"

namespace worker
{
	namespace values
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
			bool total_values_counted; // if initially total (valid) values has been already counted

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

		worker::values::State* get_state();
		worker::values::ProcessingType* get_processing_type();
		void init(worker::values::State* state, worker::values::ProcessingType* processing_type);
	}
}