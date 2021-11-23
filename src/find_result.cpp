#include "find_result.h"

std::mutex _i_result_mutex;

#pragma region Private Functions

/// <summary>
/// Processing logic for each value that processed in find()
/// </summary>
void _find_result_job(double* buffer, size_t i, double percentil_value, bool* first_set,
	size_t* first_occurance_index, size_t* last_occurance_index)
{
	double v = buffer[i];
	//DEBUG_MSG("-> " << v << std::endl);

	if (utils::is_double_valid(v) && v == percentil_value)
	{
		if (!*first_set)
		{
			*first_occurance_index = i;
			*first_set = true;
		}
		*last_occurance_index = i;
	}
}

#pragma endregion

void worker::result::find(std::ifstream* file, size_t* fsize, double percentil_value,
	size_t* first_occurance_index, size_t* last_occurance_index)
{
	size_t fi; // data file iterator
	size_t fi_fsize_remaining; // data file iterator counter based on remaining file size to read
	std::streamoff fi_seekfrom = 0;

	char* buffer = nullptr;
	size_t buffer_size = 0;

	// Iterate over the segments
	// ... to preserve the memory limit
	fi_fsize_remaining = *fsize;
	for (fi = 0; fi_fsize_remaining > 0; ++fi)
	{
		if (worker::values::get_state()->terminate_process_requested) return;
		worker::values::get_state()->result_search_task = fi;

		// Set seek position
		fi_seekfrom = constants::SEGMENT_PICK_MEMORY_LIMIT * fi;
		// Set buffer
		utils::fi_set_buffer(&buffer, &buffer_size, &fi_fsize_remaining, constants::SEGMENT_PICK_MEMORY_LIMIT);
		// Read data into buffer
		(*file).seekg(fi_seekfrom, std::ios::beg);
		(*file).read(buffer, buffer_size);

		bool first_set = false;

		// If multithread processing...
		if (*worker::values::get_processing_type() == worker::values::ProcessingType::MultiThread || *worker::values::get_processing_type() == worker::values::ProcessingType::OpenCL)
		{
			// Parallel work (job)
			auto work = [&](tbb::blocked_range<size_t> it)
			{
				for (size_t i = it.begin(); i < it.end(); ++i)
				{
					if (worker::values::get_state()->terminate_process_requested) return;

					// Do the job
					const std::lock_guard<std::mutex> lock(_i_result_mutex);
					_find_result_job((double*)buffer, i, percentil_value, &first_set, first_occurance_index, last_occurance_index);
				}
			};

			// Process buffer chunks in parallel
			tbb::parallel_for(tbb::blocked_range<std::size_t>(0, buffer_size / sizeof(double)), work);
		}
		// Otherwise, rest processing types...
		else
		{
			// Iterate over the segments
			// ... to preserve the memory limit
			for (size_t i = 0; i < buffer_size / sizeof(double); ++i)
			{
				if (worker::values::get_state()->terminate_process_requested) return;

				// Do the job
				_find_result_job((double*)buffer, i, percentil_value, &first_set, first_occurance_index, last_occurance_index);
			}
		}
	}

	// Free the last buffer once done
	utils::fi_try_free_buffer(&buffer);
}