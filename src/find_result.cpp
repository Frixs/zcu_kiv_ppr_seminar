#include "find_result.h"

std::mutex _i_result_mutex;

#pragma region Private Functions

/// <summary>
/// Processing logic for each value that processed in find() specific for OpenCL
/// </summary>
std::string _find_result_job()
{
	std::string code = R"CLC(
		__kernel void run(__global double* data, 
			const double percentil_value)
		{
			int i = get_global_id(0);

			double v = data[i];

			ulong value_long = as_ulong(v);
			const ulong exp = 0x7FF0000000000000;
			const ulong p_zero = 0x0000000000000000;
			const ulong n_zero = 0x8000000000000000;
			bool inf_or_nan = (value_long & exp) == exp;
			bool sub_or_zero = (~value_long & exp) == exp;
			bool zero = value_long == p_zero || value_long == n_zero;
			bool normal = (!inf_or_nan && !sub_or_zero) || zero;

			if (normal && v == percentil_value)
			{
				data[i] = 1;
			}
			else
			{
				data[i] = 0;
			}
		}
	)CLC";
	return code;
}

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
			*first_set = true;
			*first_occurance_index = i + 1;
		}
		*last_occurance_index = i + 1;
	}
}

#pragma endregion

void worker::result::find(std::ifstream* file, size_t* fsize, double percentil_value,
	size_t* first_occurance_index, size_t* last_occurance_index)
{
	size_t fi; // data file iterator
	size_t fi_fsize_remaining; // data file iterator counter based on remaining file size to read
	std::streamoff fi_seekfrom = 0;
	const unsigned int memory_limit = *worker::values::get_processing_type() == worker::values::ProcessingType::OpenCL
		? constants::SEGMENT_SEARCH_MEMORY_LIMIT_CL : constants::SEGMENT_SEARCH_MEMORY_LIMIT;

	char* buffer = nullptr;
	size_t buffer_size = 0;

	bool first_set = false;
	size_t total_buckets_index = 0;

	// Iterate over the segments
	// ... to preserve the memory limit
	fi_fsize_remaining = *fsize;
	for (fi = 0; fi_fsize_remaining > 0; ++fi)
	{
		if (worker::values::get_state()->terminate_process_requested) return;
		worker::values::get_state()->result_search_task = fi;

		// Set seek position
		fi_seekfrom = memory_limit * fi;
		// Set buffer
		utils::fi_set_buffer(&buffer, &buffer_size, &fi_fsize_remaining, memory_limit);
		// Read data into buffer
		(*file).seekg(fi_seekfrom, std::ios::beg);
		(*file).read(buffer, buffer_size);

		size_t first_occurance_index_local = 0;
		size_t last_occurance_index_local = 0;

		// If OpenCL processing...
		if (*worker::values::get_processing_type() == worker::values::ProcessingType::OpenCL)
		{
			auto program = utils::cl_create_program(_find_result_job());
			auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
			auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
			auto& device = devices.front();

			cl_int error;

			cl::Buffer cl_buf_buffer_vals(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size, (double*)buffer, &error);

			cl::Kernel kernel(program, "run");
			error = kernel.setArg(0, cl_buf_buffer_vals);
			error = kernel.setArg(1, percentil_value);

			cl::CommandQueue queue(context, device);
			error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(buffer_size / sizeof(double)));

			error = queue.enqueueReadBuffer(cl_buf_buffer_vals, CL_TRUE, 0, buffer_size, (double*)buffer);
			cl::finish();

			// Parallel work (job)
			auto work = [&](tbb::blocked_range<size_t> it)
			{
				for (size_t i = it.begin(); i < it.end(); ++i)
				{
					if (worker::values::get_state()->terminate_process_requested) return;

					double v = ((double*)buffer)[i];
					
					if (v > 0)
					{
						const std::lock_guard<std::mutex> lock(_i_result_mutex);
						if (!first_set)
						{
							first_set = true;
							first_occurance_index_local = i + 1;
						}
						last_occurance_index_local = i + 1;
					}
				}
			};

			// Process buffer chunks in parallel
			tbb::parallel_for(tbb::blocked_range<std::size_t>(0, buffer_size / sizeof(double)), work);

			if (first_occurance_index_local > 0)
				*first_occurance_index = first_occurance_index_local + total_buckets_index;
			if (last_occurance_index_local > 0)
				*last_occurance_index = last_occurance_index_local + total_buckets_index;
		}
		// If multithread processing...
		else if (*worker::values::get_processing_type() == worker::values::ProcessingType::MultiThread)
		{
			// Parallel work (job)
			auto work = [&](tbb::blocked_range<size_t> it)
			{
				for (size_t i = it.begin(); i < it.end(); ++i)
				{
					if (worker::values::get_state()->terminate_process_requested) return;

					// Do the job
					const std::lock_guard<std::mutex> lock(_i_result_mutex);
					_find_result_job((double*)buffer, i, percentil_value, &first_set, &first_occurance_index_local, &last_occurance_index_local);
				}
			};

			// Process buffer chunks in parallel
			tbb::parallel_for(tbb::blocked_range<std::size_t>(0, buffer_size / sizeof(double)), work);

			if (first_occurance_index_local > 0)
				*first_occurance_index = first_occurance_index_local + total_buckets_index;
			if (last_occurance_index_local > 0)
				*last_occurance_index = last_occurance_index_local + total_buckets_index;
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
				_find_result_job((double*)buffer, i, percentil_value, &first_set, &first_occurance_index_local, &last_occurance_index_local);
			}

			if (first_occurance_index_local > 0)
				*first_occurance_index = first_occurance_index_local + total_buckets_index;
			if (last_occurance_index_local > 0)
				*last_occurance_index = last_occurance_index_local + total_buckets_index;
		}

		// Increment already processed item count
		total_buckets_index += buffer_size / sizeof(double);
	}

	// Indexing starts at 0, take care of it
	*first_occurance_index -= 1;
	*last_occurance_index -= 1;

	// Free the last buffer once done
	utils::fi_try_free_buffer(&buffer);

	worker::values::get_state()->result_search_done = true;
}