#include "find_percentil.h"

std::mutex _i_percentil_mutex;

#pragma region Private Functions

/// <summary>
/// Processing logic for each value that processed in find() specific for OpenCL
/// </summary>
std::string _find_percentil_job()
{
	std::string code = R"CLC(
		__kernel void run(__global double* data, 
			const double h, const double l, const double inf_val)
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

			if (normal)
			{
				// Check the limits
				if (v >= l && v <= h)
				{
					data[i] = v;
				}
				else
				{
					data[i] = inf_val;
				}
			}
			else
			{
				data[i] = inf_val;
			}
		}
	)CLC";
	return code;
}

/// <summary>
/// Processing logic for each value that processed in find()
/// </summary>
void _find_percentil_job(double* buffer, size_t i,
	std::vector<double>* percentil_bucket, size_t* iv, double bucket_lower_val, double bucket_upper_val)
{
	double v = buffer[i];
	//DEBUG_MSG("-> " << v << std::endl);

	if (utils::is_double_valid(v))
	{
		// Check the limits
		if (v >= bucket_lower_val && v <= bucket_upper_val)
		{
			(*percentil_bucket)[*iv] = v;
			(*iv)++;
		}
	}
}

#pragma endregion

void worker::percentil::find(std::ifstream* file, size_t* fsize, size_t total_values, int percentil, double bucket_lower_val, double bucket_upper_val, size_t bucket_value_offset, size_t bucket_total_found,
	double* percentil_value)
{
	size_t fi; // data file iterator
	size_t fi_fsize_remaining; // data file iterator counter based on remaining file size to read
	std::streamoff fi_seekfrom = 0;

	char* buffer = nullptr;
	size_t buffer_size = 0;

	size_t percentil_pos = (size_t)floor(percentil * total_values / 100.0); // get percentil number position relative to the entire (valid) data sequence
	size_t percentil_bucket_idx = percentil_pos - bucket_value_offset; //- 1;
	//if (percentil_bucket_idx > 0) percentil_bucket_idx -= 1;

	std::vector<double> percentil_bucket(bucket_total_found);
	size_t iv = 0;

	// Iterate over the segments
	// ... to preserve the memory limit
	fi_fsize_remaining = *fsize;
	for (fi = 0; fi_fsize_remaining > 0; ++fi)
	{
		if (worker::values::get_state()->terminate_process_requested) return;
		worker::values::get_state()->percentil_search_task = fi;

		// Set seek position
		fi_seekfrom = constants::SEGMENT_PICK_MEMORY_LIMIT * fi;
		// Set buffer
		utils::fi_set_buffer(&buffer, &buffer_size, &fi_fsize_remaining, constants::SEGMENT_PICK_MEMORY_LIMIT);
		// Read data into buffer
		(*file).seekg(fi_seekfrom, std::ios::beg);
		(*file).read(buffer, buffer_size);

		// If OpenCL processing...
		if (*worker::values::get_processing_type() == worker::values::ProcessingType::OpenCL)
		{
			auto program = utils::cl_create_program(_find_percentil_job(), *worker::values::get_processing_type_value());
			auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
			auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
			auto& device = devices.front();

			cl_int error = 0;

			cl::Buffer cl_buf_buffer_vals(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size, (double*)buffer, &error); utils::cl_track_error_code(error, 2);

			cl::Kernel kernel(program, "run");
			error = kernel.setArg(0, cl_buf_buffer_vals); utils::cl_track_error_code(error, 2);
			error = kernel.setArg(1, bucket_upper_val); utils::cl_track_error_code(error, 2);
			error = kernel.setArg(2, bucket_lower_val); utils::cl_track_error_code(error, 2);
			error = kernel.setArg(3, std::numeric_limits<double>::infinity()); utils::cl_track_error_code(error, 2);

			cl::CommandQueue queue(context, device);
			error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(buffer_size / sizeof(double))); utils::cl_track_error_code(error, 2);

			error = queue.enqueueReadBuffer(cl_buf_buffer_vals, CL_TRUE, 0, buffer_size, (double*)buffer); utils::cl_track_error_code(error, 2);
			error = cl::finish(); utils::cl_track_error_code(error, 2);

			// Finalize computation
			for (size_t i = 0; i < buffer_size / sizeof(double); ++i)
			{
				double v = ((double*)buffer)[i];
				if (v < std::numeric_limits<double>::infinity())
				{
					percentil_bucket[iv] = v;
					iv++;
				}
			}
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
					const std::lock_guard<std::mutex> lock(_i_percentil_mutex);
					_find_percentil_job((double*)buffer, i, &percentil_bucket, &iv, bucket_lower_val, bucket_upper_val);
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
				_find_percentil_job((double*)buffer, i, &percentil_bucket, &iv, bucket_lower_val, bucket_upper_val);
			}
		}
	}

	// Free the last buffer once done
	utils::fi_try_free_buffer(&buffer);

	worker::values::get_state()->percentil_search_done = true;
	if (worker::values::get_state()->terminate_process_requested) return;
	// Quick select (sort)
	if (percentil_bucket.size() == percentil_bucket_idx) percentil_bucket_idx -= 1;
	auto m = percentil_bucket.begin() + percentil_bucket_idx;
	std::nth_element(percentil_bucket.begin(), m, percentil_bucket.end());
	worker::values::get_state()->waiting_for_percentil_pickup = true;
	if (worker::values::get_state()->terminate_process_requested) return;

	// DEBUG MESSAGES
	DEBUG_MSG("percentil_pos = " << percentil_pos << std::endl);
	DEBUG_MSG("percentil_bucket_idx = " << percentil_bucket_idx << std::endl);
	DEBUG_MSG("bucket_upper_val = " << bucket_upper_val << std::endl);
	DEBUG_MSG("bucket_lower_val = " << bucket_lower_val << std::endl);
	/*DEBUG_MSG("v= {");
	for (double i : percentil_bucket)
		DEBUG_MSG(i << ", ");
	DEBUG_MSG("}\n";)*/
	DEBUG_MSG("PERCENTIL : " << percentil << std::endl);

	// Set the value
	*percentil_value = percentil_bucket[percentil_bucket_idx];
}