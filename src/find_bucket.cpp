#include "find_bucket.h"

std::mutex _i_bucket_mutex;

#pragma region Private Functions

/// <summary>
/// Processing logic for each value that processed in _process_segment_data() specific for OpenCL
/// </summary>
std::string _process_segment_data_job()
{
	std::string code = R"CLC(
		unsigned int generate_rand(unsigned int i) // 0 .. i
		{
			unsigned int max = -1;
			return (unsigned int)(i * ((double)i) / ((double)max));
		}

		__kernel void run(__global double* data, __global double* outs, __global double* lower_pivot_sample, __global double* upper_pivot_sample, __global double* equal_pivot_sample,
			const unsigned int total_values_counted, const double p, const double h, const double l)
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
					// Greater than pivot...
					if (v > p)
					{
						outs[i] = 2;
						*upper_pivot_sample = v;
						/*
						if ((*highs_local)++ > 0)
						{
							if (i == generate_rand(i)) // 0 .. i
								*upper_pivot_sample = v;
						}
						else
							*upper_pivot_sample = v;
						*/
					}
					// Lower than pivot...
					else if (v < p)
					{
						outs[i] = 1;
						*lower_pivot_sample = v;
					}
					// Otherwise, equal to pivot...
					else
					{
						outs[i] = 3;
						*equal_pivot_sample = v;
					}
				}
			}
		}
	)CLC";
	return code;
}

/// <summary>
/// Processing logic for each value that processed in _process_segment_data()
/// </summary>
void _process_segment_data_job(double* values, size_t i,
	double p, double h, double l,
	size_t* total_values, size_t* lows_local, size_t* highs_local, size_t* equals_local, double* lower_pivot_sample, double* upper_pivot_sample, double* equal_pivot_sample)
{
	double v = values[i];
	//DEBUG_MSG("-> " << v << std::endl);

	if (utils::is_double_valid(v))
	{
		// Count total (valid) values, if not counted yet...
		if (!worker::values::get_state()->total_values_counted)
			(*total_values)++;

		// Check the limits
		if (v >= l && v <= h)
		{
			// Greater than pivot...
			if (v > p)
			{
				// If next number to highs (+ increment highs)...
				if ((*highs_local)++ > 0)
				{
					if (i == utils::generate_rand(i)) // 0 .. i
						*upper_pivot_sample = v;
				}
				// Otherwise, first number to highs...
				else
					*upper_pivot_sample = v;
			}
			// Lower than pivot...
			else if (v < p)
			{
				// If next number to lows (+ increment lows)...
				if ((*lows_local)++ > 0)
				{
					if (i == utils::generate_rand(i)) // 0 .. i
						*lower_pivot_sample = v;
				}
				// Otherwise, first number to lows...
				else
					*lower_pivot_sample = v;
			}
			// Otherwise, equal to pivot...
			else
			{
				// If next number to equals (+ increment equals)...
				if ((*equals_local)++ > 0)
				{
					if (i == utils::generate_rand(i)) // 0 .. i
						*equal_pivot_sample = v;
				}
				// Otherwise, first number to equals...
				else
					*equal_pivot_sample = v;
			}
		}
	}
}

/// <summary>
/// Counts the result from all jobs made in _process_segment_data()
/// </summary>
void _process_segment_data_count(size_t lows_local, size_t highs_local, size_t equals_local, double lower_pivot_sample, double upper_pivot_sample, double equal_pivot_sample,
	size_t* lows, size_t* highs, size_t* equals, std::vector<double>* pivot_lower_samples, std::vector<double>* pivot_upper_samples, std::vector<double>* pivot_equal_samples)
{
	// Select next segment pivot samples
	if (lows_local > 0)
		pivot_lower_samples->push_back(lower_pivot_sample);
	if (highs_local > 0)
		pivot_upper_samples->push_back(upper_pivot_sample);
	if (equals_local > 0)
		pivot_equal_samples->push_back(equal_pivot_sample);

	// Update counters
	*lows += lows_local;
	*highs += highs_local;
	*equals += equals_local;
}

/// <summary>
/// Processes values (segment of the input data based on limited memory / chunk) to get number of lower, greater, and equal values to the inputing pivot based on limit upper and lower values.
/// For each of the selecting classes, it also selects posible pivots.
/// </summary>
void _process_segment_data(double* values, size_t n,
	double p, double h, double l,
	size_t* total_values, size_t* lows, size_t* highs, size_t* equals, std::vector<double>* pivot_lower_samples, std::vector<double>* pivot_upper_samples, std::vector<double>* pivot_equal_samples)
{
	// If OpenCL processing...
	if (*worker::values::get_processing_type() == worker::values::ProcessingType::OpenCL)
	{
		auto program = utils::cl_create_program(_process_segment_data_job());
		auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
		auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		cl_int error;
		size_t lows_local = 0;
		size_t highs_local = 0;
		size_t equals_local = 0;
		double lower_pivot_sample = NAN;
		double upper_pivot_sample = NAN;
		double equal_pivot_sample = NAN;

		cl::Buffer cl_buf_buffer_vals(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, n * sizeof(double), values, &error);
		cl::Buffer cl_buf_buffer_outs(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n * sizeof(double), nullptr, &error);
		cl::Buffer cl_buf_lower_pivot_sample(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(double), nullptr, &error);
		cl::Buffer cl_buf_upper_pivot_sample(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(double), nullptr, &error);
		cl::Buffer cl_buf_equal_pivot_sample(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(double), nullptr, &error);

		cl::Kernel kernel(program, "run");
		error = kernel.setArg(0, cl_buf_buffer_vals);
		error = kernel.setArg(1, cl_buf_buffer_outs);
		error = kernel.setArg(2, cl_buf_lower_pivot_sample);
		error = kernel.setArg(3, cl_buf_upper_pivot_sample);
		error = kernel.setArg(4, cl_buf_equal_pivot_sample);
		error = kernel.setArg(5, worker::values::get_state()->total_values_counted);
		error = kernel.setArg(6, p);
		error = kernel.setArg(7, h);
		error = kernel.setArg(8, l);

		cl::CommandQueue queue(context, device);
		error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n));

		error = queue.enqueueReadBuffer(cl_buf_lower_pivot_sample, CL_TRUE, 0, sizeof(double), &lower_pivot_sample);
		error = queue.enqueueReadBuffer(cl_buf_upper_pivot_sample, CL_TRUE, 0, sizeof(double), &upper_pivot_sample);
		error = queue.enqueueReadBuffer(cl_buf_equal_pivot_sample, CL_TRUE, 0, sizeof(double), &equal_pivot_sample);
		error = queue.enqueueReadBuffer(cl_buf_buffer_outs, CL_TRUE, 0, n * sizeof(double), values);
		cl::finish();

		// Finalize computation
		for (size_t i = 0; i < n; ++i)
		{
			if (values[i] > 0)
			{
				// Count total (valid) values, if not counted yet...
				if (!worker::values::get_state()->total_values_counted)
					(*total_values)++;

				if (values[i] > 2)
					equals_local++;
				else if (values[i] > 1)
					highs_local++;
				else
					lows_local++;
			}
		}

		// Count the job values
		_process_segment_data_count(lows_local, highs_local, equals_local, lower_pivot_sample, upper_pivot_sample, equal_pivot_sample,
			lows, highs, equals, pivot_lower_samples, pivot_upper_samples, pivot_equal_samples);
	}
	// If multithread processing...
	else if (*worker::values::get_processing_type() == worker::values::ProcessingType::MultiThread)
	{
		tbb::combinable<size_t> total_values_comb(0);

		// Parallel work (job)
		auto work = [&](tbb::blocked_range<size_t> it)
		{
			size_t& total_values_local = total_values_comb.local();

			size_t lows_local = 0;
			size_t highs_local = 0;
			size_t equals_local = 0;
			double lower_pivot_sample = 0;
			double upper_pivot_sample = 0;
			double equal_pivot_sample = 0;

			for (size_t i = it.begin(); i < it.end(); ++i)
			{
				if (worker::values::get_state()->terminate_process_requested) return;

				// Do the job
				_process_segment_data_job(values, i,
					p, h, l,
					&total_values_local, &lows_local, &highs_local, &equals_local, &lower_pivot_sample, &upper_pivot_sample, &equal_pivot_sample);
			}

			// Count the job values
			const std::lock_guard<std::mutex> lock(_i_bucket_mutex);
			_process_segment_data_count(lows_local, highs_local, equals_local, lower_pivot_sample, upper_pivot_sample, equal_pivot_sample,
				lows, highs, equals, pivot_lower_samples, pivot_upper_samples, pivot_equal_samples);
		};

		// Process buffer chunks in parallel
		tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n), work);

		// Combine values
		*total_values += total_values_comb.combine(std::plus<>());
	}
	// Otherwise, rest processing types...
	else
	{
		size_t lows_local = 0;
		size_t highs_local = 0;
		size_t equals_local = 0;
		double lower_pivot_sample = 0;
		double upper_pivot_sample = 0;
		double equal_pivot_sample = 0;

		for (size_t i = 0; i < n; ++i)
		{
			if (worker::values::get_state()->terminate_process_requested) return;

			// Do the job
			_process_segment_data_job(values, i,
				p, h, l,
				total_values, &lows_local, &highs_local, &equals_local, &lower_pivot_sample, &upper_pivot_sample, &equal_pivot_sample);
		}

		// Count the job values
		_process_segment_data_count(lows_local, highs_local, equals_local, lower_pivot_sample, upper_pivot_sample, equal_pivot_sample,
			lows, highs, equals, pivot_lower_samples, pivot_upper_samples, pivot_equal_samples);

		DEBUG_MSG("* Pivot: " << p << "\n");
		DEBUG_MSG("- Lows: " << *lows << "(+" << lows_local << ")\n");
		DEBUG_MSG("+ Highs: " << *highs << "(+" << highs_local << ")\n");
		DEBUG_MSG("= Equals: " << *equals << "(+" << equals_local << ")\n");
		DEBUG_MSG("==> L+H+E: " << (*lows + *highs + *equals) << "\n");
	}
}

#pragma endregion

void worker::bucket::find(std::ifstream* file, size_t* fsize, int percentil,
	size_t* total_values, double* bucket_lower_val, double* bucket_upper_val, size_t* bucket_value_offset, size_t* bucket_total_found)
{
	size_t fi; // data file iterator
	size_t fi_fsize_remaining; // data file iterator counter based on remaining file size to read
	std::streamoff fi_seekfrom = 0;

	char* buffer = nullptr;
	size_t buffer_size = 0;

	float pctp_offset = 0; // offset % from the lowest (initial) limit to the current one; it maintain percentil position in all data sequence

	double bucket_pivot_val = 0; // initial pivot value of currently calculated bucket
	size_t lows = 0; // Bucket numers lower than pivot
	size_t highs = 0; // Bucket numbers greater than pivot
	size_t equals = 0; // Bucket numbers equal to pivot
	std::vector<double> pivot_lower_samples;
	std::vector<double> pivot_upper_samples;
	std::vector<double> pivot_equal_samples;
	// THe main data iteration
	do
	{
		if (worker::values::get_state()->terminate_process_requested) return;
		worker::values::get_state()->bucket_task++;
		worker::values::get_state()->bucket_task_sub = 0;

		// Reset segment counters
		fi_seekfrom = 0;
		lows = 0;
		highs = 0;
		equals = 0;
		pivot_lower_samples.clear();
		pivot_upper_samples.clear();
		pivot_equal_samples.clear();

		// Iterate over the segments
		// ... to preserve the memory limit
		fi_fsize_remaining = *fsize;
		for (fi = 0; fi_fsize_remaining > 0; ++fi)
		{
			worker::values::get_state()->bucket_task_sub = fi;

			// Set seek position
			fi_seekfrom = constants::SEGMENT_SEARCH_MEMORY_LIMIT * fi;
			// Set buffer
			utils::fi_set_buffer(&buffer, &buffer_size, &fi_fsize_remaining, constants::SEGMENT_SEARCH_MEMORY_LIMIT);
			// Read data into buffer
			(*file).seekg(fi_seekfrom, std::ios::beg);
			(*file).read(buffer, buffer_size);

			// Process segment data
			_process_segment_data((double*)buffer, buffer_size / sizeof(double),
				bucket_pivot_val, *bucket_upper_val, *bucket_lower_val,
				total_values, &lows, &highs, &equals, &pivot_lower_samples, &pivot_upper_samples, &pivot_equal_samples);
			if (worker::values::get_state()->terminate_process_requested) return;
		}

		// Free the last buffer once done
		utils::fi_try_free_buffer(&buffer);

		// Let only first round count it
		worker::values::get_state()->total_values_counted = true;

		// Count found values
		*bucket_total_found = lows + highs + equals;

		// If there is need for the next segment calculations...
		if (*bucket_total_found * sizeof(double) > constants::SEGMENT_SEARCH_MEMORY_LIMIT)
		{
			// Get pct of upper/lower segment counters
			float pctp_lower = lows / (*total_values / 100.0f);
			float pctp_upper = highs / (*total_values / 100.0f);
			float pctp_equal = equals / (*total_values / 100.0f);

			// Set limits for the next segment calculation...
			// If UPPER goes...
			if (percentil > pctp_lower + pctp_equal + pctp_offset)
			{
				*bucket_lower_val = bucket_pivot_val;
				pivot_upper_samples.insert(pivot_upper_samples.end(), pivot_equal_samples.begin(), pivot_equal_samples.end());
				bucket_pivot_val = utils::select_r_item(pivot_upper_samples, (int)pivot_upper_samples.size());
				pctp_offset += pctp_lower;
				*bucket_value_offset += lows;
				DEBUG_MSG("=== UPPER GOES NEXT ==============" << "\n");
			}
			// If EQUAL goes...
			else if (percentil > pctp_lower + pctp_offset)
			{
				*bucket_lower_val = bucket_pivot_val;
				*bucket_upper_val = bucket_pivot_val;
				break; // pivot => percentil value
			}
			// Otherwise, LOWER goes...
			else
			{
				*bucket_upper_val = bucket_pivot_val;
				pivot_lower_samples.insert(pivot_lower_samples.end(), pivot_equal_samples.begin(), pivot_equal_samples.end());
				bucket_pivot_val = utils::select_r_item(pivot_lower_samples, (int)pivot_lower_samples.size());
				DEBUG_MSG("=== LOWER GOES NEXT ==============" << "\n");
			}

			DEBUG_MSG("========== SEGMENT DONE ==========" << "\n");
			DEBUG_MSG("L+H+E = " << (lows + highs + equals) << "\n");
			DEBUG_MSG("pctp_upper: " << pctp_upper << "\n");
			DEBUG_MSG("pctp_lower: " << pctp_lower << "\n");
			DEBUG_MSG("--- new ---" << "\n");
			DEBUG_MSG("bucket_pivot_val:(" << bucket_pivot_val << ")\n");
			DEBUG_MSG("bucket_upper_val: " << *bucket_upper_val << "\n");
			DEBUG_MSG("bucket_lower_val: " << *bucket_lower_val << "\n");
			DEBUG_MSG("bucket_value_offset: " << *bucket_value_offset << "\n");
			DEBUG_MSG("pctp_offset: " << pctp_offset << "\n");
			DEBUG_MSG("==================================" << "\n\n");

			// If the buffer ends up in sequence of the same single number above the memory limit...
			if (*bucket_lower_val == *bucket_upper_val)
				break;
		}

	} while (*bucket_total_found * sizeof(double) > constants::SEGMENT_SEARCH_MEMORY_LIMIT); // if there is need for the next segment calculations...

	worker::values::get_state()->bucket_found = true;
}