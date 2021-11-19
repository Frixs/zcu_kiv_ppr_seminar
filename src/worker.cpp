#include "worker.h"

#pragma region Process Variables

/// Global mutex for iterations
std::mutex _i_mutex;

/// State values of the currently processing worker
worker::State* _state;

/// Processing type of the currently processing worker
worker::ProcessingType* _processing_type;

#pragma endregion

#pragma region General Functions

/// <summary>
/// Frees buffer, if any.
/// </summary>
void _try_free_buffer(char** buffer)
{
	// Free buffer from previous calculation...
	if (*buffer != nullptr)
	{
		// Free the buffer memory
		mymem::free(*buffer);
		*buffer = nullptr;
	}
}

/// <summary>
/// Sets buffer based on iterator going through the data file (it frees the memory if the buffer is not nullptr).
/// </summary>
/// <param name="buffer">The buffer</param>
/// <param name="buffer_size">Size of the set buffer.</param>
/// <param name="fi_fsize_remaining">Iterator remaining file size to go through based on which is set buffer size.</param>
/// <param name="memory_limit">Memory limit.</param>
void _fi_set_buffer(char** buffer, size_t* buffer_size, size_t* fi_fsize_remaining, unsigned int memory_limit)
{
	_try_free_buffer(buffer);

	// Allocate buffer memory and get buffer size for the corresponding segment
	if (*fi_fsize_remaining >= memory_limit)
	{
		*buffer = (char*)mymem::malloc(memory_limit);
		*buffer_size = memory_limit;
		*fi_fsize_remaining -= memory_limit;
	}
	// Otherwise, take the rest...
	else
	{
		*buffer = (char*)mymem::malloc((size_t)(*fi_fsize_remaining));
		*buffer_size = *fi_fsize_remaining;
		*fi_fsize_remaining = 0;
	}
}

#pragma endregion

#pragma region 1. Find Bucket Limits

/// <summary>
/// Processing logic for each value that processed in _process_segment_data() specific for OpenCL
/// </summary>
std::string _process_segment_data_job()
{
	std::string code = R"CLC(
		__kernel void run(__global double* data, 
			__global size_t* total_values, __global double* bucket_lower_val, __global double* bucket_upper_val, __global double* test,
			const double inf_val)
		{
			int i = get_global_id(0);

			double v = data[i];
			test[i] = 2;

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
				(*total_values)++;

				if (*bucket_upper_val < +inf_val)
				{
					if (v > *bucket_upper_val)
						*bucket_upper_val = v;
				}
				else
				{
					*bucket_upper_val = v;
				}

				if (*bucket_lower_val > -inf_val)
				{
					if (v < *bucket_lower_val)
						*bucket_lower_val = v;
				}
				else
				{
					*bucket_lower_val = v;
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
	//std::cout << "-> " << v << std::endl;

	if (utils::is_double_valid(v))
	{
		// Count total (valid) values, if not counted yet...
		if (!_state->total_values_counted)
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
	const std::lock_guard<std::mutex> lock(_i_mutex);

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
	if (*_processing_type == worker::ProcessingType::OpenCL)
	{
		auto program = utils::cl_create_program(_process_segment_data_job());
		auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
		auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		//cl_int error;
		//cl::Buffer in_buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(double) * vec.size(), vec.data(), &error);
		//cl::Buffer out_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(double) * vec.size(), nullptr, &error);
		//cl::Kernel kernel(program, "run");
		//error = kernel.setArg(0, in_buf);
		//error = kernel.setArg(1, out_buf);

		//cl::CommandQueue queue(context, device);

		//error = queue.enqueueFillBuffer(in_buf, 3, sizeof(double) * 10, sizeof(double) * (vec.size() - 10));
		//error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()));
		//error = queue.enqueueReadBuffer(out_buf, CL_FALSE, 0, sizeof(double) * vec.size(), vec.data());

		/*std::vector<double> test(buffer_size / sizeof(double));
		cl::Buffer cl_buf_buffer_vals(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, buffer_size, buffer_vals, &error);
		cl::Buffer cl_buf_total_values(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(size_t), total_values, &error);
		cl::Buffer cl_buf_bucket_lower_val(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double), bucket_lower_val, &error);
		cl::Buffer cl_buf_bucket_upper_val(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double), bucket_upper_val, &error);
		cl::Buffer cl_buf_test(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, buffer_size, nullptr, &error);
		cl::Kernel kernel(program, "run");
		error = kernel.setArg(0, cl_buf_buffer_vals);
		error = kernel.setArg(1, cl_buf_total_values);
		error = kernel.setArg(2, cl_buf_bucket_lower_val);
		error = kernel.setArg(3, cl_buf_bucket_upper_val);
		error = kernel.setArg(4, cl_buf_test);
		error = kernel.setArg(5, std::numeric_limits<double>::infinity());

		cl::CommandQueue queue(context, device);
		error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(buffer_size / sizeof(double)));
		error = queue.enqueueReadBuffer(cl_buf_total_values, CL_FALSE, 0, sizeof(size_t), total_values);
		error = queue.enqueueReadBuffer(cl_buf_bucket_lower_val, CL_FALSE, 0, sizeof(double), bucket_lower_val);
		error = queue.enqueueReadBuffer(cl_buf_bucket_upper_val, CL_FALSE, 0, sizeof(double), bucket_upper_val);
		error = queue.enqueueReadBuffer(cl_buf_test, CL_FALSE, 0, buffer_size, test.data());

		cl::finish();*/

	}
	// If multithread processing...
	else if (*_processing_type == worker::ProcessingType::MultiThread)
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
				if (_state->terminate_process_requested) return;

				// Do the job
				_process_segment_data_job(values, i,
					p, h, l,
					&total_values_local, &lows_local, &highs_local, &equals_local, &lower_pivot_sample, &upper_pivot_sample, &equal_pivot_sample);
			}

			// Count the job values
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
			if (_state->terminate_process_requested) return;

			// Do the job
			_process_segment_data_job(values, i,
				p, h, l,
				total_values, &lows_local, &highs_local, &equals_local, &lower_pivot_sample, &upper_pivot_sample, &equal_pivot_sample);
		}

		// Count the job values
		_process_segment_data_count(lows_local, highs_local, equals_local, lower_pivot_sample, upper_pivot_sample, equal_pivot_sample,
			lows, highs, equals, pivot_lower_samples, pivot_upper_samples, pivot_equal_samples);

		std::cout << "* Pivot: " << p << "\n";
		std::cout << "- Lows: " << *lows << "(+" << lows_local << ")\n";
		std::cout << "+ Highs: " << *highs << "(+" << highs_local << ")\n";
		std::cout << "= Equals: " << *equals << "(+" << equals_local << ")\n";
		std::cout << "==> L+H+E: " << (*lows + *highs + *equals) << "\n";
	}
}

/// <summary>
/// 1. part of the algorithm - find limit upper and lower value to specify range for the final bucket that can be read in limited memory.
/// </summary>
void _find_bucket_limits(std::ifstream* file, size_t* fsize, int percentil,
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
		if (_state->terminate_process_requested) return;
		_state->bucket_task++;
		_state->bucket_task_sub = 0;

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
			_state->bucket_task_sub = fi;

			// Set seek position
			fi_seekfrom = constants::SEGMENT_SEARCH_MEMORY_LIMIT * fi;
			// Set buffer
			_fi_set_buffer(&buffer, &buffer_size, &fi_fsize_remaining, constants::SEGMENT_SEARCH_MEMORY_LIMIT);
			// Read data into buffer
			(*file).seekg(fi_seekfrom, std::ios::beg);
			(*file).read(buffer, buffer_size);

			// Process segment data
			_process_segment_data((double*)buffer, buffer_size / sizeof(double),
				bucket_pivot_val, *bucket_upper_val, *bucket_lower_val,
				total_values, &lows, &highs, &equals, &pivot_lower_samples, &pivot_upper_samples, &pivot_equal_samples);
			if (_state->terminate_process_requested) return;
		}

		// Free the last buffer once done
		_try_free_buffer(&buffer);

		// Let only first round count it
		_state->total_values_counted = true;

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
				std::cout << "=== UPPER GOES NEXT ==========================" << "\n";
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
				std::cout << "=== LOWER GOES NEXT ==========================" << "\n";
			}

			std::cout << "========== SEGMENT DONE ==========" << "\n";
			std::cout << "L+H+E = " << (lows + highs + equals) << "\n";
			std::cout << "pctp_upper: " << pctp_upper << "\n";
			std::cout << "pctp_lower: " << pctp_lower << "\n";
			std::cout << "--- new ---" << "\n";
			std::cout << "bucket_pivot_val:(" << bucket_pivot_val << ")\n";
			std::cout << "bucket_upper_val: " << *bucket_upper_val << "\n";
			std::cout << "bucket_lower_val: " << *bucket_lower_val << "\n";
			std::cout << "bucket_value_offset: " << *bucket_value_offset << "\n";
			std::cout << "pctp_offset: " << pctp_offset << "\n";
			std::cout << "==================================" << "\n\n";

			// If the buffer ends up in sequence of the same single number above the memory limit...
			if (*bucket_lower_val == *bucket_upper_val)
				break;
		}

	} while (*bucket_total_found * sizeof(double) > constants::SEGMENT_SEARCH_MEMORY_LIMIT); // if there is need for the next segment calculations...
}

#pragma endregion

#pragma region 2. Find Percentil

/// <summary>
/// Processing logic for each value that processed in _find_percentil()
/// </summary>
void _find_percentil_job(double* buffer, size_t i,
	std::vector<double>* percentil_bucket, size_t* iv, double bucket_lower_val, double bucket_upper_val)
{
	const std::lock_guard<std::mutex> lock(_i_mutex);

	double v = buffer[i];
	//std::cout << "-> " << v << std::endl;

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

/// <summary>
/// 2. part of the algorithm - find percentil value in the data file based on limit upper and lower value.
/// </summary>
void _find_percentil(std::ifstream* file, size_t* fsize, size_t total_values, int percentil, double bucket_lower_val, double bucket_upper_val, size_t bucket_value_offset, size_t bucket_total_found,
	double* percentil_value)
{
	size_t fi; // data file iterator
	size_t fi_fsize_remaining; // data file iterator counter based on remaining file size to read
	std::streamoff fi_seekfrom = 0;

	char* buffer = nullptr;
	size_t buffer_size = 0;

	size_t percentil_pos = (size_t)round(total_values * (percentil / 100.0f)); // get percentil number position relative to the entire (valid) data sequence
	size_t percentil_bucket_idx = percentil_pos - bucket_value_offset - 1;

	std::vector<double> percentil_bucket(bucket_total_found);
	size_t iv = 0;

	// Iterate over the segments
	// ... to preserve the memory limit
	fi_fsize_remaining = *fsize;
	for (fi = 0; fi_fsize_remaining > 0; ++fi)
	{
		if (_state->terminate_process_requested) return;
		_state->percentil_search_task = fi;

		// Set seek position
		fi_seekfrom = constants::SEGMENT_PICK_MEMORY_LIMIT * fi;
		// Set buffer
		_fi_set_buffer(&buffer, &buffer_size, &fi_fsize_remaining, constants::SEGMENT_PICK_MEMORY_LIMIT);
		// Read data into buffer
		(*file).seekg(fi_seekfrom, std::ios::beg);
		(*file).read(buffer, buffer_size);

		// If multithread processing...
		if (*_processing_type == worker::ProcessingType::MultiThread)
		{
			// Parallel work (job)
			auto work = [&](tbb::blocked_range<size_t> it)
			{
				for (size_t i = it.begin(); i < it.end(); ++i)
				{
					if (_state->terminate_process_requested) return;

					// Do the job
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
				if (_state->terminate_process_requested) return;

				// Do the job
				_find_percentil_job((double*)buffer, i, &percentil_bucket, &iv, bucket_lower_val, bucket_upper_val);
			}
		}
	}

	// Free the last buffer once done
	_try_free_buffer(&buffer);

	_state->percentil_search_done = true;
	if (_state->terminate_process_requested) return;
	// Quick select (sort)
	auto m = percentil_bucket.begin() + percentil_bucket_idx;
	std::nth_element(percentil_bucket.begin(), m, percentil_bucket.end());
	_state->waiting_for_percentil_pickup = true;
	if (_state->terminate_process_requested) return;

	// DEBUG MESSAGES
	std::cout << "percentil_pos = " << percentil_pos << std::endl;
	std::cout << "percentil_bucket_idx = " << percentil_bucket_idx << std::endl;
	std::cout << "bucket_upper_val = " << bucket_upper_val << std::endl;
	std::cout << "bucket_lower_val = " << bucket_lower_val << std::endl;
	/*std::cout << "v= {";
	for (double i : percentil_bucket)
		std::cout << i << ", ";
	std::cout << "}\n";*/
	std::cout << "PERCENTIL : " << percentil << std::endl;

	// Set the value
	*percentil_value = percentil_bucket[percentil_bucket_idx];
}

#pragma endregion

#pragma region 3. Find Result

/// <summary>
/// Processing logic for each value that processed in _find_result()
/// </summary>
void _find_result_job(double* buffer, size_t i, double percentil_value,
	size_t* first_occurance_index, size_t* last_occurance_index)
{
	const std::lock_guard<std::mutex> lock(_i_mutex);

	double v = buffer[i];
	//std::cout << "-> " << v << std::endl;

	if (utils::is_double_valid(v) && v == percentil_value)
	{
		if (*first_occurance_index == NAN)
			*first_occurance_index = i;
		*last_occurance_index = i;
	}
}

/// <summary>
/// 3. part of the algorithm - find result values in the data file.
/// </summary>
void _find_result(std::ifstream* file, size_t* fsize, double percentil_value,
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
		if (_state->terminate_process_requested) return;
		_state->result_search_task = fi;

		// Set seek position
		fi_seekfrom = constants::SEGMENT_PICK_MEMORY_LIMIT * fi;
		// Set buffer
		_fi_set_buffer(&buffer, &buffer_size, &fi_fsize_remaining, constants::SEGMENT_PICK_MEMORY_LIMIT);
		// Read data into buffer
		(*file).seekg(fi_seekfrom, std::ios::beg);
		(*file).read(buffer, buffer_size);

		// If multithread processing...
		if (*_processing_type == worker::ProcessingType::MultiThread)
		{
			// Parallel work (job)
			auto work = [&](tbb::blocked_range<size_t> it)
			{
				for (size_t i = it.begin(); i < it.end(); ++i)
				{
					if (_state->terminate_process_requested) return;

					// Do the job
					_find_result_job((double*)buffer, i, percentil_value, first_occurance_index, last_occurance_index);
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
				if (_state->terminate_process_requested) return;

				// Do the job
				_find_result_job((double*)buffer, i, percentil_value, first_occurance_index, last_occurance_index);
			}
		}
	}

	// Free the last buffer once done
	_try_free_buffer(&buffer);
}

#pragma endregion

#pragma region Process Functions

void worker::run(worker::State* state, std::string filePath, int percentil, worker::ProcessingType* processing_type)
{
	// Assign new state
	_state = state;
	_processing_type = processing_type;

	// Open file
	std::ifstream file(filePath, std::ios::binary);
	if (file.is_open())
	{
		if (_state->terminate_process_requested) return;
		_state->file_loaded = true;
		_state->terminated = false;

		double percentil_value;

		size_t fsize; // size of the opened data file

		double bucket_upper_val; // upper limit bucket value
		double bucket_lower_val; // lower limit bucket value
		size_t total_values = 0; // total valid values
		size_t bucket_value_offset = 0; // offset from the lowest (initial) limit to the current one of total valid values
		size_t bucket_total_found = 0;



		// Get file size
		file.seekg(0, std::ios::end);
		fsize = (size_t)file.tellg(); // in bytes
		file.seekg(0, std::ios::beg);

		// Set limits
		bucket_upper_val = +(std::numeric_limits<double>::infinity());
		bucket_lower_val = -(std::numeric_limits<double>::infinity());



		// Get starting timepoint
		auto time_start = std::chrono::high_resolution_clock::now();

		// 1. - Find lower/upper limit values
		std::cout << "Finding lower/upper bucket values according to memory limits..." << std::endl;
		_find_bucket_limits(&file, &fsize, percentil, &total_values, &bucket_lower_val, &bucket_upper_val, &bucket_value_offset, &bucket_total_found);
		std::cout << "Lower/Upper values successfully found!" << std::endl << std::endl;
		_state->bucket_found = true;
		if (_state->terminate_process_requested) return;

		// Get ending timepoint
		auto time_stop = std::chrono::high_resolution_clock::now();

		// Get duration. Substart timepoints to 
		// get durarion. To cast it to proper unit
		// use duration cast method
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(time_stop - time_start);

		std::cout << "Time taken to select final bucket: "
			<< duration.count() << " seconds" << std::endl << std::endl;

		// If the data is NOT sequence of the same single number...
		if (bucket_lower_val != bucket_upper_val)
		{
			// 2. - Get the percentil
			std::cout << "Selecting percentil value..." << std::endl;
			_find_percentil(&file, &fsize, total_values, percentil, bucket_lower_val, bucket_upper_val, bucket_value_offset, bucket_total_found, &percentil_value);
			std::cout << "Percentil value succesfully selected!" << std::endl << std::endl;
			if (_state->terminate_process_requested) return;
		}
		// Otherwise, there is only 1 number...
		else
		{
			// ... so any percentil is any number of the sequence
			percentil_value = bucket_upper_val;
			_state->percentil_search_done = true;
			_state->waiting_for_percentil_pickup = true;
			_state->bucket_found = true;
		}

		if (_state->terminate_process_requested) return;

		std::cout << "Percentil value = " << percentil_value << std::endl << std::endl;

		// 3. - Find result
		size_t first_occurance_index = (size_t)NAN;
		size_t last_occurance_index = (size_t)NAN;
		std::cout << "Finding result..." << std::endl;
		_find_result(&file, &fsize, percentil_value, &first_occurance_index, &last_occurance_index);
		std::cout << "Result succesfully found!" << std::endl << std::endl;

		std::cout << "Percentil first index = " << first_occurance_index << std::endl;
		std::cout << "Percentil last index = " << last_occurance_index << std::endl << std::endl;

		// Close the file
		file.close();

	}
	else
	{
		std::cout << "Unable to open file";
	}

	mymem::print_counter();
}

#pragma endregion

#pragma region State Class

worker::State::State()
{
	this->set_defaults();
}

void worker::State::set_defaults()
{
	this->terminated = true;

	this->terminate_process_requested = false;
	this->recovery_requested = false;

	this->file_loaded = false;
	this->total_values_counted = false;

	this->bucket_task_sub = 0;
	this->bucket_task = 0;
	this->bucket_found = false;

	this->percentil_search_task = 0;
	this->percentil_search_done = false;

	this->waiting_for_percentil_pickup = false;

	this->result_search_task = 0;
	this->result_search_done = false;
}

#pragma endregion