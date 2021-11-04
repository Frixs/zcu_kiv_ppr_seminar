#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <chrono>
#include "farmer.h"
#include "mymem.h"
#include "constants.h"
#include "utils.h"

#pragma region Private Functions

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

/// <summary>
/// UNDONE
/// </summary>
/// <param name="values"></param>
/// <param name="n"></param>
/// <param name="p"></param>
/// <param name="h"></param>
/// <param name="l"></param>
/// <param name="highs"></param>
/// <param name="lows"></param>
/// <param name="pivot_upper_samples"></param>
/// <param name="pivot_lower_samples"></param>
void _process_segment_data(double* values, size_t n,
	double p, double h, double l,
	size_t* lows, size_t* highs, size_t* equals, std::vector<double>* pivot_lower_samples, std::vector<double>* pivot_upper_samples, std::vector<double>* pivot_equal_samples)
{
	size_t curr_lows = 0;
	size_t curr_highs = 0;
	size_t curr_equals = 0;
	double lower_pivot_sample;
	double upper_pivot_sample;
	double equal_pivot_sample;

	for (size_t i = 0; i < n; ++i)
	{
		double v = values[i];
		//std::cout << "-> " << v << std::endl;

		if (utils::is_double_valid(v))
		{
			// Check the limits
			if (v >= l && v <= h)
			{
				// Greater than pivot...
				if (v > p)
				{
					// If next number to highs (+ increment highs)...
					if (curr_highs++ > 0)
					{
						if (i == rand() % (i + 1)) // 0 .. i
							upper_pivot_sample = v;
					}
					// Otherwise, first number to highs...
					else
						upper_pivot_sample = v;
				}
				// Lower than pivot...
				else if (v < p)
				{
					// If next number to lows (+ increment lows)...
					if (curr_lows++ > 0)
					{
						if (i == rand() % (i + 1)) // 0 .. i
							lower_pivot_sample = v;
					}
					// Otherwise, first number to lows...
					else
						lower_pivot_sample = v;
				}
				// Otherwise, equal to pivot...
				else
				{
					// If next number to equals (+ increment equals)...
					if (curr_equals++ > 0)
					{
						if (i == rand() % (i + 1)) // 0 .. i
							equal_pivot_sample = v;
					}
					// Otherwise, first number to equals...
					else
						equal_pivot_sample = v;
				}
			}
		}
	}

	// Select next segment pivot samples
	if (curr_lows > 0)
		pivot_lower_samples->push_back(lower_pivot_sample);
	if (curr_highs > 0)
		pivot_upper_samples->push_back(upper_pivot_sample);
	if (curr_equals > 0)
		pivot_equal_samples->push_back(equal_pivot_sample);

	// Update counters
	*lows += curr_lows;
	*highs += curr_highs;
	*equals += curr_equals;

	std::cout << "* Pivot: " << p << "\n";
	std::cout << "- Lows: " << *lows << "(+" << curr_lows << ")\n";
	std::cout << "+ Highs: " << *highs << "(+" << curr_highs << ")\n";
	std::cout << "= Equals: " << *equals << "(+" << curr_equals << ")\n";
	std::cout << "==> L+H+E: " << (*lows + *highs + *equals) << "\n";
}

/// <summary>
/// UNDONE
/// </summary>
void _analyze_data(std::ifstream* file, size_t* fsize, size_t* total_values, double* bucket_lower_val, double* bucket_upper_val)
{
	size_t fi; // data file iterator
	size_t fi_fsize_remaining; // data file iterator counter based on remaining file size to read
	std::streamoff fi_seekfrom = 0;

	char* buffer = nullptr;
	size_t buffer_size = 0;

	fi_fsize_remaining = *fsize;
	for (fi = 0; fi_fsize_remaining > 0; ++fi)
	{
		// Set seek position
		fi_seekfrom = constants::SEGMENT_SEARCH_MEMORY_LIMIT * fi;
		// Set buffer
		_fi_set_buffer(&buffer, &buffer_size, &fi_fsize_remaining, constants::SEGMENT_SEARCH_MEMORY_LIMIT);
		// Read data into buffer
		(*file).seekg(fi_seekfrom, std::ios::beg);
		(*file).read(buffer, buffer_size);
		// Get values from the buffer
		double* buffer_vals = (double*)buffer;

		// Iterate over the segments
		// ... to preserve the memory limit
		for (size_t i = 0; i < buffer_size / sizeof(double); ++i)
		{
			double v = buffer_vals[i];

			if (utils::is_double_valid(v))
			{
				(*total_values)++;

				// Look for UPPER bucket value
				if (*bucket_upper_val < +(std::numeric_limits<double>::infinity()))
				{
					if (v > *bucket_upper_val)
						*bucket_upper_val = v;
				}
				// Otherwise, set it for the first time...
				else
					*bucket_upper_val = v;

				// Look for LOWER bucket value
				if (*bucket_lower_val > -(std::numeric_limits<double>::infinity()))
				{
					if (v < *bucket_lower_val)
						*bucket_lower_val = v;
				}
				// Otherwise, set it for the first time...
				else
					*bucket_lower_val = v;
			}
		}

		// Free the last buffer once done
		_try_free_buffer(&buffer);
	}
}

/// <summary>
/// UNDONE
/// </summary>
void _find_bucket_limits(std::ifstream* file, size_t* fsize, size_t total_values, int percentil, double* bucket_lower_val, double* bucket_upper_val, size_t* bucket_value_offset)
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
		// Reset segment counters
		fi_seekfrom = 0;
		lows = 0;
		highs = 0;
		equals = 0;
		pivot_upper_samples.clear();
		pivot_lower_samples.clear();

		// Iterate over the segments
		// ... to preserve the memory limit
		fi_fsize_remaining = *fsize;
		for (fi = 0; fi_fsize_remaining > 0; ++fi)
		{
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
				&lows, &highs, &equals, &pivot_lower_samples, &pivot_upper_samples, &pivot_equal_samples);
		}

		// Free the last buffer once done
		_try_free_buffer(&buffer);

		// If there is need for the next segment calculations...
		if ((lows + highs + equals) * sizeof(double) > constants::SEGMENT_SEARCH_MEMORY_LIMIT)
		{
			// Get pct of upper/lower segment counters
			float pctp_lower = lows / (total_values / 100.0f);
			float pctp_upper = highs / (total_values / 100.0f);
			float pctp_equal = equals / (total_values / 100.0f);

			// Set limits for the next segment calculation...
			// If UPPER goes...
			if (percentil > pctp_lower + pctp_equal + pctp_offset)
			{
				*bucket_lower_val = bucket_pivot_val;
				bucket_pivot_val = utils::select_r_item(pivot_upper_samples, pivot_upper_samples.size());
				pctp_offset += pctp_lower + pctp_equal;
				*bucket_value_offset += lows + equals;
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
				bucket_pivot_val = utils::select_r_item(pivot_lower_samples, pivot_lower_samples.size());
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

	} while ((lows + highs + equals) * sizeof(double) > constants::SEGMENT_SEARCH_MEMORY_LIMIT); // if there is need for the next segment calculations...
}

/// <summary>
/// UNDONE
/// </summary>
void _find_percentil(std::ifstream* file, size_t* fsize, size_t total_values, int percentil, double bucket_lower_val, double bucket_upper_val, size_t bucket_value_offset, double* percentil_value)
{
	size_t fi; // data file iterator
	size_t fi_fsize_remaining; // data file iterator counter based on remaining file size to read
	std::streamoff fi_seekfrom = 0;

	char* buffer = nullptr;
	size_t buffer_size = 0;

	size_t percentil_pos = (size_t)round(total_values * (percentil / 100.0f)); // get percentil number position relative to the entire (valid) data sequence
	size_t percentil_bucket_idx = percentil_pos - bucket_value_offset - 1;

	std::vector<double> percentil_bucket{};

	// Iterate over the segments
	// ... to preserve the memory limit
	fi_fsize_remaining = *fsize;
	for (fi = 0; fi_fsize_remaining > 0; ++fi)
	{
		// Set seek position
		fi_seekfrom = constants::SEGMENT_PICK_MEMORY_LIMIT * fi;
		// Set buffer
		_fi_set_buffer(&buffer, &buffer_size, &fi_fsize_remaining, constants::SEGMENT_PICK_MEMORY_LIMIT);
		// Read data into buffer
		(*file).seekg(fi_seekfrom, std::ios::beg);
		(*file).read(buffer, buffer_size);

		for (size_t i = 0; i < buffer_size / sizeof(double); ++i)
		{
			double v = ((double*)buffer)[i];
			//std::cout << "-> " << v << std::endl;

			if (utils::is_double_valid(v))
			{
				// Check the limits
				if (v >= bucket_lower_val && v <= bucket_upper_val)
					percentil_bucket.push_back(v);
			}
		}
	}

	// Free the last buffer once done
	_try_free_buffer(&buffer);

	// Quick select (sort)
	auto m = percentil_bucket.begin() + percentil_bucket_idx;
	std::nth_element(percentil_bucket.begin(), m, percentil_bucket.end());

	// DEBUG MESSAGES
	std::cout << "percentil_pos = " << percentil_pos << std::endl;
	std::cout << "percentil_bucket_idx = " << percentil_bucket_idx << std::endl;
	std::cout << "bucket_upper_val = " << bucket_upper_val << std::endl;
	std::cout << "bucket_lower_val = " << bucket_lower_val << std::endl;
	//std::cout << "v= {";
	//for (double i : percentil_bucket)
	//	std::cout << i << ", ";
	//std::cout << "}\n";
	std::cout << "PERCENTIL : " << percentil << std::endl;
	
	// Set the value
	*percentil_value = percentil_bucket[percentil_bucket_idx];
}

#pragma endregion

#pragma region Public Functions

void farmer::process(int percentil)
{
	// Open file
	std::ifstream file("data/data4.bin", std::ios::binary);
	if (file.is_open())
	{
		double percentil_value;

		size_t fsize; // size of the opened data file

		double bucket_upper_val; // upper limit bucket value
		double bucket_lower_val; // lower limit bucket value
		size_t total_values = 0; // total valid values
		size_t bucket_value_offset = 0; // offset from the lowest (initial) limit to the current one of total valid values



		// Get file size
		file.seekg(0, std::ios::end);
		fsize = (size_t)file.tellg(); // in bytes
		file.seekg(0, std::ios::beg);

		// Set limits
		bucket_upper_val = +(std::numeric_limits<double>::infinity());
		bucket_lower_val = -(std::numeric_limits<double>::infinity());



		// Get starting timepoint
		auto time_start = std::chrono::high_resolution_clock::now();

		// 1. - First iteration to analyze the data
		std::cout << "Analyzing data..." << std::endl;
		_analyze_data(&file, &fsize, &total_values, &bucket_lower_val, &bucket_upper_val);
		std::cout << "Data sucessfully analyzed!" << std::endl << std::endl;

		// If the data is NOT sequence of the same single number...
		if (bucket_lower_val != bucket_upper_val)
		{
			// 2. - Find lower/upper limit values
			std::cout << "Finding lower/upper bucket values according to memory limits..." << std::endl;
			_find_bucket_limits(&file, &fsize, total_values, percentil, &bucket_lower_val, &bucket_upper_val, &bucket_value_offset);
			std::cout << "Lower/Upper values successfully found!" << std::endl << std::endl;

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
				// 3. - Get the percentil
				std::cout << "Selecting percentil value..." << std::endl;
				_find_percentil(&file, &fsize, total_values, percentil, bucket_lower_val, bucket_upper_val, bucket_value_offset, &percentil_value);
				std::cout << "Percentil value succesfully selected!" << std::endl << std::endl;
			}
			// Otherwise, there is only 1 number...
			else
			{
				// ... so any percentil is any number of the sequence
				percentil_value = bucket_upper_val;
			}
		}
		// Otherwise, there is only 1 number...
		else
		{
			// ... so any percentil is any number of the sequence
			percentil_value = bucket_upper_val;
		}

		std::cout << "Percentil value = " << percentil_value << std::endl << std::endl;


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
