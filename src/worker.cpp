#include "worker.h"

#pragma region Process Functions

void worker::run(worker::values::State* state, std::string filePath, int percentil, worker::values::ProcessingType* processing_type)
{
	// Assign new state
	worker::values::init(state, processing_type);

	// Open file
	std::ifstream file(filePath, std::ios::binary);
	if (file.is_open())
	{
		if (worker::values::get_state()->terminate_process_requested) return;
		worker::values::get_state()->file_loaded = true;
		worker::values::get_state()->terminated = false;

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
		DEBUG_MSG("Finding lower/upper bucket values according to memory limits..." << std::endl);
		worker::bucket::find(&file, &fsize, percentil, &total_values, &bucket_lower_val, &bucket_upper_val, &bucket_value_offset, &bucket_total_found);
		DEBUG_MSG("Lower/Upper values successfully found!" << std::endl << std::endl);
		if (worker::values::get_state()->terminate_process_requested) return;

		// Get ending timepoint
		auto time_stop = std::chrono::high_resolution_clock::now();

		// Get duration. Substart timepoints to 
		// get durarion. To cast it to proper unit
		// use duration cast method
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(time_stop - time_start);

		DEBUG_MSG("Time taken to find final bucket: "
			<< duration.count() << " seconds" << std::endl << std::endl);

		// Get starting timepoint
		time_start = std::chrono::high_resolution_clock::now();

		// If valid data
		if (total_values > 0)
		{
			// If the data is NOT sequence of the same single number...
			if (bucket_lower_val != bucket_upper_val)
			{
				// 2. - Get the percentil
				DEBUG_MSG("Selecting percentil value..." << std::endl);
				worker::percentil::find(&file, &fsize, total_values, percentil, bucket_lower_val, bucket_upper_val, bucket_value_offset, bucket_total_found, &percentil_value);
				DEBUG_MSG("Percentil value succesfully selected!" << std::endl << std::endl);
				if (worker::values::get_state()->terminate_process_requested) return;
			}
			// Otherwise, there is only 1 number...
			else
			{
				// ... so any percentil is any number of the sequence
				percentil_value = bucket_upper_val;
				worker::values::get_state()->bucket_found = true;
				worker::values::get_state()->percentil_search_done = true;
				worker::values::get_state()->waiting_for_percentil_pickup = true;
				worker::values::get_state()->result_search_done = true;
			}

			if (worker::values::get_state()->terminate_process_requested) return;

			DEBUG_MSG("PERCENTIL VALUE: " << std::endl);
			DEBUG_MSG(std::hexfloat << percentil_value << std::defaultfloat << std::endl);
			DEBUG_MSG("(" << percentil_value << ")" << std::endl << std::endl);

			// Get ending timepoint
			time_stop = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::seconds>(time_stop - time_start);
			DEBUG_MSG("Time taken to find percentil: "
				<< duration.count() << " seconds" << std::endl << std::endl);

			// Get starting timepoint
			time_start = std::chrono::high_resolution_clock::now();

			// 3. - Find result
			size_t first_occurance_index = (size_t)NAN;
			size_t last_occurance_index = (size_t)NAN;
			DEBUG_MSG("Finding result..." << std::endl);
			worker::result::find(&file, &fsize, percentil_value, &first_occurance_index, &last_occurance_index);
			DEBUG_MSG("Result succesfully found!" << std::endl << std::endl);

			DEBUG_MSG("RESULT: " << std::endl);
			DEBUG_MSG((first_occurance_index * 8) << std::endl);
			DEBUG_MSG((last_occurance_index * 8) << std::endl);

			// Get ending timepoint
			time_stop = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::seconds>(time_stop - time_start);
			DEBUG_MSG("Time taken to find result: "
				<< duration.count() << " seconds" << std::endl << std::endl);

			std::cout << std::hexfloat << percentil_value << std::defaultfloat;
			std::cout << " " << (first_occurance_index * 8);
			std::cout << " " << (last_occurance_index * 8);
			std::cout << std::endl;
		}
		// Otherwise, invalid data
		else
		{
			std::cout << "Invalid data!" << std::endl;
			worker::values::get_state()->bucket_found = true;
			worker::values::get_state()->percentil_search_done = true;
			worker::values::get_state()->waiting_for_percentil_pickup = true;
			worker::values::get_state()->result_search_done = true;
		}
		// Close the file
		file.close();

	}
	else
	{
		std::cout << "Unable to open file!" << std::endl;
	}

	//mymem::print_counter();
}

#pragma endregion