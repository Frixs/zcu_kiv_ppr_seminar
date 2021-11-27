#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <future>
#include <iomanip>

#include "mymem.h"
#include "constants.h"
#include "worker_values.h"
#include "worker.h"
#include "watchdog.h"

#pragma region Init Helper Functions

inline bool file_exists(const std::string& name)
{
	std::ifstream ifile;
	ifile.open(name);
	if (ifile)
	{
		ifile.close();
		return true;
	}
	return false;
}

std::ifstream::pos_type get_file_size(const char* filename)
{
	std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
	return in.tellg();
}

#pragma endregion

/// <summary>
/// Check parameters
/// </summary>
bool check_parameters(char* p1, char* p2, char* p3)
{
	if (!file_exists(p1))
	{
		std::cout << "Data file does not exist!" << std::endl;
		return false;
	}

	if (get_file_size(p1) < 8)
	{
		std::cout << "Data file is required to have minimum size of 8 bytes!" << std::endl;
		return false;
	}

	char* endptr;
	int value = strtol(p2, &endptr, 10);
	if (*endptr != '\0')
	{
		std::cout << "Invalid percentil value!" << std::endl;
		return false;
	}

	if (value <= 0 || value > 100)
	{
		std::cout << "Invalid percentil range!" << std::endl;
		return false;
	}
	
	if (strcmp(p3, "single") != 0 && strcmp(p3, "SMP") != 0 && strcmp(p3, "OpenCL") != 0)
	{
		std::cout << "Invalid computation type!" << std::endl;
		return false;
	}

	return true;
}

/// <summary>
/// Program main function
/// </summary>
int main(int argc, char* argv[])
{
	bool ok_to_run = false;
	std::string filePath = "";
	int percentil = 1;
	auto processing_type = worker::values::ProcessingType::SingleThread;

	// Get parameters
	if (argc > 3)
	{
		char* p1 = nullptr;
		char* p2 = nullptr;
		char* p3 = nullptr;
		
		if (strcmp(argv[1], "-v") == 0)
		{
			utils::cout_toggle_set_default(true);
			p1 = argv[2];
			p2 = argv[3];
			p3 = argv[4];
		}
		else
		{
			utils::cout_toggle_set_default(false);
			p1 = argv[1];
			p2 = argv[2];
			p3 = argv[3];
		}

		utils::cout_toggle(true);
		if (check_parameters(p1, p2, p3))
		{
			filePath = p1;
			percentil = std::stoi(p2);
			processing_type = strcmp(p3, "OpenCL") == 0 ? worker::values::ProcessingType::OpenCL
				: (strcmp(p3, "SMP") == 0 ? worker::values::ProcessingType::MultiThread : worker::values::ProcessingType::SingleThread);

			ok_to_run = true;
		}
		utils::cout_toggle_to_default();
	}
	else
	{
		std::cout << "Invalid parameters!" << std::endl;
	}
	
	// Check if everyything is ok to start run...
	if (ok_to_run)
	{
		DEBUG_MSG("Starting...\n\n");

		// Create new state values
		auto state = worker::values::State();

		// Start watchdog
		auto watchdog_res = std::async(watchdog::run, &state);

		// Start processing
		do
		{
			state.set_defaults();
			worker::run(&state, filePath, percentil, &processing_type);
		} while (state.recovery_requested);

		// Wait for shutting down the watchdog...
		DEBUG_MSG(std::endl << "Waiting for watchdog shutdown..." << std::endl);
	}
	else
	{
		utils::cout_toggle(true);
		std::cout << "Please, call the program according to following parameter isntructions:" << std::endl;
		std::cout << "=========================================================================" << std::endl;
		std::cout << "pprsolver.exe [-v] <YOUR_FILE_PATH> <PERCENTIL_NUMBER> <COMPUTATION_TYPE>" << std::endl;
		std::cout << "-------------------------------------------------------------------------" << std::endl;
		std::cout << "1: " << std::setw(7) << "text" << std::setw(5) << "" << "Toggle verbose output" << std::endl;
		std::cout << "2: " << std::setw(7) << "text" << std::setw(5) << "" << "Data file path" << std::endl;
		std::cout << "3: " << std::setw(7) << "number" << std::setw(5) << "" << "Searched percentil <1,100>" << std::endl;
		std::cout << "4: " << std::setw(7) << "text" << std::setw(5) << "" << "Type of computation (single|SMP|OpenCL)" << std::endl;
		std::cout << "=========================================================================" << std::endl;
		utils::cout_toggle_to_default();
	}



	//// percentil = 2 (35%)
	//double x1 = 8;
	//double x2 = 20;
	//double x3 = 10;
	//double x4 = 2;
	//double x5 = 0;
	//double x6 = 1;
	//double x7 = -5;
	//double x8 = 4;
	//double x9 = 7;
	//double x10 = 9;
	//std::ofstream outfile;
	//outfile.open("data/data5.bin", std::ios::binary | std::ios::out);
	//outfile.write(reinterpret_cast<const char*>(&x1), sizeof(double));
	//outfile.write(reinterpret_cast<const char*>(&x2), sizeof(double));
	//outfile.write(reinterpret_cast<const char*>(&x3), sizeof(double));
	//outfile.write(reinterpret_cast<const char*>(&x4), sizeof(double));
	//outfile.write(reinterpret_cast<const char*>(&x5), sizeof(double));
	//outfile.write(reinterpret_cast<const char*>(&x6), sizeof(double));
	//outfile.write(reinterpret_cast<const char*>(&x7), sizeof(double));
	//outfile.write(reinterpret_cast<const char*>(&x8), sizeof(double));
	//outfile.write(reinterpret_cast<const char*>(&x9), sizeof(double));
	//outfile.write(reinterpret_cast<const char*>(&x10), sizeof(double));
	//outfile.close();
	//std::ifstream file("data/data5.bin", std::ios::binary);
	//file.seekg(0, std::ios::end);
	//std::streampos fsize = file.tellg(); // in bytes
	//file.seekg(0, std::ios::beg);
	//char* buffer = new char[fsize];
	//file.read(buffer, fsize);
	//for (size_t i = 0; i < fsize / sizeof(double); i++)
	//	std::cout << ((double*)buffer)[i] << std::endl;
	//file.close();



	return 0;
}
