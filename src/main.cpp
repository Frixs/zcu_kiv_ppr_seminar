#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <future>
#include "mymem.h"
#include "worker.h"
#include "watchdog.h"


// TODO: check file has some required minimum file size - min 8 bits / 1 byte

int main()
{
	bool ok_to_run = false;
	
	std::cout << "Starting...\n\n";

	// Get parameters
	std::string filePath = "data/data1.bin";
	int percentil = 35;
	auto processing_type = worker::ProcessingType::OpenCL;

	ok_to_run = true;

	// Check if everyything is ok to start run...
	if (ok_to_run)
	{
		// Create new state values
		auto state = worker::State();
		
		// Start watchdog
		auto watchdog_res = std::async(watchdog::run, &state);

		// Start processing
		do
		{
			state.set_defaults();
			worker::run(&state, filePath, percentil, &processing_type);
		} while (state.recovery_requested);
	}
	else
	{
		// TODO - unable to run message
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
