#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <future>
#include "mymem.h"
#include "farmer.h"
#include "watchdog.h"


// TODO: check file has some required minimum file size

int main()
{
	std::cout << "Starting...\n\n";
	
	// Create new state values
	auto state = farmer::State();
	
	// Start watchdog
	auto watchdog_res = std::async(watchdog::run, state);

	// Start processing
	farmer::process(state, "data/data4.bin", 35, farmer::ProcessingType::SingleThread);



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
