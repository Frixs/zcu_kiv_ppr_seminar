#include "utils.h"

bool utils::is_double_valid(double d)
{
	int cl = std::fpclassify(d);
	return cl == FP_NORMAL || cl == FP_ZERO;
}

double utils::select_r_item(std::vector<double> stream, int n)
{
	int i; // index for elements in stream[]
	double reservoir = stream[0];

	// Use a different seed value so that we don't get
	// same result each time we run this program
	srand(time(NULL));

	// Iterate from the (k+1)th element to nth element
	for (i = 1; i < n; ++i)
	{
		// Pick a random index from 0 to i.
		int j = rand() % (i + 1);

		if (j < 1)
			reservoir = stream[i];
	}

	return reservoir;
}

size_t utils::generate_rand(size_t max)
{
	return rand() % (max + 1); // 0 .. max
}

cl::Device utils::cl_get_gpu_device()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.size() > 0)
	{
		auto platform = platforms.front();
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

		if (devices.size() > 0)
			return devices.front();
	}

	throw std::runtime_error("Error occurred while selecting the OpenCL GPU device.");
	return cl::Device();
}

cl::Program utils::cl_create_program(const std::string& src)
{
	cl_int error;

	cl::Context context(utils::cl_get_gpu_device());
	cl::Program program(context, src);

	error = program.build("-cl-std=CL1.2");
	if (error != CL_BUILD_SUCCESS)
	{
		throw std::runtime_error("Error occurred while building the OpenCL program: " + std::to_string(error));
	}

	return program;
}