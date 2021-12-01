#include "utils.h"

// Default cout state
bool cout_default_toggle_state = true;

void utils::cout_toggle_set_default(bool def)
{
	cout_default_toggle_state = def;
}

void utils::cout_toggle(bool toggle)
{
	if (toggle)
		std::cout.clear();
	else
		std::cout.setstate(std::ios::failbit);
}

void utils::cout_toggle_to_default()
{
	utils::cout_toggle(cout_default_toggle_state);
}

void utils::fi_try_free_buffer(char** buffer)
{
	// Free buffer from previous calculation...
	if (*buffer != nullptr)
	{
		// Free the buffer memory
		mymem::free(*buffer);
		*buffer = nullptr;
	}
}

void utils::fi_set_buffer(char** buffer, size_t* buffer_size, size_t* fi_fsize_remaining, unsigned int memory_limit)
{
	fi_try_free_buffer(buffer);

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
	srand((unsigned int)time(NULL));

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

cl::Device utils::cl_get_gpu_device(const std::string& device_name)
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for (auto& platform : platforms)
	{
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		for (auto& device : devices)
		{
			std::string name = device.getInfo<CL_DEVICE_NAME>().c_str();
			if (name == device_name)
				return device;
		}
	}

	DEBUG_MSG("Error occurred while selecting the OpenCL GPU device." << std::endl);
	throw std::runtime_error("Error occurred while selecting the OpenCL GPU device.");
	return cl::Device();
}

cl::Program utils::cl_create_program(const std::string& src, const std::string& device_name)
{
	cl_int error = 0;

	cl::Context context(utils::cl_get_gpu_device(device_name));
	cl::Program program(context, src);

	try {
		program.build("-cl-std=CL1.2");
	}
	catch (...) {
		DEBUG_MSG("Error occurred while building the OpenCL program: " << std::to_string(error) << std::endl);
		throw std::runtime_error("Error occurred while building the OpenCL program: " + std::to_string(error));
	}

	return program;
}