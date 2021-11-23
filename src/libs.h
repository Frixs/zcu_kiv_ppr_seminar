#pragma once

#include "tbb/parallel_for.h"
#include "tbb/combinable.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#pragma comment(lib, "OpenCL.lib") // evade MS Studio 2019 error LNK2019
#include "CL/cl.h"
#include "CL/cl.hpp"