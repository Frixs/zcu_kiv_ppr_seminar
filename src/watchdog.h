#pragma once

#include <iostream>
#include <thread>
#include <chrono>
#include "constants.h"
#include "worker.h"

namespace watchdog
{
	int run(worker::State* state);
}
