#pragma once

#include <iostream>
#include <thread>
#include <chrono>
#include "constants.h"
#include "farmer.h"

namespace watchdog
{
	int run(farmer::State state);
}
