#include "watchdog.h"

/// <summary>
/// UNDONE
/// </summary>
int watchdog::run(worker::State state)
{
	while (true)
	{
		std::cout << std::endl << std::endl << "watchdog here" << std::endl << std::endl;
		// Sleep
		std::this_thread::sleep_for(std::chrono::seconds(constants::WATCHDOG_TIME_INTERVAL));
	}
}