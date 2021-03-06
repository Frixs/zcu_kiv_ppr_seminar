#include "watchdog.h"

/// <summary>
/// Max fails that watchdog can accept in a row
/// </summary>
const size_t MAX_STRIKES = 3;

/// <summary>
/// Max tries for recovery
/// </summary>
const size_t MAX_RECOVERY_TRIES = 1;

/// <summary>
/// State watchdog strcuture
/// </summary>
struct t_watchdog_state
{
	size_t strikes;
	size_t recovery_tries = 0; // number of recovery tries
	size_t bucket_task_sub;
	size_t bucket_task;
	size_t percentil_search_task;
	size_t result_search_task;
} _watchdog_state{};

/// <summary>
/// Sets the strikes
/// </summary>
void _set_strike(t_watchdog_state* watchdog_state, bool up)
{
	if (up)
		(*watchdog_state).strikes++;
	else
		(*watchdog_state).strikes = (*watchdog_state).strikes > 0 ? (*watchdog_state).strikes - 1 : 0;
}

/// <summary>
/// The main test function
/// </summary>
int _test(worker::values::State* state, t_watchdog_state* watchdog_state)
{
	// File load
	if (!(*state).file_loaded)
	{
		return 2;
	}

	// Bucketing
	if (!(*state).bucket_found)
	{
		if ((*state).bucket_task > (*watchdog_state).bucket_task)
		{
			(*watchdog_state).bucket_task = (*state).bucket_task;
			(*watchdog_state).bucket_task_sub = 0;
			_set_strike(watchdog_state, false);
		}
		else if ((*state).bucket_task_sub > (*watchdog_state).bucket_task_sub)
		{
			(*watchdog_state).bucket_task_sub = (*state).bucket_task_sub;
			_set_strike(watchdog_state, false);
		}
		else
		{
			_set_strike(watchdog_state, true);
		}

		return 1;
	}

	// Percentil search
	if (!(*state).percentil_search_done)
	{
		if ((*state).percentil_search_task > (*watchdog_state).percentil_search_task)
		{
			(*watchdog_state).percentil_search_task = (*state).percentil_search_task;
			_set_strike(watchdog_state, false);
		}
		else
		{
			_set_strike(watchdog_state, true);
		}

		return 1;
	}

	// Waitting for percentil pickup
	if (!(*state).waiting_for_percentil_pickup)
	{
		return 1;
	}

	// Result search done
	if (!(*state).result_search_done)
	{
		if ((*state).result_search_task > (*watchdog_state).result_search_task)
		{
			(*watchdog_state).result_search_task = (*state).result_search_task;
			_set_strike(watchdog_state, false);
		}
		else
		{
			_set_strike(watchdog_state, true);
		}

		return 1;
	}

	return 0;
}

/// <summary>
/// Run watchdog
/// </summary>
int watchdog::run(worker::values::State* state)
{
	while (true)
	{
		// Sleep
		std::this_thread::sleep_for(std::chrono::seconds(constants::WATCHDOG_TIME_INTERVAL));

		if (!(*state).terminated && !(*state).terminate_process_requested)
		{
			// Test the process
			int res = _test(state, &_watchdog_state);

			// Error
			if (res == 2 || _watchdog_state.strikes >= MAX_STRIKES)
			{
				(*state).terminate_process_requested = true;
				
				if (_watchdog_state.recovery_tries < MAX_RECOVERY_TRIES)
				{
					(*state).recovery_requested = true;
					_watchdog_state.recovery_tries++;
					_watchdog_state.strikes = 0;
					_watchdog_state.bucket_task_sub = 0;
					_watchdog_state.bucket_task = 0;
					_watchdog_state.percentil_search_task = 0;
					_watchdog_state.result_search_task = 0;

					DEBUG_MSG("##################################################" << std::endl);
					DEBUG_MSG("[WATCHDOG] Trying to start the job again..." << std::endl);
					DEBUG_MSG("##################################################" << std::endl);
				}
				else
				{
					DEBUG_MSG("##################################################" << std::endl);
					DEBUG_MSG("[WATCHDOG] No more tries, something went wrong..." << std::endl);
					DEBUG_MSG("##################################################" << std::endl);
					break;
				}
			}
			// Done
			else if (res == 0)
			{
				DEBUG_MSG("##################################################" << std::endl);
				DEBUG_MSG("[WATCHDOG] All OK. Shutting down..." << std::endl);
				DEBUG_MSG("##################################################" << std::endl);
				break;
			}
			// Otherwise, processing ok...
			else
			{
				DEBUG_MSG("##################################################" << std::endl);
				DEBUG_MSG("[WATCHDOG] Everything seems to be fine so far..." << std::endl);
				DEBUG_MSG("##################################################" << std::endl);
			}
		}
	}

	return 0;
}