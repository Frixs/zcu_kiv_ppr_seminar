#pragma once

namespace constants
{
	/// <summary>
	/// Limit algorithm program memory
	/// </summary>
	//const unsigned int SEGMENT_SEARCH_MEMORY_LIMIT = 230 * 1024 * 1024; // in bytes (divisible by sizeof(double))
	const unsigned int SEGMENT_SEARCH_MEMORY_LIMIT = 50 * 1024 * 1024; // in bytes (divisible by sizeof(double))
	//const unsigned int SEGMENT_SEARCH_MEMORY_LIMIT = 4 * 8; // in bytes (divisible by sizeof(double))

	/// <summary>
	/// Additional sorting memory that is taken above SEGMENT_SEARCH_MEMORY_LIMIT.
	/// </summary>
	const unsigned int SEGMENT_PICK_MEMORY_LIMIT = 10 * 1024 * 1024; // in bytes (divisible by sizeof(double))

	/// <summary>
	/// Watchdog check interval
	/// </summary>
	const unsigned int WATCHDOG_TIME_INTERVAL = 30; // in seconds
}