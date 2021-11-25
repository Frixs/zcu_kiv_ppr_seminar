#pragma once

namespace constants
{
	/// <summary>
	/// MBs to Bytes
	/// </summary>
	const unsigned int SEGMENT_MEMORY_LIMIT_TO_BYTES = 1024 * 1024;

	/// <summary>
	/// Max allowed memory (in MBs)
	/// </summary>
	const unsigned int SEGMENT_SEARCH_MEMORY_LIMIT_VALUE = 230;

	/// <summary>
	/// Limit algorithm program memory
	/// </summary>
	const unsigned int SEGMENT_SEARCH_MEMORY_LIMIT = SEGMENT_SEARCH_MEMORY_LIMIT_VALUE * SEGMENT_MEMORY_LIMIT_TO_BYTES; // in bytes (divisible by sizeof(double))
	const unsigned int SEGMENT_SEARCH_MEMORY_LIMIT_CL = (unsigned int)ceil(SEGMENT_SEARCH_MEMORY_LIMIT_VALUE / 2.5) * SEGMENT_MEMORY_LIMIT_TO_BYTES; // in bytes (divisible by sizeof(double))
	//const unsigned int SEGMENT_SEARCH_MEMORY_LIMIT = 50 * SEGMENT_MEMORY_LIMIT_TO_BYTES; // in bytes (divisible by sizeof(double))
	//const unsigned int SEGMENT_SEARCH_MEMORY_LIMIT_CL = (unsigned int)ceil(50 / 2.5) * SEGMENT_MEMORY_LIMIT_TO_BYTES; // in bytes (divisible by sizeof(double))
	//const unsigned int SEGMENT_SEARCH_MEMORY_LIMIT = 4 * 8; // in bytes (divisible by sizeof(double))
	//const unsigned int SEGMENT_SEARCH_MEMORY_LIMIT_CL = 4 * 8; // in bytes (divisible by sizeof(double))

	/// <summary>
	/// Additional sorting memory that is taken above SEGMENT_SEARCH_MEMORY_LIMIT.
	/// </summary>
	const unsigned int SEGMENT_PICK_MEMORY_LIMIT = 10 * 1024 * 1024; // in bytes (divisible by sizeof(double))

	/// <summary>
	/// Watchdog check interval
	/// </summary>
	const unsigned int WATCHDOG_TIME_INTERVAL = 15; // in seconds
}