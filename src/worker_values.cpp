#include "worker_values.h"

/// State values of the currently processing worker
worker::values::State* _state;

/// Processing type of the currently processing worker
worker::values::ProcessingType* _processing_type;

/// Processing type value (OpenCL specific)
std::string* _processing_type_value;



worker::values::State* worker::values::get_state()
{
	return _state;
}

worker::values::ProcessingType* worker::values::get_processing_type()
{
	return _processing_type;
}

std::string* worker::values::get_processing_type_value()
{
	return _processing_type_value;
}

void worker::values::init(worker::values::State* state, worker::values::ProcessingType* processing_type, std::string* processing_type_value)
{
	_state = state;
	_processing_type = processing_type;
	_processing_type_value = processing_type_value;
}

#pragma region State Class

worker::values::State::State()
{
	this->set_defaults();
}

void worker::values::State::set_defaults()
{
	this->terminated = true;

	this->terminate_process_requested = false;
	this->recovery_requested = false;

	this->file_loaded = false;
	this->total_values_counted = false;

	this->bucket_task_sub = 0;
	this->bucket_task = 0;
	this->bucket_found = false;

	this->percentil_search_task = 0;
	this->percentil_search_done = false;

	this->waiting_for_percentil_pickup = false;

	this->result_search_task = 0;
	this->result_search_done = false;
}

#pragma endregion