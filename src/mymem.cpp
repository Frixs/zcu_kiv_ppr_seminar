#include <iostream>
#include "mymem.h"

int _mem_counter;

void* mymem::malloc(size_t size)
{
	++_mem_counter;
	return std::malloc(size);
}

void mymem::free(void* ptr)
{
	std::free(ptr);
	--_mem_counter;
}

void mymem::print_counter()
{
	std::cout << "MEMORY COUNTER: " << _mem_counter << "\n";
}