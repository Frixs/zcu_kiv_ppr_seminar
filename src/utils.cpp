#include <iostream>
#include <cstdlib>
#include<stdio.h>
#include "utils.h"

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
	srand(time(NULL));

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