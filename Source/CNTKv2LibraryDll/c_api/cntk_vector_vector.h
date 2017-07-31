#pragma once
#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_types_priv.h"

SK_C_PLUS_PLUS_BEGIN_GUARD
CNTK_API int cntk_vector_vector_get_rows(const cntk_vector_vector_t *cntk_vector_vector)
{
	const std::vector<std::vector<float>> *sequences = 
		reinterpret_cast<const std::vector<std::vector<float>> *>(cntk_vector_vector);
	return (int)(*sequences).size();
}

CNTK_API void cntk_vector_vector_get_jagged_shape(const cntk_vector_vector_t *cntk_vector_vector, int* shape)
{
	const std::vector<std::vector<float>> *sequences =
		reinterpret_cast<const std::vector<std::vector<float>> *>(cntk_vector_vector);
	for (int row = 0; row < (*sequences).size(); row++)
	{
		shape[row] = (int)(*sequences)[row].size();
	}
}

CNTK_API void cntk_vector_vector_get_jagged_data(const cntk_vector_vector_t *cntk_vector_vector, float* buffer)
{
	const std::vector<std::vector<float>> *sequences =
		reinterpret_cast<const std::vector<std::vector<float>> *>(cntk_vector_vector);
	int count = 0;
	for (int row = 0; row < (*sequences).size(); row++)
	{
		for (int i = 0; i < (*sequences)[row].size(); i++)
			buffer[count++] = (*sequences)[row][i];
	}
}

SK_C_PLUS_PLUS_END_GUARD
