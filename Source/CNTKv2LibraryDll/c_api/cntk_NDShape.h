#pragma once
#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_types_priv.h"

#include <vector>
SK_C_PLUS_PLUS_BEGIN_GUARD

CNTK_API void cntk_NDShape_destroy(cntk_NDShape_t *cntk_NDShape)
{
	delete AsNDShape(cntk_NDShape);
}

CNTK_API cntk_NDShape_t* cntk_NDShape_new_from_dimensions(const int* dimensions, int count) {
	std::vector<size_t> dims;
	for (int i = 0; i < count; i++)
	{
		dims.push_back(dimensions[i]);
	}
	return ToNDShape(new CNTK::NDShape(dims));
}

CNTK_API int cntk_NDShape_get_dim(cntk_NDShape_t *cntk_NDShape, int dim)
{
	NDShape *shape = reinterpret_cast<NDShape *>(cntk_NDShape);
	unsigned long z = *(unsigned long *)shape;
	printf("cntk_NDShape_TotalSize0: %d", z);
	return (int)(*shape)[dim];
}

CNTK_API int cntk_NDShape_TotalSize(const cntk_NDShape_t *cntk_NDShape)
{
	const NDShape *shape = reinterpret_cast<const NDShape *>(cntk_NDShape);
	return (int)shape->TotalSize();
}

SK_C_PLUS_PLUS_END_GUARD