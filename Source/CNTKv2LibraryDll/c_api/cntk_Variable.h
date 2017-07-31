#pragma once
#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_types_priv.h"

SK_C_PLUS_PLUS_BEGIN_GUARD

CNTK_API void cntk_Variable_destroy(cntk_Variable_t *cntk_Variable)
{
	delete AsVariable(cntk_Variable);
}

CNTK_API const cntk_NDShape_t* cntk__Variable_Shape(cntk_Variable_t *cntk_Variable)
{
	Variable *variable = reinterpret_cast<Variable *>(cntk_Variable);
	printf("dim: \n");
	const NDShape &shape = variable->Shape();
	printf("dim: %d\n", (int)shape.TotalSize());
	return reinterpret_cast<const cntk_NDShape_t*>(&variable->Shape());
}
SK_C_PLUS_PLUS_END_GUARD
