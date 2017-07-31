#pragma once
#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_types_priv.h"

SK_C_PLUS_PLUS_BEGIN_GUARD
CNTK_API void cntk_Value_destroy(cntk_Value_t *cntk_Value)
{
	Value *value = reinterpret_cast<Value *>(cntk_Value);
	delete value;
}

CNTK_API cntk_Value_t* cntk_Value_CreateBatch(cntk_NDShape_t *cntk_NDShape,
	const cntk_list_float_wrapper_t *cntk_list_float_wrapper)
{
	const NDShape *shape = reinterpret_cast<NDShape*>(cntk_NDShape);
	const std::vector<float> *batchData = reinterpret_cast<const std::vector<float> *>(cntk_list_float_wrapper);

	ValuePtr valuePtr = Value::CreateBatch(*shape, *batchData, DeviceDescriptor::CPUDevice());

	// do we have to create another ptr
	return reinterpret_cast<cntk_Value_t *>(new CNTK::ValuePtr(valuePtr));
}

CNTK_API cntk_vector_vector_t *cntk_value_CopyVariableValueTo(cntk_Value_t * cntk_Value, const cntk_Variable_t *cntk_variable)
{
	ValuePtr *valuePtr = reinterpret_cast<ValuePtr *>(cntk_Value);
	const Variable *variable = reinterpret_cast<const Variable *>(cntk_variable);
	std::vector<std::vector<float>> *sequences = new std::vector<std::vector<float>>();
	(*valuePtr)->CopyVariableValueTo(*variable, *sequences);
	cntk_vector_vector_t *cntk_vector_vector = reinterpret_cast<cntk_vector_vector_t *>(sequences);
	return cntk_vector_vector;
}
SK_C_PLUS_PLUS_END_GUARD