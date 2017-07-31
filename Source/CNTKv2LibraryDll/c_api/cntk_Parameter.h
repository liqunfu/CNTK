#pragma once
#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_types_priv.h"

#include <string>

SK_C_PLUS_PLUS_BEGIN_GUARD

CNTK_API void cntk_Parameter_destroy(cntk_Parameter_t *cntk_Parameter)
{
	delete AsParameter(cntk_Parameter);
}

CNTK_API cntk_Parameter_t* cntk_Parameter_new_initializer(
	const cntk_NDShape_t* cntk_NDShape, cntk_DataType_t cntk_DataType, 
	cntk_ParameterInitializer_t *initializer, 
	const cntk_DeviceDescriptor_t* cntk_DeviceDescriptor, 
	const wchar_t *name) 
{
	DataType dataType = from_c(cntk_DataType);
	const NDShape *ndShape = AsNDShape(cntk_NDShape);
	ParameterInitializer *parameterInitializer = AsParameterInitializer(initializer);
	DeviceDescriptor deviceDescriptor = from_c(cntk_DeviceDescriptor);
	std::wstring parameterName(name);
	return ToParameter(new Parameter(*ndShape, dataType, 
		*parameterInitializer, deviceDescriptor, parameterName));
}

SK_C_PLUS_PLUS_END_GUARD