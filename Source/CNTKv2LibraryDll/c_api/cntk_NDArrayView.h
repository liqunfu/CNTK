#pragma once
#include "..\API\CNTKLibraryInternals.h"
#include "cntk_enums.h"
#include "cntk_types.h"

SK_C_PLUS_PLUS_BEGIN_GUARD
CNTK_API void cntk_NDArrayView_destroy(cntk_NDArrayView_t *cntk_NDArrayView);
CNTK_API cntk_NDArrayView_t *cntk_NDArrayView_new_from_buffer(
	const cntk_DataType_t cntk_DataType,
	const cntk_NDShape_t* cntk_NDShape,
	void* dataBuffer,
	size_t bufferSizeInBytes,
	const cntk_DeviceDescriptor_t* cntk_DeviceDescriptor,
	bool readOnly);

CNTK_API const float * cntk_NDArrayView_DataBuffer(const cntk_NDArrayView_t *);
SK_C_PLUS_PLUS_END_GUARD