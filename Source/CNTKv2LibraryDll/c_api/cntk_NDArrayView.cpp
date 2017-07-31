#include "cntk_NDArrayView.h"
#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_enums.h"
#include "cntk_types_priv.h"

using namespace CNTK;

SK_C_PLUS_PLUS_BEGIN_GUARD
CNTK_API void cntk_NDArrayView_destroy(cntk_NDArrayView_t *cntk_NDArrayView)
{
	CNTK::NDArrayView *arrayView = AsNDArrayView(cntk_NDArrayView);
	delete arrayView;
}

CNTK_API cntk_NDArrayView_t *cntk_NDArrayView_new_from_buffer(
	const cntk_DataType_t cntk_DataType,
	const cntk_NDShape_t* cntk_NDShape,
	void* dataBuffer,
	size_t bufferSizeInBytes,
	const cntk_DeviceDescriptor_t* cntk_DeviceDescriptor,
	bool readOnly)
{
	DataType dataType = from_c(cntk_DataType);
	const NDShape *ndShape = AsNDShape(cntk_NDShape);
	DeviceDescriptor deviceDescriptor = from_c(cntk_DeviceDescriptor);
	NDArrayView *ndArrayView = new NDArrayView(dataType, *ndShape, dataBuffer, 
		bufferSizeInBytes, deviceDescriptor, readOnly);
	return ToNDArrayView(ndArrayView);

}

CNTK_API const float * cntk_NDArrayView_DataBuffer(const cntk_NDArrayView_t *cntk_NDArrayView)
{
	const NDArrayView *arrayView = AsNDArrayView(cntk_NDArrayView);
	return arrayView->DataBuffer<float>();
}

CNTK_API float * cntk_NDArrayView_WritableDataBuffer(cntk_NDArrayView_t *cntk_NDArrayView)
{
	NDArrayView *arrayView = AsNDArrayView(cntk_NDArrayView);
	return arrayView->WritableDataBuffer<float>();
}
SK_C_PLUS_PLUS_END_GUARD