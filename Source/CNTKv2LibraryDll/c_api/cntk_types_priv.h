#pragma once
#include "..\API\CNTKLibrary.h"
#include "CNTK_types.h"

using namespace CNTK;

static inline cntk_DataType_t from_cntk(DataType dataType)
{
	switch (dataType)
	{
	default:
	case DataType::Unknown:
		return Unknown;
	case DataType::Float:
		return Float;
	case DataType::Double:
		return Double;
	}
}

static inline DataType from_c(cntk_DataType_t cntk_DataType)
{
	switch(cntk_DataType)
	{
	default:
	case Unknown:
		return DataType::Unknown;
	case Float:
		return DataType::Float;
	case Double:
		return DataType::Double;
	}
}

static inline DeviceKind from_c(cntk_device_type_t cntk_device_type)
{
	switch (cntk_device_type)
	{
	default:
	case Device_Kind_CPU:
		return DeviceKind::CPU;
	case Device_Kind_GPU:
		return DeviceKind::GPU;
	}
}

static inline cntk_device_type_t from_cntk(DeviceKind deviceKind)
{
	switch (deviceKind)
	{
	default:
	case DeviceKind::CPU:
		return Device_Kind_CPU;
	case DeviceKind::GPU:
		return Device_Kind_GPU;
	}
}

static inline DeviceDescriptor from_c(const cntk_DeviceDescriptor_t* cDeviceDescriptor) {
	switch (cDeviceDescriptor->m_deviceType)
	{
	default:
	case Device_Kind_CPU:
		return DeviceDescriptor::CPUDevice();
	case Device_Kind_GPU:
		return DeviceDescriptor::GPUDevice(cDeviceDescriptor->m_deviceId);
	}
}

static inline void from_cntk(const DeviceDescriptor& deviceDescriptor, cntk_DeviceDescriptor_t* cDeviceDescriptor) {
	if (cDeviceDescriptor) {
		*cDeviceDescriptor = {
			deviceDescriptor.Id(),
			from_cntk(deviceDescriptor.Type())
		};
	}
}

static inline void from_cntk(const GPUProperties& gpuProperties, cntk_GPUProperties_t* cGPUProperties)
{
	if (cGPUProperties)
	{
		*cGPUProperties = {
			gpuProperties.deviceId,
			gpuProperties.versionMajor,
			gpuProperties.versionMinor,
			gpuProperties.cudaCores,
			// gpuProperties.name.c_str(),
			gpuProperties.totalMemory
		};
	}	
}

static inline const NDShape *AsNDShape(const cntk_NDShape_t *cntk_NDShape)
{
	return reinterpret_cast<const NDShape *>(cntk_NDShape);
}

static inline NDShape *AsNDShape(cntk_NDShape_t *cntk_NDShape)
{
	return reinterpret_cast<NDShape *>(cntk_NDShape);
}

static inline cntk_NDShape_t* ToNDShape(NDShape* p) {
	return reinterpret_cast<cntk_NDShape_t*>(p);
}

static inline const cntk_NDShape_t* ToNDShape(const NDShape* p) {
	return reinterpret_cast<const cntk_NDShape_t*>(p);
}

static inline const cntk_NDArrayView_t *ToNDArrayView(const NDArrayView *ndArrayView)
{
	return reinterpret_cast<const cntk_NDArrayView_t *>(ndArrayView);
}

static inline cntk_NDArrayView_t *ToNDArrayView(NDArrayView *ndArrayView)
{
	return reinterpret_cast<cntk_NDArrayView_t *>(ndArrayView);
}

static inline const NDArrayView *AsNDArrayView(const cntk_NDArrayView_t *cntk_NDArrayView)
{
	return reinterpret_cast<const NDArrayView *>(cntk_NDArrayView);
}

static inline NDArrayView *AsNDArrayView(cntk_NDArrayView_t *cntk_NDArrayView)
{
	return reinterpret_cast< NDArrayView *>(cntk_NDArrayView);
}

static inline const cntk_DictionaryValue_t *ToDictionaryValue(const DictionaryValue *dictionaryValue)
{
	return reinterpret_cast<const cntk_DictionaryValue_t *>(dictionaryValue);
}

static inline cntk_DictionaryValue_t *ToDictionaryValue(DictionaryValue *dictionaryValue)
{
	return reinterpret_cast<cntk_DictionaryValue_t *>(dictionaryValue);
}

static inline DictionaryValue *AsDictionaryValue(cntk_DictionaryValue_t *cntk_DictionaryValue)
{
	return reinterpret_cast<DictionaryValue *>(cntk_DictionaryValue);
}

static inline const DictionaryValue *AsDictionaryValue(const cntk_DictionaryValue_t *cntk_DictionaryValue)
{
	return reinterpret_cast<const DictionaryValue *>(cntk_DictionaryValue);
}


static inline const cntk_Dictionary_t *ToDictionary(const Dictionary *dictionary)
{
	return reinterpret_cast<const cntk_Dictionary_t *>(dictionary);
}

static inline cntk_Dictionary_t *ToDictionary(Dictionary *dictionary)
{
	return reinterpret_cast<cntk_Dictionary_t *>(dictionary);
}

static inline Dictionary *AsDictionary(cntk_Dictionary_t *cntk_Dictionary)
{
	return reinterpret_cast<Dictionary *>(cntk_Dictionary);
}

static inline const Dictionary *AsDictionary(const cntk_Dictionary_t *cntk_Dictionary)
{
	return reinterpret_cast<const Dictionary *>(cntk_Dictionary);
}


static inline const cntk_Variable_t *ToVariable(const Variable *Variable)
{
	return reinterpret_cast<const cntk_Variable_t *>(Variable);
}

static inline cntk_Variable_t *ToVariable(Variable *Variable)
{
	return reinterpret_cast<cntk_Variable_t *>(Variable);
}

static inline Variable *AsVariable(cntk_Variable_t *cntk_Variable)
{
	return reinterpret_cast<Variable *>(cntk_Variable);
}

static inline const Variable *AsVariable(const cntk_Variable_t *cntk_Variable)
{
	return reinterpret_cast<const Variable *>(cntk_Variable);
}

static inline const cntk_ParameterInitializer_t *ToParameterInitializer(
	const ParameterInitializer *parameterInitializer)
{
	return reinterpret_cast<const cntk_ParameterInitializer_t *>(parameterInitializer);
}

static inline cntk_ParameterInitializer_t *ToParameterInitializer(
	ParameterInitializer *parameterInitializer)
{
	return reinterpret_cast<cntk_ParameterInitializer_t *>(parameterInitializer);
}

static inline ParameterInitializer *AsParameterInitializer(
	cntk_ParameterInitializer_t *cntk_ParameterInitializer)
{
	return reinterpret_cast<ParameterInitializer *>(cntk_ParameterInitializer);
}

static inline const ParameterInitializer *AsParameterInitializer(
	const cntk_ParameterInitializer_t *cntk_ParameterInitializer)
{
	return reinterpret_cast<const ParameterInitializer *>(cntk_ParameterInitializer);
}

static inline const cntk_Parameter_t *ToParameter(const Parameter *parameter)
{
	return reinterpret_cast<const cntk_Parameter_t *>(parameter);
}

static inline cntk_Parameter_t *ToParameter(Parameter *parameter)
{
	return reinterpret_cast<cntk_Parameter_t *>(parameter);
}

static inline Parameter *AsParameter(cntk_Parameter_t *cntk_Parameter)
{
	return reinterpret_cast<Parameter *>(cntk_Parameter);
}

static inline const Parameter *AsVariable(const cntk_Parameter_t *cntk_Parameter)
{
	return reinterpret_cast<const Parameter *>(cntk_Parameter);
}


static inline const cntk_Function_t *ToFunction(const Function *function)
{
	return reinterpret_cast<const cntk_Function_t *>(function);
}

static inline cntk_Function_t *ToFunction(Function *function)
{
	return reinterpret_cast<cntk_Function_t *>(function);
}

static inline Function *AsFunction(cntk_Function_t *cntk_Function)
{
	return reinterpret_cast<Function *>(cntk_Function);
}

static inline const Function *AsFunction(const cntk_Function_t *cntk_Function)
{
	return reinterpret_cast<const Function *>(cntk_Function);
}

//static inline const cntk_FunctionPtr_t *ToFunctionPtr(const FunctionPtr *function)
//{
//	return reinterpret_cast<const cntk_FunctionPtr_t *>(function);
//}
//
//static inline cntk_FunctionPtr_t *ToFunctionPtr(FunctionPtr function)
//{
//	return reinterpret_cast<cntk_FunctionPtr_t *>(function);
//}
//
//static inline FunctionPtr *AsFunctionPtr(cntk_FunctionPtr_t *cntk_Function)
//{
//	return reinterpret_cast<FunctionPtr *>(cntk_Function);
//}
//
//static inline const FunctionPtr *AsFunctionPtr(const cntk_FunctionPtr_t *cntk_Function)
//{
//	return reinterpret_cast<const FunctionPtr *>(cntk_Function);
//}


