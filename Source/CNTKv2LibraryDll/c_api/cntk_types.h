#pragma once
#include "CNTK_enums.h"

SK_C_PLUS_PLUS_BEGIN_GUARD
typedef struct cntk_GPUProperties_t
{
	unsigned int deviceId;
	int versionMajor;
	int versionMinor;
	int cudaCores;
	// char* name;
	size_t totalMemory;
} cntk_GPUProperties_t;

typedef struct {
	unsigned int m_deviceId;
	cntk_device_type_t m_deviceType;
} cntk_DeviceDescriptor_t;


typedef struct cntk_NDShape_t cntk_NDShape_t;
typedef struct cntk_NDArrayView_t cntk_NDArrayView_t;
typedef struct cntk_DictionaryValue_t cntk_DictionaryValue_t;
typedef struct cntk_Dictionary_t cntk_Dictionary_t;
typedef struct cntk_Value_t cntk_Value_t;
typedef struct cntk_Variable_t cntk_Variable_t;
typedef struct cntk_Variable_List_t cntk_Variable_List_t;
typedef struct cntk_Dictionary_t cntk_ParameterInitializer_t;
typedef struct cntk_Parameter_t cntk_Parameter_t;

typedef struct cntk_list_float_wrapper_t cntk_list_float_wrapper_t;

typedef struct cntk_unordered_map_wrapper_t cntk_unordered_map_wrapper_t;
typedef struct cntk_vector_vector_t cntk_vector_vector_t;

typedef struct cntk_Function_t cntk_Function_t;
typedef struct cntk_FunctionPtr_t cntk_FunctionPtr_t;

SK_C_PLUS_PLUS_END_GUARD


