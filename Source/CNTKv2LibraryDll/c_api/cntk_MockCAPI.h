#pragma once
#include "..\API\CNTKLibraryInternals.h"
#include "cntk_enums.h"
#include "cntk_types.h"

SK_C_PLUS_PLUS_BEGIN_GUARD

CNTK_API void CNTK_SetCheckedMode(bool enable);
CNTK_API bool CNTK_GetCheckedMode();

CNTK_API wchar_t *CNTK_DeviceKindName(cntk_device_type_t device_kind);

CNTK_API bool CNTK_DeviceDescriptor_IsLocked(const cntk_DeviceDescriptor_t* cntk_DeviceDescriptor);

CNTK_API bool CNTK_DeviceDescriptor_TrySetDefaultDevice(const cntk_DeviceDescriptor_t* cntk_DeviceDescriptor, bool acquireDeviceLock);

CNTK_API void CNTK_DeviceDescriptor_GetGPUProperties(const cntk_DeviceDescriptor_t* cntk_DeviceDescriptor, cntk_GPUProperties_t* cGPUProperties);

CNTK_API void CNTK_DeviceDescriptor_SetExcludedDevices(cntk_DeviceDescriptor_t excluded[], size_t count);

// CNTK_API const cntk_DeviceDescriptor_t** CNTK_DeviceDescriptor_AllDevices();

// CNTK_API wchar_t *AsString();


SK_C_PLUS_PLUS_END_GUARD
