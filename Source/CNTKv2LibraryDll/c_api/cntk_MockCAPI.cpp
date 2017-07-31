#include "cntk_MockCAPI.h"
#include "..\API\CNTKLibrary.h"
#include "CNTK_types_priv.h"

using namespace CNTK;

void CNTK_SetCheckedMode(bool enable)
{
	CNTK::SetCheckedMode(enable);
}

bool CNTK_GetCheckedMode()
{
	return CNTK::GetCheckedMode();
}

wchar_t* CNTK_DeviceKindName(cntk_device_type_t device_kind)
{
	switch (device_kind)
	{
	default:
	case Device_Kind_CPU:
		return const_cast<wchar_t*>(CNTK::DeviceKindName(CNTK::DeviceKind::CPU));
	case Device_Kind_GPU:
		return const_cast<wchar_t*>(CNTK::DeviceKindName(CNTK::DeviceKind::GPU));
	}	
}

bool CNTK_DeviceDescriptor_IsLocked(const cntk_DeviceDescriptor_t* cntk_DeviceDescriptor)
{
	DeviceDescriptor deviceDescriptor = from_c(cntk_DeviceDescriptor);
	return deviceDescriptor.IsLocked();
}

bool CNTK_DeviceDescriptor_TrySetDefaultDevice(const cntk_DeviceDescriptor_t* cntk_DeviceDescriptor, bool acquireDeviceLock)
{
	DeviceDescriptor deviceDescriptor = from_c(cntk_DeviceDescriptor);
	return DeviceDescriptor::TrySetDefaultDevice(deviceDescriptor, acquireDeviceLock);
}

void CNTK_DeviceDescriptor_GetGPUProperties(const cntk_DeviceDescriptor_t* cntk_DeviceDescriptor, cntk_GPUProperties_t* cGPUProperties)
{
	DeviceDescriptor deviceDescriptor = from_c(cntk_DeviceDescriptor);
	GPUProperties gpuProperties = DeviceDescriptor::DeviceDescriptor::GetGPUProperties(deviceDescriptor);
	from_cntk(gpuProperties, cGPUProperties);
}

void CNTK_DeviceDescriptor_SetExcludedDevices(cntk_DeviceDescriptor_t excluded[], size_t count)
{
	std::vector<DeviceDescriptor> excludedDeviceDescriptor;
	for (int i = 0; i < count; i++)
	{
		DeviceDescriptor deviceDescriptor = from_c(&excluded[i]);
		excludedDeviceDescriptor.push_back(deviceDescriptor);
	}
	DeviceDescriptor::SetExcludedDevices(excludedDeviceDescriptor);
}

//const cntk_DeviceDescriptor_t CNTK_DeviceDescriptor_AllDevices()
//{
//	cntk_DeviceDescriptor_t**allDevices = new cntk_DeviceDescriptor_t*
//}
//
//wchar_t *AsString();
