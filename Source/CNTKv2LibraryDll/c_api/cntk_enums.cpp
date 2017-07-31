#include "..\API\CNTKLibrary.h"
#include "cntk_enums.h"

static_assert((int)CNTK::DeviceKind::CPU == (int)Device_Kind_CPU, "Mismatch enums");
static_assert((int)CNTK::DeviceKind::GPU == (int)Device_Kind_GPU, "Mismatch enums");