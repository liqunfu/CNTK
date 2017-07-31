#pragma once
SK_C_PLUS_PLUS_BEGIN_GUARD

typedef enum {
	Unknown = 0,
	Float = 1,
	Double = 2,
} cntk_DataType_t;

typedef enum {
	Device_Kind_CPU,
	Device_Kind_GPU,
	// TODO: FPGA
} cntk_device_type_t;
SK_C_PLUS_PLUS_END_GUARD
