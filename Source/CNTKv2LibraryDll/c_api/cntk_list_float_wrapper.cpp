#include "cntk_list_float_wrapper.h"

#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_types_priv.h"

SK_C_PLUS_PLUS_BEGIN_GUARD
CNTK_API void cntk_list_float_wrapper_destroy(cntk_list_float_wrapper_t *cntk_list_float_wrapper)
{
	std::vector<float> *list = reinterpret_cast<std::vector<float> *>(cntk_list_float_wrapper);
	delete list;
}

CNTK_API cntk_list_float_wrapper_t * cntk_list_float_wrapper_new()
{
	std::vector<float> *list = new std::vector<float>();
	return reinterpret_cast<cntk_list_float_wrapper_t *>(list);
}

CNTK_API void cntk_list_float_add(cntk_list_float_wrapper_t *cntk_list_float_wrapper, float value)
{
	std::vector<float> *list = reinterpret_cast<std::vector<float> *>(cntk_list_float_wrapper);
	(*list).push_back(value);
}

SK_C_PLUS_PLUS_END_GUARD