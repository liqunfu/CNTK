#pragma once

#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_types_priv.h"

SK_C_PLUS_PLUS_BEGIN_GUARD

CNTK_API void cntk_DictionaryValue_destroy(cntk_DictionaryValue_t *cntk_DictionaryValue)
{
	delete AsDictionaryValue(cntk_DictionaryValue);
}

CNTK_API cntk_DictionaryValue_t* cntk_Dictionary_new_as_Bool(bool value)
{
	return ToDictionaryValue(new DictionaryValue(value));
}

CNTK_API cntk_DictionaryValue_t* cntk_Dictionary_new_as_Int(int value)
{
	return ToDictionaryValue(new DictionaryValue(value));
}

CNTK_API cntk_DictionaryValue_t* cntk_Dictionary_new_as_Float(float value)
{
	return ToDictionaryValue(new DictionaryValue(value));
}

SK_C_PLUS_PLUS_END_GUARD