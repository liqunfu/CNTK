#pragma once
#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_types_priv.h"

SK_C_PLUS_PLUS_BEGIN_GUARD

CNTK_API void cntk_Dictionary_destroy(cntk_Dictionary_t *cntk_Dictionary)
{
	delete AsDictionary(cntk_Dictionary);
}

CNTK_API cntk_Dictionary_t* cntk_Dictionary_new()
{
	return ToDictionary(new CNTK::Dictionary());
}

CNTK_API void cntk_Dictionary_add_or_set_bool(cntk_Dictionary_t *cntk_Dictionary, 
	const wchar_t *name, bool value)
{
	Dictionary *dictionary = AsDictionary(cntk_Dictionary);
	(*dictionary)[name] = value;
}

CNTK_API void cntk_Dictionary_add_or_set_int(cntk_Dictionary_t *cntk_Dictionary,
	const wchar_t *name, int value)
{
	Dictionary *dictionary = AsDictionary(cntk_Dictionary);
	(*dictionary)[name] = value;
}

CNTK_API void cntk_Dictionary_add_or_set_float(cntk_Dictionary_t *cntk_Dictionary,
	const wchar_t *name, float value)
{
	Dictionary *dictionary = AsDictionary(cntk_Dictionary);
	(*dictionary)[name] = value;
}

SK_C_PLUS_PLUS_END_GUARD
