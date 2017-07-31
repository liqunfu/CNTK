#pragma once

#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_types_priv.h"

SK_C_PLUS_PLUS_BEGIN_GUARD

CNTK_API cntk_unordered_map_wrapper_t *cntk_unordered_map_wrapper_new()
{
	std::unordered_map<Variable, ValuePtr> *map = new std::unordered_map<Variable, ValuePtr>();
	return reinterpret_cast<cntk_unordered_map_wrapper_t *>(map);
}

CNTK_API void cntk_unordered_map_wrapper_destroy(cntk_unordered_map_wrapper_t *map_wrapper)
{
	std::unordered_map<Variable, ValuePtr> *map =
		reinterpret_cast<std::unordered_map<Variable, ValuePtr> *>(map_wrapper);
	delete map;
}

CNTK_API void cntk_unordered_map_wrapper_add(cntk_unordered_map_wrapper_t *cntk_map,
	cntk_Variable_t *cntk_Variable, cntk_Value_t *cntk_Value)
{
	std::unordered_map<Variable, ValuePtr> *map = reinterpret_cast<std::unordered_map<Variable, ValuePtr> *>(cntk_map);
	Variable *variable = reinterpret_cast<Variable *>(cntk_Variable);
	ValuePtr *valuePtr;
	if (cntk_Value == NULL)
		valuePtr = new ValuePtr(NULL);
	else 
		valuePtr = reinterpret_cast<ValuePtr *>(cntk_Value);
	(*map).insert(std::unordered_map<Variable, ValuePtr>::value_type(*variable, *valuePtr));
}

CNTK_API cntk_Value_t *cntk_unordered_map_wrapper_get_by_key(cntk_unordered_map_wrapper_t *cntk_map,
	const cntk_Variable_t *cntk_Variable)
{
	std::unordered_map<Variable, ValuePtr> *map = reinterpret_cast<std::unordered_map<Variable, ValuePtr> *>(cntk_map);
	const Variable *variable = reinterpret_cast<const Variable *>(cntk_Variable);
	if ((*map).find(*variable) == (*map).end())
	{
		return NULL;
	}
	ValuePtr value = (*map)[*variable];
	return reinterpret_cast<cntk_Value_t *>(new ValuePtr(value));
}

SK_C_PLUS_PLUS_END_GUARD