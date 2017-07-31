#pragma once
#include "..\API\CNTKLibrary.h"
#include "cntk_types.h"
#include "cntk_types_priv.h"

SK_C_PLUS_PLUS_BEGIN_GUARD

CNTK_API void cntk_Function_destroy(cntk_Function_t *cntk_Function)
{
	delete AsFunction(cntk_Function);
}

CNTK_API cntk_Function_t* cntk_Function_new_sigmoid(const cntk_Variable_t *operand, const wchar_t *name)
{
	const Variable *variable = AsVariable(operand);
	return ToFunction(Sigmoid(*variable, std::wstring(name)).get());
}

CNTK_API cntk_FunctionPtr_t* cntk_Function_new_Load_from_filepath(const wchar_t *filepath, const cntk_DeviceDescriptor_t *device)
{
	FunctionPtr function = Function::Load(std::wstring(filepath), from_c(device));
	// return ToFunction(function.get());
	return reinterpret_cast<cntk_FunctionPtr_t *>(new CNTK::FunctionPtr(function));
}

CNTK_API cntk_FunctionPtr_t* cntk_Function_new_Load_from_filepath_default_device(const wchar_t *filepath)
{
	FunctionPtr function = Function::Load(std::wstring(filepath));
	printf("input size: %d\n", (int)function->Inputs().size());

	//
	Variable inputVar = function->Arguments()[0];

	// Get shape data for the input variable
	NDShape inputShape = inputVar.Shape();
	int imageWidth = (int)inputShape[0];
	// int imageHeight = (int)inputShape[1];
	int imageChannels = (int)inputShape[2];
	int imageSize = (int)inputShape.TotalSize();
	printf("imageWidth: %d\t imageChannels: %d\t imageSize: %d\n", imageWidth, imageChannels, imageSize);

	//
	return reinterpret_cast<cntk_FunctionPtr_t *>(new CNTK::FunctionPtr(function));
}

// sk_path_convert_conic_to_quads
CNTK_API void cntk_Function_Inputs(const cntk_Function_t* function, cntk_Variable_t* variables)
{
	Variable* var = AsVariable(variables);

	const FunctionPtr *func = reinterpret_cast<const FunctionPtr *>(function);
	std::vector<Variable> vars = (*func)->Inputs();

	for (int i = 0; i < vars.size(); i++)
	{
		var[i] = vars[i];
	}
}

CNTK_API cntk_Variable_List_t* cntk_Function_Arguments(const cntk_Function_t* function)
{
	const FunctionPtr *func = reinterpret_cast<const FunctionPtr *>(function);
	std::vector<Variable> vars = (*func)->Arguments();

	NDShape inputShape = vars[0].Shape();
	int imageWidth = (int)inputShape[0];
	// int imageHeight = (int)inputShape[1];
	int imageChannels = (int)inputShape[2];
	int imageSize = (int)inputShape.TotalSize();
	printf("imageWidth: %d\t imageChannels: %d\t imageSize: %d\n", imageWidth, imageChannels, imageSize);


	return reinterpret_cast<cntk_Variable_List_t*>(new std::vector<Variable>(vars));
}

CNTK_API cntk_Variable_t *cntk_Function_Output(const cntk_Function_t* function)
{
	const FunctionPtr *func = reinterpret_cast<const FunctionPtr *>(function);
	Variable variable = (*func)->Output();
	return reinterpret_cast<cntk_Variable_t *>(new Variable(variable));
}

CNTK_API void cntk_Function_Evaluate(cntk_Function_t *function,
	const cntk_list_float_wrapper_t *inputDataMap, cntk_list_float_wrapper_t *outputData)
{
	const std::unordered_map<Variable, ValuePtr> *arguments =
		reinterpret_cast<const std::unordered_map<Variable, ValuePtr> *>(inputDataMap);
	std::unordered_map<Variable, ValuePtr> *outputs =
		reinterpret_cast<std::unordered_map<Variable, ValuePtr> *>(outputData);
	FunctionPtr *func = reinterpret_cast<FunctionPtr *>(function);
	(*func)->Evaluate(*arguments, *outputs);
}

CNTK_API void cntk_VariableList_destroy(cntk_Variable_List_t *cntk_Variable_List)
{
	std::vector<Variable> *list = reinterpret_cast<std::vector<Variable>*>(cntk_Variable_List);
	delete list;
}

CNTK_API cntk_Variable_t *cntk_VariableList_getItem(cntk_Variable_List_t *cntk_Variable_List, int index)
{
	std::vector<Variable>* list = reinterpret_cast<std::vector<Variable>*> (cntk_Variable_List);
	Variable &var = (*list)[index];
	NDShape inputShape = var.Shape();
	int imageWidth = (int)inputShape[0];
	// int imageHeight = (int)inputShape[1];
	int imageChannels = (int)inputShape[2];
	int imageSize = (int)inputShape.TotalSize();
	printf("imageWidth: %d\t imageChannels: %d\t imageSize: %d\n", imageWidth, imageChannels, imageSize);

	return reinterpret_cast<cntk_Variable_t *>(&var);
}

SK_C_PLUS_PLUS_END_GUARD