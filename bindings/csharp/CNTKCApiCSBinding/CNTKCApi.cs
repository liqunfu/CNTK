using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

using CNTKCApiCSBinding.CNTKObjects;

using cntk_NDArrayView_t = System.IntPtr;
using cntk_NDShape_t = System.IntPtr;
using cntk_Dictionary_t = System.IntPtr;
using cntk_Variable_t = System.IntPtr;
using cntk_Variable_List_t = System.IntPtr;
using cntk_Function_t = System.IntPtr;
using cntk_FunctionPtr_t = System.IntPtr;
using cntk_Value_t = System.IntPtr;
using cntk_list_float_wrapper_t = System.IntPtr;

using cntk_unordered_map_wrapper_t = System.IntPtr;
using cntk_vector_vector_t = System.IntPtr;

namespace CNTKCApiCSBinding
{
    internal static class CNTKCApi
    {
        private const string CNTK_CORE_DLL = "Cntk.Core-2.0d.dll";

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static int next(int n);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void CNTK_SetCheckedMode(bool enable);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static bool CNTK_GetCheckedMode();

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr CNTK_DeviceKindName(DeviceKind deviceKind);

        public static string DeviceKindName(DeviceKind deviceKind)
        {
            return Marshal.PtrToStringUni(CNTK_DeviceKindName(deviceKind));
        }
        public enum DataType
        {
            Unknown = 0,
            Float = 1,
            Double = 2,
        }

        public enum DeviceKind
        {
            CPU,
            GPU
        }

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_NDShape_destroy(cntk_NDShape_t cntk_NDShape);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_NDShape_t cntk_NDShape_new_from_dimensions(
            int[] dimensions, int count);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static int cntk_NDShape_get_dim(cntk_NDShape_t cntk_NDShape, int dim);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static int cntk_NDShape_TotalSize(cntk_NDShape_t cntk_NDShape);

        //[DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        //public extern static void sk_image_ref(cntk_NDArrayView_t cntk_NDArrayView);
        //[DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        //public extern static void sk_image_unref(cntk_NDArrayView_t cntk_NDArrayView);  
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_NDArrayView_destroy(cntk_NDArrayView_t cntk_NDArrayView);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_NDArrayView_t cntk_NDArrayView_new_from_buffer(CNTK_DataType cntk_DataType,  
            cntk_NDShape_t cntk_NDShape, IntPtr dataBuffer, int bufferSizeInBytes, 
            ref CNTKDeviceDescriptor deviceDescriptor, bool readOnly);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr cntk_NDArrayView_DataBuffer(cntk_NDArrayView_t cntk_NDArrayView);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr cntk_NDArrayView_WritableDataBuffer(cntk_NDArrayView_t cntk_NDArrayView);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_Dictionary_t cntk_Dictionary_new();
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_Dictionary_destroy(cntk_Dictionary_t cntk_Dictionary);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_Variable_destroy(cntk_Function_t cntk_Variable);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_NDShape_t cntk__Variable_Shape(cntk_Variable_t cntk_Variable);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_Function_destroy(cntk_Function_t cntk_Function);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_Function_t cntk_Function_new_sigmoid(cntk_Variable_t operand, IntPtr name);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_Function_t cntk_Function_new_Load_from_filepath(IntPtr filepath, CNTKDeviceDescriptor device);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_Function_t cntk_Function_new_Load_from_filepath_default_device(IntPtr filepath);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_Function_Inputs(cntk_FunctionPtr_t function, [Out] cntk_Variable_t[] variables);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_Variable_List_t cntk_Function_Arguments(cntk_FunctionPtr_t function);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_Variable_t cntk_Function_Output(cntk_Function_t function);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_Function_Evaluate(cntk_Function_t function,
            cntk_list_float_wrapper_t inputDataMap, cntk_list_float_wrapper_t outputData);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_VariableList_destroy(cntk_Variable_List_t cntk_Variable_List);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_Variable_t cntk_VariableList_getItem(cntk_Variable_List_t cntk_Variable_List, int index);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_unordered_map_wrapper_destroy(cntk_unordered_map_wrapper_t map);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_unordered_map_wrapper_t cntk_unordered_map_wrapper_new();
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_unordered_map_wrapper_add(cntk_unordered_map_wrapper_t cntk_map,
            cntk_Variable_t cntk_Variable, cntk_Value_t cntk_Value);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_Value_t cntk_unordered_map_wrapper_get_by_key(
            cntk_unordered_map_wrapper_t cntk_map, cntk_Variable_t cntk_Variable);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_Value_destroy(cntk_Value_t cntk_Value);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_Value_t cntk_Value_CreateBatch(cntk_NDShape_t cntk_NDShape,
            cntk_list_float_wrapper_t cntk_list_float_wrapper);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_vector_vector_t cntk_value_CopyVariableValueTo(
            cntk_Value_t cntk_Value, cntk_Variable_t cntk_variable);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_list_float_wrapper_destroy(cntk_list_float_wrapper_t cntk_list_float_wrapper);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static cntk_list_float_wrapper_t cntk_list_float_wrapper_new();
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_list_float_add(cntk_list_float_wrapper_t cntk_list_float_wrapper, float value);

        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static int cntk_vector_vector_get_rows(cntk_vector_vector_t cntk_vector_vector);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_vector_vector_get_jagged_shape(
            cntk_vector_vector_t cntk_vector_vector, [Out]int[] shape);
        [DllImport(CNTK_CORE_DLL, CallingConvention = CallingConvention.Cdecl)]
        public extern static void cntk_vector_vector_get_jagged_data(
            cntk_vector_vector_t cntk_vector_vector, [Out]float[] buffer);
    }
}
