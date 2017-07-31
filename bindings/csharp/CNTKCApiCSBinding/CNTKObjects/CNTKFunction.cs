using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SkiaSharp;
using System.Runtime.InteropServices;

namespace CNTKCApiCSBinding.CNTKObjects
{
    public class CNTKFunction : SKObject
    {
        [Preserve]
        internal CNTKFunction(IntPtr handle, bool owns) : base(handle, owns)
        {
        }

        protected override void Dispose(bool disposing)
        {
            if (Handle != IntPtr.Zero && OwnsHandle)
            {
                CNTKCApi.cntk_Function_destroy(Handle);
            }

            base.Dispose(disposing);
        }

        public static CNTKFunction Sigmoid(CNTKVariable operand, string name)
        {
            IntPtr hName = Marshal.StringToCoTaskMemUni(name);
            return new CNTKFunction(CNTKCApi.cntk_Function_new_sigmoid(operand.Handle, hName), true);
        }
        public static CNTKFunction Load(string filePath)
        {
            IntPtr hFilePath = Marshal.StringToCoTaskMemUni(filePath);
            return new CNTKFunction(CNTKCApi.cntk_Function_new_Load_from_filepath_default_device(hFilePath), true);
        }

        // remove this! 
        public void Inputs(IntPtr[] inputVariables)
        {
            CNTKCApi.cntk_Function_Inputs(this.Handle, inputVariables);
        }

        public CNTKVariableVariableList Arguments()
        {            
            return new CNTKVariableVariableList(CNTKCApi.cntk_Function_Arguments(this.Handle), true);
        }

        public CNTKVariable Output()
        {
            return new CNTKVariable(CNTKCApi.cntk_Function_Output(this.Handle), true);
        }

        public void Evaluate(CNTKUnorderedVariableValueMap inputDataMap, CNTKUnorderedVariableValueMap outputData)
        {
            CNTKCApi.cntk_Function_Evaluate(this.Handle, inputDataMap.Handle, outputData.Handle);
        }
    }
}
