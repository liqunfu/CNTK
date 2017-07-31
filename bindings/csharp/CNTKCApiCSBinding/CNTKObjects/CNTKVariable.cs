using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SkiaSharp;
using System.Runtime.InteropServices;

namespace CNTKCApiCSBinding.CNTKObjects
{
    // [StructLayout(LayoutKind.Sequential)]
    public class CNTKVariable : SKObject
    {
        [Preserve]
        internal CNTKVariable(IntPtr handle, bool owns) : base(handle, owns)
        {
        }

        protected override void Dispose(bool disposing)
        {
            if (Handle != IntPtr.Zero && OwnsHandle)
            {
                CNTKCApi.cntk_Variable_destroy(Handle);
            }

            base.Dispose(disposing);
        }

        public CNTKNDShape Shape()
        {
            return new CNTKNDShape(CNTKCApi.cntk__Variable_Shape(this.Handle), true);
        }
    }

    public class CNTKVariableVariableList : SKObject
    {
        [Preserve]
        internal CNTKVariableVariableList(IntPtr handle, bool owns) : base(handle, owns)
        {
        }

        protected override void Dispose(bool disposing)
        {
            if (Handle != IntPtr.Zero && OwnsHandle)
            {
                CNTKCApi.cntk_Variable_destroy(Handle);
            }

            base.Dispose(disposing);
        }
   
        public CNTKVariable GetItem(int index)
        {
            return new CNTKVariable(CNTKCApi.cntk_VariableList_getItem(this.Handle, index), true);
        }
    }
}
