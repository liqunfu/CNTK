using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SkiaSharp;
using System.Runtime.InteropServices;

namespace CNTKCApiCSBinding.CNTKObjects
{
    public class CNTKListFloatWrapper : SKObject
    {
        [Preserve]
        internal CNTKListFloatWrapper(IntPtr handle, bool owns) : base(handle, owns)
        {
        }

        protected override void Dispose(bool disposing)
        {
            if (Handle != IntPtr.Zero && OwnsHandle)
            {
                CNTKCApi.cntk_list_float_wrapper_destroy(Handle);
            }

            base.Dispose(disposing);
        }

        public CNTKListFloatWrapper(List<float> list) : this(CNTKCApi.cntk_list_float_wrapper_new(), true) 
        {
            for (int i = 0; i < list.Count(); i++)
            {
                CNTKCApi.cntk_list_float_add(this.Handle, list[i]);
            }
        }
    }
}
