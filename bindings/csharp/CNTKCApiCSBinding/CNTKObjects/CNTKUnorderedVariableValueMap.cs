using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SkiaSharp;
using System.Runtime.InteropServices;

namespace CNTKCApiCSBinding.CNTKObjects
{
    public class CNTKUnorderedVariableValueMap : SKObject
    {
        [Preserve]
        internal CNTKUnorderedVariableValueMap(IntPtr handle, bool owns) : base(handle, owns)
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

        public CNTKUnorderedVariableValueMap() : this (CNTKCApi.cntk_unordered_map_wrapper_new(), true)
        {

        }

        public void Add(CNTKVariable variable, CNTKValue value)
        {
            CNTKCApi.cntk_unordered_map_wrapper_add(this.Handle, variable.Handle, value == null ? IntPtr.Zero : value.Handle);
        }

        public CNTKValue this[CNTKVariable variable]
        {
            get
            {
                return new CNTKValue(CNTKCApi.cntk_unordered_map_wrapper_get_by_key(this.Handle, variable.Handle), true);
            }
            set
            {
                //  setitem(index, value);
            }
        }
    }
}
