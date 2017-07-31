using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SkiaSharp;
using System.Runtime.InteropServices;

namespace CNTKCApiCSBinding.CNTKObjects
{
    public class CNTKDictionary : SKObject
    {
        [Preserve]
        internal CNTKDictionary(IntPtr handle, bool owns) : base(handle, owns)
        {
        }

        protected override void Dispose(bool disposing)
        {
            if (Handle != IntPtr.Zero && OwnsHandle)
            {
                CNTKCApi.cntk_Dictionary_destroy(Handle);
            }

            base.Dispose(disposing);
        }

        public CNTKDictionary() :
            this(CNTKCApi.cntk_Dictionary_new(), true)
        {

        }
    }
}
