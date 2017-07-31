using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SkiaSharp;

namespace CNTKCApiCSBinding.CNTKObjects
{
    public class CNTKNDShape : SKObject
    {
        [Preserve]
        internal CNTKNDShape(IntPtr handle, bool owns) : base(handle, owns)
        {
        }

        protected override void Dispose(bool disposing)
        {
            if (Handle != IntPtr.Zero && OwnsHandle)
            {
                CNTKCApi.cntk_NDShape_destroy(Handle);
            }

            base.Dispose(disposing);
        }

        public CNTKNDShape(int[] dimensions) :
            this(CNTKCApi.cntk_NDShape_new_from_dimensions(dimensions, dimensions.Count()), true)
        {

        }

        public int this[int index]
        {
            get
            {
                return CNTKCApi.cntk_NDShape_get_dim(this.Handle, index);
            }
            set
            {
                //  setitem(index, value);
            }
        }

        public int TotalSize()
        {
            return CNTKCApi.cntk_NDShape_TotalSize(this.Handle);
        }
    }
}