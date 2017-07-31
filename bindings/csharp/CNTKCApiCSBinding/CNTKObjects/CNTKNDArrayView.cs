using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SkiaSharp;

namespace CNTKCApiCSBinding.CNTKObjects
{
    public class CNTKNDArrayView : SKObject
    {
        [Preserve]
        internal CNTKNDArrayView(IntPtr handle, bool owns) : base(handle, owns)
        {
        }

        protected override void Dispose(bool disposing)
        {
            if (Handle != IntPtr.Zero && OwnsHandle)
            {
                CNTKCApi.cntk_NDArrayView_destroy(Handle);
            }

            base.Dispose(disposing);
        }

        public CNTKNDArrayView(
            CNTK_DataType dataType, 
            CNTKNDShape viewShape, 
            IntPtr dataBuffer, 
            int bufferSizeInBytes, 
            CNTKDeviceDescriptor device, 
            bool readOnly = false) :
            this(CNTKCApi.cntk_NDArrayView_new_from_buffer(
                dataType,
                viewShape.Handle,
                dataBuffer,
                bufferSizeInBytes,
                ref device,
                readOnly), true)
        {

        }
    }
}
