using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SkiaSharp;
using System.Runtime.InteropServices;

namespace CNTKCApiCSBinding.CNTKObjects
{
    public class CNTKValue : SKObject
    {
        [Preserve]
        internal CNTKValue(IntPtr handle, bool owns) : base(handle, owns)
        {
        }

        protected override void Dispose(bool disposing)
        {
            if (Handle != IntPtr.Zero && OwnsHandle)
            {
                CNTKCApi.cntk_Value_destroy(Handle);
            }

            base.Dispose(disposing);
        }

        public static CNTKValue CreateBatch(CNTKNDShape shape, CNTKListFloatWrapper listValues)
        {
            return new CNTKValue(CNTKCApi.cntk_Value_CreateBatch(shape.Handle, listValues.Handle), true);
        }

        public IList<IList<float>> GetDenseDataFloat(CNTKVariable variable)
        {
            IntPtr cntk_vector_vector = CNTKCApi.cntk_value_CopyVariableValueTo(
                this.Handle, variable.Handle);
            int rows = CNTKCApi.cntk_vector_vector_get_rows(cntk_vector_vector);
            int[] shape = new int[rows];
            CNTKCApi.cntk_vector_vector_get_jagged_shape(cntk_vector_vector, shape);

            int totalCount = shape.Sum();
            float[] dataBuffer = new float[totalCount];
            CNTKCApi.cntk_vector_vector_get_jagged_data(cntk_vector_vector, dataBuffer);

            var sequences = new List<IList<float>>();
            int count = 0;
            for (int row = 0; row < rows; row++)
            {
                IList<float> rowList = new List<float>();
                sequences.Add(rowList);
                for (int i = 0; i < shape[row]; i++)
                {
                    rowList.Add(dataBuffer[count++]);
                }
            }
            return sequences;
        }
    }
}
