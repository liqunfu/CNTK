using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTKCApiCSBinding.CNTKObjects
{
    public enum CNTK_DataType
    {
        Unknown = 0,
        Float = 1,
        Double = 2,
    }

    public enum CNTK_DeviceKind
    {
        CPU,
        GPU,
    }
}
