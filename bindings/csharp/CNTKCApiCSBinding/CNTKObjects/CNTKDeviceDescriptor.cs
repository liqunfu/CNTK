using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTKCApiCSBinding.CNTKObjects
{
    public struct CNTKDeviceDescriptor
    {
        public uint m_deviceId;
        public CNTK_DeviceKind m_deviceType;
    }
}
