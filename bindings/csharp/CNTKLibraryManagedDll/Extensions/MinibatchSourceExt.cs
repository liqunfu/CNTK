using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class MinibatchSource
    {
        public StreamInformation StreamInfo(string streamName)
        {
            return _StreamInfo(streamName);
        }
    }
}
