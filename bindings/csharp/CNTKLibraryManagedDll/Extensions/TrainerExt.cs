using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class Trainer
    {
        public bool TrainMinibatch(IDictionary<Variable, MinibatchData> arguments, DeviceDescriptor computeDevice)
        {
            UnorderedMapVariableMinibatchData vectorData = Helper.AsUnorderedMapVariableMinibatchData(arguments);
            return TrainMinibatch(vectorData, computeDevice);
        }
    }
}
