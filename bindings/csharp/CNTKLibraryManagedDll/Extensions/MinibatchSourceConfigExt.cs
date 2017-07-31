using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class MinibatchSourceConfig
    {
        public MinibatchSourceConfig(IList<Dictionary> deserializers) : this(Helper.AsDictionaryVector(deserializers))
        {
        }
    }
}
