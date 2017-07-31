using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class StreamInformation
    {
        override public bool Equals(object s) 
        {
            if (s == null)
            {
                return false;
            }

            StreamInformation streamInformation = s as StreamInformation;
            if ((System.Object)streamInformation == null)
            {
                return false;
            }

            return CNTKLib.AreEqual(this, streamInformation);
        }

        // Value equality.
        public bool Equals(StreamInformation s)
        {
            // If parameter is null return false:
            if ((object)s == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, s);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        //public static bool operator ==(StreamInformation s1, StreamInformation s2)
        //{
        //    return s1.Equals(s2);
        //}

        //public static bool operator !=(StreamInformation s1, StreamInformation s2)
        //{
        //    return !s1.Equals(s2);
        //}
    }
}
