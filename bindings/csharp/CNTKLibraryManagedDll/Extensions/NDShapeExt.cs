using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class NDShape
    {
        //public NDShape(int numAxes, int dimension) : this((uint)numAxes, (uint)dimension)
        //{
        //    if (numAxes < 0 || dimension < 0)
        //    {
        //        throw new System.ArgumentException("The paraemter numAxes or dimension should not be a negative value");
        //    }
        //}

        // Creates a new NDShape.
        public NDShape(System.Collections.Generic.IEnumerable<uint> dimensions) 
            : this(Helper.AsSizeTVector(dimensions)) {}

        public NDShape(int numAxes) : this((uint)numAxes)
        {
            if (numAxes < 0)
            {
                throw new System.ArgumentException("The paraemter numAxes should not be a negative value");
            }
        }

        public static implicit operator NDShape(uint[] dim)
        {
            return new NDShape(dim);
        }

        // Property Rank.
        public int Rank
        {
            get { return (int)_Rank(); }
        }

        // Property Dimensions.
        public System.Collections.Generic.IList<uint> Dimensions
        {
            get
            {
                var dimList = _Dimensions();
                var retList = new System.Collections.Generic.List<uint>(dimList.Count);
                foreach (var element in dimList)
                {
                    retList.Add((uint)element);
                }
                return retList;
            }
        }

        // Property IsUnknown.
        public bool IsUnknown
        {
            get { return _IsUnknown(); }
        }

        // Property HasInferredDimension.
        public bool HasInferredDimension
        {
            get { return _HasInferredDimension(); }
        }

        // Property HasFreeDimension.
        public bool HasFreeDimension
        {
            get { return _HasFreeDimension(); }
        }

        // Property HasUnboundDimension.
        public bool HasUnboundDimension
        {
            get { return _HasUnboundDimension(); }
        }

        // Property TotalSize.
        public int TotalSize
        {
            get { return (int)_TotalSize(); }
        }

        // Indexer operator
        public int this[int key]
        {
            get { return (int)_DimensionSize((uint)key); }
            set { }
        }

        // Returns a subshape.
        public NDShape SubShape(int beginAxisId, int endAxisId)
        {
            if (beginAxisId < 0 || endAxisId < 0)
            {
                throw new System.ArgumentException("The paraemter beginAxisId or endAxisId should not be a negative value");
            }
            return _SubShape((uint)beginAxisId, (uint)endAxisId);
        }

        // Returns a subshape.
        public NDShape SubShape(int beginAxisId = 0)
        {
            if (beginAxisId < 0)
            {
                throw new System.ArgumentException("The paraemter beginAxisId should not be a negative value");
            }
            return _SubShape((uint)beginAxisId);
        }

        // Value equality.
        public override bool Equals(System.Object obj)
        {
            // If parameter is null return false.
            if (obj == null)
            {
                return false;
            }

            // If parameter cannot be cast to Point return false.
            NDShape p = obj as NDShape;
            if ((System.Object)p == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, p);
        }

        // Value Equality.
        public bool Equals(NDShape p)
        {
            // If parameter is null return false:
            if ((object)p == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, p);
        }

        // Returns hash code value.
        public override int GetHashCode()
        {
            //Todo: another hash function??
            return this._Dimensions().GetHashCode();
        }

        // Constants
        public static readonly int InferredDimension = -1;
        public static readonly int FreeDimension = -3;
    }
}
