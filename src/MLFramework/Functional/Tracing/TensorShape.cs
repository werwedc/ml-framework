using System;
using System.Linq;

namespace MLFramework.Functional.Tracing
{
    /// <summary>
    /// Immutable tensor shape representation.
    /// </summary>
    public class TensorShape : IEquatable<TensorShape>
    {
        private readonly int[] _dimensions;

        public int Rank => _dimensions.Length;
        public IReadOnlyList<int> Dimensions => _dimensions;

        public int this[int index] => _dimensions[index];

        public TensorShape(params int[] dimensions)
        {
            _dimensions = dimensions ?? Array.Empty<int>();
        }

        public int TotalElements => _dimensions.Aggregate(1, (a, b) => a * b);

        public bool Equals(TensorShape other)
        {
            if (other == null) return false;
            if (Rank != other.Rank) return false;

            for (int i = 0; i < Rank; i++)
            {
                if (_dimensions[i] != other._dimensions[i])
                    return false;
            }

            return true;
        }

        public override bool Equals(object obj) => Equals(obj as TensorShape);

        public override int GetHashCode()
        {
            return _dimensions.Aggregate(17, (hash, dim) => hash * 31 + dim.GetHashCode());
        }

        public override string ToString()
        {
            return $"[{string.Join(", ", _dimensions)}]";
        }

        public static TensorShape Scalar => new TensorShape();
    }
}
