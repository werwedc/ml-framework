using System;

namespace MLFramework.IR.Types
{
    /// <summary>
    /// Represents a scalar (single-value) type in the IR.
    /// </summary>
    public class ScalarType : IIRType
    {
        /// <summary>Gets the data type of the scalar.</summary>
        public DataType DataType { get; }

        /// <summary>Gets the name of the type.</summary>
        public string Name { get; }

        /// <summary>
        /// Initializes a new instance of the ScalarType class.
        /// </summary>
        /// <param name="dataType">The data type of the scalar.</param>
        public ScalarType(DataType dataType)
        {
            DataType = dataType;
            Name = $"scalar_{dataType.ToString().ToLower()}";
        }

        /// <summary>
        /// Checks if this type equals another type.
        /// </summary>
        public bool Equals(IIRType other)
        {
            if (other is ScalarType otherScalar)
            {
                return DataType == otherScalar.DataType;
            }
            return false;
        }

        /// <summary>
        /// Gets the hash code for this type.
        /// </summary>
        public override int GetHashCode()
        {
            return HashCode.Combine(Name, DataType);
        }

        /// <summary>
        /// Canonicalizes this type (returns a normalized representation).
        /// </summary>
        public IIRType Canonicalize()
        {
            return this; // Scalar types are already canonical
        }
    }
}
