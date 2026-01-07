namespace MLFramework.IR.LLIR.Types
{
    using MLFramework.IR.Types;

    /// <summary>
    /// Represents a scalar type in the Low-Level IR (LLIR).
    /// Scalar types represent single values of a specific data type.
    /// </summary>
    public class ScalarType : IIRType
    {
        /// <summary>Gets the underlying data type of this scalar.</summary>
        public DataType DataType { get; }

        /// <summary>Gets whether this scalar type is a floating-point type.</summary>
        public bool IsFloat => DataType == DataType.Float32 || DataType == DataType.Float64 ||
                              DataType == DataType.Float16 || DataType == DataType.BFloat16;

        /// <summary>Gets whether this scalar type is an integer type.</summary>
        public bool IsInteger => DataType == DataType.Int8 || DataType == DataType.Int16 ||
                               DataType == DataType.Int32 || DataType == DataType.Int64 ||
                               DataType == DataType.UInt8 || DataType == DataType.UInt16 ||
                               DataType == DataType.UInt32 || DataType == DataType.UInt64;

        /// <summary>Gets the name of this type.</summary>
        public string Name => $"{DataType.ToString().ToLower()}";

        /// <summary>
        /// Initializes a new instance of the ScalarType class.
        /// </summary>
        /// <param name="dataType">The underlying data type.</param>
        public ScalarType(DataType dataType)
        {
            DataType = dataType;
        }

        /// <summary>
        /// Determines if this type is equal to another type.
        /// </summary>
        /// <param name="other">The other type to compare with.</param>
        /// <returns>True if types are equal, false otherwise.</returns>
        public bool Equals(IIRType other)
        {
            if (other is not ScalarType otherScalar)
                return false;

            return DataType == otherScalar.DataType;
        }

        /// <summary>
        /// Returns a canonicalized version of this type.
        /// </summary>
        /// <returns>The canonicalized type.</returns>
        public IIRType Canonicalize()
        {
            return this;
        }

        public override string ToString()
        {
            return Name;
        }

        public override bool Equals(object obj)
        {
            return Equals(obj as IIRType);
        }

        public override int GetHashCode()
        {
            return DataType.GetHashCode();
        }
    }
}
