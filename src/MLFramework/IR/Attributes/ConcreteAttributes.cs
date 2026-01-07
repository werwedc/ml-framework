using System;

namespace MLFramework.IR.Attributes
{
    using MLFramework.IR.Types;

    /// <summary>
    /// Represents a tensor attribute (multi-dimensional array constant).
    /// </summary>
    public class TensorAttribute : IIRAttribute
    {
        /// <summary>Gets the type of the tensor.</summary>
        public IIRType Type { get; }

        /// <summary>Gets the tensor value as a multi-dimensional array.</summary>
        public object Value { get; }

        /// <summary>
        /// Initializes a new instance of the TensorAttribute class.
        /// </summary>
        /// <param name="type">The tensor type.</param>
        /// <param name="value">The tensor value as a multi-dimensional array.</param>
        public TensorAttribute(TensorType type, object value)
        {
            Type = type;
            Value = value;
        }
    }

    /// <summary>
    /// Represents a floating-point attribute.
    /// </summary>
    public class FloatAttribute : IIRAttribute
    {
        /// <summary>Gets the type (scalar float32).</summary>
        public IIRType Type { get; }

        /// <summary>Gets the floating-point value.</summary>
        public object Value { get; }

        /// <summary>
        /// Initializes a new instance of the FloatAttribute class.
        /// </summary>
        /// <param name="value">The floating-point value.</param>
        public FloatAttribute(float value)
        {
            Type = new ScalarType(DataType.Float32);
            Value = value;
        }
    }

    /// <summary>
    /// Represents an integer attribute.
    /// </summary>
    public class IntAttribute : IIRAttribute
    {
        /// <summary>Gets the type (scalar int32).</summary>
        public IIRType Type { get; }

        /// <summary>Gets the integer value.</summary>
        public object Value { get; }

        /// <summary>
        /// Initializes a new instance of the IntAttribute class.
        /// </summary>
        /// <param name="value">The integer value.</param>
        public IntAttribute(int value)
        {
            Type = new ScalarType(DataType.Int32);
            Value = value;
        }
    }

    /// <summary>
    /// Represents a boolean attribute.
    /// </summary>
    public class BoolAttribute : IIRAttribute
    {
        /// <summary>Gets the type (scalar bool).</summary>
        public IIRType Type { get; }

        /// <summary>Gets the boolean value.</summary>
        public object Value { get; }

        /// <summary>
        /// Initializes a new instance of the BoolAttribute class.
        /// </summary>
        /// <param name="value">The boolean value.</param>
        public BoolAttribute(bool value)
        {
            Type = new ScalarType(DataType.Bool);
            Value = value;
        }
    }

    /// <summary>
    /// Represents an array attribute.
    /// </summary>
    public class ArrayAttribute : IIRAttribute
    {
        /// <summary>Gets the type of the array elements.</summary>
        public IIRType Type { get; }

        /// <summary>Gets the array of values.</summary>
        public object Value { get; }

        /// <summary>
        /// Initializes a new instance of the ArrayAttribute class.
        /// </summary>
        /// <param name="elementType">The element type.</param>
        /// <param name="values">The array of values.</param>
        public ArrayAttribute(IIRType elementType, object values)
        {
            Type = elementType;
            Value = values;
        }
    }

    /// <summary>
    /// Represents a scalar type for attributes.
    /// </summary>
    public class ScalarType : IIRType
    {
        /// <summary>Gets the data type.</summary>
        public DataType DataType { get; }

        /// <summary>Gets the name of the type.</summary>
        public string Name => DataType.ToString();

        /// <summary>
        /// Initializes a new instance of the ScalarType class.
        /// </summary>
        /// <param name="dataType">The data type.</param>
        public ScalarType(DataType dataType)
        {
            DataType = dataType;
        }

        /// <summary>
        /// Determines if this type is equal to another type.
        /// </summary>
        public bool Equals(IIRType other)
        {
            if (other is not ScalarType otherScalar)
                return false;

            return DataType == otherScalar.DataType;
        }

        /// <summary>
        /// Returns a canonicalized version of this type.
        /// </summary>
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
