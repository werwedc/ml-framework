using System;
using System.Linq;

namespace MLFramework.IR.Types
{
    /// <summary>
    /// Represents a tensor type with an element type and shape.
    /// </summary>
    public class TensorType : IIRType
    {
        /// <summary>Gets the element data type.</summary>
        public DataType ElementType { get; }

        /// <summary>Gets the shape of the tensor. Can contain -1 for dynamic dimensions.</summary>
        public int[] Shape { get; }

        /// <summary>Gets whether this tensor type has any dynamic dimensions.</summary>
        public bool IsDynamic => Shape.Any(dim => dim < 0);

        /// <summary>Gets the rank (number of dimensions) of the tensor.</summary>
        public int Rank => Shape.Length;

        /// <summary>Gets the name of the type.</summary>
        public string Name => $"tensor<{ElementType}{GetShapeString()}>";

        /// <summary>
        /// Initializes a new instance of the TensorType class.
        /// </summary>
        /// <param name="elementType">The element data type.</param>
        /// <param name="shape">The shape of the tensor. Use -1 for dynamic dimensions.</param>
        public TensorType(DataType elementType, int[] shape)
        {
            if (shape == null || shape.Length == 0)
            {
                throw new ArgumentException("Shape must be non-null and non-empty.", nameof(shape));
            }

            ElementType = elementType;
            Shape = (int[])shape.Clone();
        }

        /// <summary>
        /// Creates a new tensor type with the same element type but a different shape.
        /// </summary>
        /// <param name="newShape">The new shape for the tensor.</param>
        /// <returns>A new TensorType with the new shape.</returns>
        public TensorType WithNewShape(int[] newShape)
        {
            return new TensorType(ElementType, newShape);
        }

        /// <summary>
        /// Determines if this tensor type has a known shape (no dynamic dimensions).
        /// </summary>
        /// <returns>True if all dimensions are known (non-negative), false otherwise.</returns>
        public bool HasKnownShape()
        {
            return !IsDynamic;
        }

        /// <summary>
        /// Determines if this type is equal to another type.
        /// </summary>
        /// <param name="other">The other type to compare with.</param>
        /// <returns>True if types are equal, false otherwise.</returns>
        public bool Equals(IIRType other)
        {
            if (other is not TensorType otherTensor)
                return false;

            if (ElementType != otherTensor.ElementType)
                return false;

            if (Shape.Length != otherTensor.Shape.Length)
                return false;

            for (int i = 0; i < Shape.Length; i++)
            {
                if (Shape[i] != otherTensor.Shape[i])
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Returns a canonicalized version of this type.
        /// </summary>
        /// <returns>The canonicalized type.</returns>
        public IIRType Canonicalize()
        {
            // For now, return this as-is
            // Could add more sophisticated canonicalization later
            return this;
        }

        /// <summary>
        /// Gets a string representation of the shape.
        /// </summary>
        /// <returns>The shape as a string, e.g., "[32, 784]".</returns>
        private string GetShapeString()
        {
            return "[" + string.Join(", ", Shape) + "]";
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
            int hash = 17;
            hash = hash * 31 + ElementType.GetHashCode();

            foreach (int dim in Shape)
            {
                hash = hash * 31 + dim.GetHashCode();
            }

            return hash;
        }
    }
}
