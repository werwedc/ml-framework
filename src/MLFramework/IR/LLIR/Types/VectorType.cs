namespace MLFramework.IR.LLIR.Types
{
    using MLFramework.IR.Types;

    /// <summary>
    /// Represents a vector type in the Low-Level IR (LLIR).
    /// Vector types are used for SIMD operations and contain multiple elements of the same type.
    /// </summary>
    public class VectorType : IIRType
    {
        /// <summary>Gets the element type of each vector element.</summary>
        public IIRType ElementType { get; }

        /// <summary>Gets the width (number of elements) of this vector.</summary>
        public int Width { get; }

        /// <summary>Gets the name of this type.</summary>
        public string Name => $"vec<{Width}x{ElementType}>";

        /// <summary>
        /// Initializes a new instance of the VectorType class.
        /// </summary>
        /// <param name="elementType">The element type for each vector element.</param>
        /// <param name="width">The width (number of elements) of this vector.</param>
        public VectorType(IIRType elementType, int width)
        {
            ElementType = elementType ?? throw new System.ArgumentNullException(nameof(elementType));
            if (width <= 0)
                throw new System.ArgumentException("Vector width must be positive", nameof(width));

            Width = width;
        }

        /// <summary>
        /// Determines if this type is equal to another type.
        /// </summary>
        /// <param name="other">The other type to compare with.</param>
        /// <returns>True if types are equal, false otherwise.</returns>
        public bool Equals(IIRType other)
        {
            if (other is not VectorType otherVec)
                return false;

            return ElementType.Equals(otherVec.ElementType) && Width == otherVec.Width;
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
            int hash = 17;
            hash = hash * 31 + ElementType.GetHashCode();
            hash = hash * 31 + Width.GetHashCode();
            return hash;
        }
    }
}
