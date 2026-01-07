namespace MLFramework.IR.LLIR.Types
{
    using MLFramework.IR.Types;

    /// <summary>
    /// Represents a pointer type in the Low-Level IR (LLIR).
    /// Pointer types are used to represent memory addresses and buffer references.
    /// </summary>
    public class PointerType : IIRType
    {
        /// <summary>Gets the element type that this pointer points to.</summary>
        public IIRType ElementType { get; }

        /// <summary>Gets the name of this type.</summary>
        public string Name => $"ptr<{ElementType}>";

        /// <summary>
        /// Initializes a new instance of the PointerType class.
        /// </summary>
        /// <param name="elementType">The element type that this pointer points to.</param>
        public PointerType(IIRType elementType)
        {
            ElementType = elementType ?? throw new System.ArgumentNullException(nameof(elementType));
        }

        /// <summary>
        /// Determines if this type is equal to another type.
        /// </summary>
        /// <param name="other">The other type to compare with.</param>
        /// <returns>True if types are equal, false otherwise.</returns>
        public bool Equals(IIRType other)
        {
            if (other is not PointerType otherPtr)
                return false;

            return ElementType.Equals(otherPtr.ElementType);
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
            return ElementType.GetHashCode();
        }
    }
}
