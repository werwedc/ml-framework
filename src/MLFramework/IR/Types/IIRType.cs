namespace MLFramework.IR.Types
{
    /// <summary>
    /// Base interface for all IR types.
    /// </summary>
    public interface IIRType
    {
        /// <summary>Gets the name of the type.</summary>
        string Name { get; }

        /// <summary>Determines if this type is equal to another type.</summary>
        bool Equals(IIRType other);

        /// <summary>Gets the hash code for this type.</summary>
        int GetHashCode();

        /// <summary>Returns a canonicalized version of this type.</summary>
        IIRType Canonicalize();
    }
}
