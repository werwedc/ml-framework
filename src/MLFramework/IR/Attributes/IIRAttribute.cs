namespace MLFramework.IR.Attributes
{
    using MLFramework.IR.Types;

    /// <summary>
    /// Base interface for all IR attributes (constant values).
    /// </summary>
    public interface IIRAttribute
    {
        /// <summary>Gets the type of this attribute.</summary>
        IIRType Type { get; }

        /// <summary>Gets the actual value of this attribute.</summary>
        object Value { get; }
    }
}
