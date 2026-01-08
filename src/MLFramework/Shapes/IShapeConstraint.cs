using System;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Defines a constraint that can be applied to a symbolic dimension.
    /// </summary>
    public interface IShapeConstraint
    {
        /// <summary>
        /// Checks if the constraint is satisfied by the given symbolic dimension.
        /// </summary>
        /// <param name="dim">The symbolic dimension to validate.</param>
        /// <returns>True if the constraint is satisfied; otherwise, false.</returns>
        bool Validate(SymbolicDimension dim);

        /// <summary>
        /// Returns a human-readable description of this constraint.
        /// </summary>
        /// <returns>A string describing the constraint.</returns>
        string ToString();
    }
}
