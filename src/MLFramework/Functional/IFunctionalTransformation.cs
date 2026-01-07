using System;

namespace MLFramework.Functional
{
    /// <summary>
    /// Defines the contract for functional transformations that can be applied to delegates.
    /// All transformations (vmap, pmap, jit, compose, partial) must implement this interface.
    /// </summary>
    public interface IFunctionalTransformation
    {
        /// <summary>
        /// Applies the transformation to a delegate.
        /// </summary>
        /// <param name="original">The original delegate to transform.</param>
        /// <returns>A new delegate with the transformation applied.</returns>
        Delegate Transform(Delegate original);

        /// <summary>
        /// Gets the name of this transformation.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the type of transformation.
        /// </summary>
        TransformationType Type { get; }
    }
}
