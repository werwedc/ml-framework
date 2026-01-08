namespace MLFramework.Functional
{
    /// <summary>
    /// Defines the types of functional transformations available in the framework.
    /// </summary>
    public enum TransformationType
    {
        /// <summary>
        /// Vectorization transformation (vmap) - maps a function over array/vector axes.
        /// </summary>
        Vectorization,

        /// <summary>
        /// Parallelization transformation (pmap) - maps a function in parallel across multiple devices/threads.
        /// </summary>
        Parallelization,

        /// <summary>
        /// Compilation transformation (jit) - just-in-time compilation for optimization.
        /// </summary>
        Compilation,

        /// <summary>
        /// Composition transformation - combines multiple functions into a single function.
        /// </summary>
        Composition
    }
}
