namespace MLFramework.Cache
{
    /// <summary>
    /// Defines a cache for compiled kernels indexed by shape signatures.
    /// </summary>
    /// <typeparam name="TKernel">The type of the compiled kernel.</typeparam>
    public interface IKernelCache<TKernel>
    {
        /// <summary>
        /// Gets a compiled kernel from the cache, or null if not found.
        /// Updates the access time and use count if found.
        /// </summary>
        /// <param name="sig">The shape signature of the kernel.</param>
        /// <returns>The compiled kernel if found; otherwise, null.</returns>
        TKernel? Get(ShapeSignature sig);

        /// <summary>
        /// Adds or updates a compiled kernel in the cache.
        /// </summary>
        /// <param name="sig">The shape signature of the kernel.</param>
        /// <param name="kernel">The compiled kernel to cache.</param>
        /// <param name="compilationTimeMs">Time taken to compile the kernel (in milliseconds).</param>
        void Set(ShapeSignature sig, TKernel kernel, long compilationTimeMs);

        /// <summary>
        /// Determines whether a compiled kernel for the given signature exists in the cache.
        /// </summary>
        /// <param name="sig">The shape signature to check.</param>
        /// <returns>True if the cache contains a kernel for the signature; otherwise, false.</returns>
        bool Contains(ShapeSignature sig);

        /// <summary>
        /// Removes a compiled kernel from the cache.
        /// </summary>
        /// <param name="sig">The shape signature of the kernel to remove.</param>
        /// <returns>True if the kernel was removed; otherwise, false.</returns>
        bool Remove(ShapeSignature sig);

        /// <summary>
        /// Clears all entries from the cache.
        /// </summary>
        void Clear();

        /// <summary>
        /// Gets statistics about the cache's performance and usage.
        /// </summary>
        /// <returns>A <see cref="CacheStats"/> object containing cache statistics.</returns>
        CacheStats GetStats();
    }
}
