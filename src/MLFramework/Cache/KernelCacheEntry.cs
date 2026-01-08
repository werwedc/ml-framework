using System;

namespace MLFramework.Cache
{
    /// <summary>
    /// Represents a single entry in the kernel cache, containing a compiled kernel
    /// and metadata about its usage.
    /// </summary>
    /// <typeparam name="TKernel">The type of the compiled kernel (e.g., IntPtr, object, or backend-specific type).</typeparam>
    public class KernelCacheEntry<TKernel>
    {
        /// <summary>
        /// Gets the shape signature that identifies this kernel.
        /// </summary>
        public ShapeSignature Signature { get; }

        /// <summary>
        /// Gets or sets the compiled kernel.
        /// </summary>
        public TKernel CompiledKernel { get; set; }

        /// <summary>
        /// Gets the timestamp when this entry was last accessed.
        /// </summary>
        public DateTime LastUsed { get; private set; }

        /// <summary>
        /// Gets the number of times this kernel has been used.
        /// </summary>
        public int UseCount { get; private set; }

        /// <summary>
        /// Gets the time (in milliseconds) it took to compile this kernel.
        /// </summary>
        public long CompilationTimeMs { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="KernelCacheEntry{TKernel}"/> class.
        /// </summary>
        /// <param name="signature">The shape signature for this kernel.</param>
        /// <param name="compiledKernel">The compiled kernel.</param>
        /// <param name="compilationTimeMs">Time taken to compile (in milliseconds).</param>
        public KernelCacheEntry(ShapeSignature signature, TKernel compiledKernel, long compilationTimeMs)
        {
            Signature = signature;
            CompiledKernel = compiledKernel;
            CompilationTimeMs = compilationTimeMs;
            LastUsed = DateTime.UtcNow;
            UseCount = 0;
        }

        /// <summary>
        /// Updates the last used timestamp to the current time.
        /// Should be called each time the kernel is retrieved from the cache.
        /// </summary>
        public void UpdateAccessTime()
        {
            LastUsed = DateTime.UtcNow;
        }

        /// <summary>
        /// Increments the usage count for this kernel.
        /// Should be called each time the kernel is retrieved from the cache.
        /// </summary>
        public void IncrementUseCount()
        {
            UseCount++;
        }

        /// <summary>
        /// Returns a string representation of this cache entry.
        /// </summary>
        /// <returns>A string describing the entry.</returns>
        public override string ToString()
        {
            return $"KernelCacheEntry({Signature}, Uses={UseCount}, LastUsed={LastUsed:u}, CompTime={CompilationTimeMs}ms)";
        }
    }
}
