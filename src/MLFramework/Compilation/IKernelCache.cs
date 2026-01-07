namespace MLFramework.Compilation;

/// <summary>
/// Interface for kernel cache
/// </summary>
public interface IKernelCache<TKernel>
{
    /// <summary>
    /// Gets a kernel from the cache
    /// </summary>
    TKernel? Get(ShapeSignature sig);

    /// <summary>
    /// Adds a kernel to the cache
    /// </summary>
    void Set(ShapeSignature sig, TKernel kernel);

    /// <summary>
    /// Checks if the cache contains a kernel for the given signature
    /// </summary>
    bool Contains(ShapeSignature sig);

    /// <summary>
    /// Removes a kernel from the cache
    /// </summary>
    void Remove(ShapeSignature sig);

    /// <summary>
    /// Clears the cache
    /// </summary>
    void Clear();

    /// <summary>
    /// Gets cache statistics
    /// </summary>
    CacheStats GetStats();
}
