namespace MLFramework.HAL;

/// <summary>
/// Interface for memory allocation strategies
/// </summary>
public interface IMemoryAllocator : IDisposable
{
    /// <summary>
    /// Allocate a memory buffer of the specified size
    /// </summary>
    /// <param name="size">Size in bytes</param>
    /// <returns>Memory buffer allocated on the device</returns>
    IMemoryBuffer Allocate(long size);

    /// <summary>
    /// Free a memory buffer back to the allocator
    /// </summary>
    /// <remarks>
    /// For caching allocators, this may return the buffer to a pool
    /// instead of freeing it to the OS
    /// </remarks>
    void Free(IMemoryBuffer buffer);

    /// <summary>
    /// Total size of cached memory (bytes)
    /// </summary>
    long CacheSize { get; }

    /// <summary>
    /// Total size of currently allocated memory (bytes)
    /// </summary>
    long AllocatedSize { get; }

    /// <summary>
    /// Empty the cache, freeing all unused memory back to the OS
    /// </summary>
    void EmptyCache();

    /// <summary>
    /// Device this allocator is associated with
    /// </summary>
    IDevice Device { get; }
}
