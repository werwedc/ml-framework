namespace MLFramework.HAL.CUDA;

/// <summary>
/// Memory usage statistics for tracking allocator performance.
/// </summary>
public class MemoryUsageStats
{
    /// <summary>
    /// Gets the total allocated memory in bytes.
    /// </summary>
    public long TotalAllocated { get; init; }

    /// <summary>
    /// Gets the pool size in bytes.
    /// </summary>
    public long PoolSize { get; init; }

    /// <summary>
    /// Gets the number of allocated blocks.
    /// </summary>
    public int BlockCount { get; init; }

    /// <summary>
    /// Gets whether graph mode is enabled.
    /// </summary>
    public bool IsGraphMode { get; init; }

    /// <summary>
    /// Returns a string representation of the memory usage statistics.
    /// </summary>
    /// <returns>Formatted string with memory usage information</returns>
    public override string ToString()
    {
        return $"Memory: {TotalAllocated / (1024 * 1024):F2} MB, " +
               $"Blocks: {BlockCount}, " +
               $"GraphMode: {IsGraphMode}";
    }
}
