using System;

namespace MLFramework.Checkpointing;

/// <summary>
/// Detailed memory statistics for checkpoints
/// </summary>
public class MemoryStats
{
    /// <summary>
    /// Total memory currently used by checkpoints (in bytes)
    /// </summary>
    public long CurrentMemoryUsed { get; set; }

    /// <summary>
    /// Peak memory used since last clear (in bytes)
    /// </summary>
    public long PeakMemoryUsed { get; set; }

    /// <summary>
    /// Number of checkpoints currently stored
    /// </summary>
    public int CheckpointCount { get; set; }

    /// <summary>
    /// Average memory per checkpoint (in bytes)
    /// </summary>
    public long AverageMemoryPerCheckpoint { get; set; }

    /// <summary>
    /// Total memory allocated since tracking started (in bytes)
    /// </summary>
    public long TotalMemoryAllocated { get; set; }

    /// <summary>
    /// Total memory deallocated since tracking started (in bytes)
    /// </summary>
    public long TotalMemoryDeallocated { get; set; }

    /// <summary>
    /// Total number of allocations
    /// </summary>
    public int AllocationCount { get; set; }

    /// <summary>
    /// Total number of deallocations
    /// </summary>
    public int DeallocationCount { get; set; }

    /// <summary>
    /// Timestamp when stats were collected
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Memory savings compared to storing all activations (in bytes) - deprecated, use CalculateMemorySavings instead
    /// </summary>
    public long MemorySavings { get; set; }

    /// <summary>
    /// Gets memory savings compared to storing all activations (in bytes)
    /// </summary>
    /// <param name="totalActivationSize">Total size of all activations if stored</param>
    /// <returns>Memory savings in bytes</returns>
    public long CalculateMemorySavings(long totalActivationSize)
    {
        return totalActivationSize - CurrentMemoryUsed;
    }

    /// <summary>
    /// Gets memory reduction percentage
    /// </summary>
    /// <param name="totalActivationSize">Total size of all activations if stored</param>
    /// <returns>Memory reduction percentage (0.0 to 1.0)</returns>
    public float CalculateMemoryReductionPercentage(long totalActivationSize)
    {
        if (totalActivationSize == 0)
            return 0f;
        return (float)CalculateMemorySavings(totalActivationSize) / totalActivationSize;
    }
}
