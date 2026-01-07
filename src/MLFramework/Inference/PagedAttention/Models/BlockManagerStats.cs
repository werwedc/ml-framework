namespace MlFramework.Inference.PagedAttention.Models;

/// <summary>
/// Statistics about the block manager's memory usage.
/// </summary>
public class BlockManagerStats
{
    /// <summary>
    /// Total number of blocks in the pool.
    /// </summary>
    public int TotalBlocks { get; set; }

    /// <summary>
    /// Number of blocks currently allocated.
    /// </summary>
    public int AllocatedBlocks { get; set; }

    /// <summary>
    /// Number of free blocks available.
    /// </summary>
    public int FreeBlocks => TotalBlocks - AllocatedBlocks;

    /// <summary>
    /// Number of active sequences tracked.
    /// </summary>
    public int ActiveSequences { get; set; }

    /// <summary>
    /// Total tokens stored across all blocks.
    /// </summary>
    public int TotalTokens { get; set; }

    /// <summary>
    /// Memory utilization percentage (tokens / capacity).
    /// </summary>
    public double MemoryUtilizationPercentage { get; set; }

    /// <summary>
    /// Number of block allocations performed.
    /// </summary>
    public long AllocationCount { get; set; }

    /// <summary>
    /// Number of block deallocations performed.
    /// </summary>
    public long DeallocationCount { get; set; }

    public override string ToString()
    {
        return $"BlockManagerStats: " +
               $"Free={FreeBlocks}/{TotalBlocks}, " +
               $"ActiveSeqs={ActiveSequences}, " +
               $"Utilization={MemoryUtilizationPercentage:F1}%, " +
               $"Tokens={TotalTokens}";
    }
}
