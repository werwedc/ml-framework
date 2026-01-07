# Spec: PagedAttention Core Data Models

## Overview
Define the core data structures required for PagedAttention implementation, focusing on MemoryBlock, BlockManagerStats, and supporting models.

## Target Directory
`src/MlFramework/Inference/PagedAttention/Models/`

## Classes to Implement

### 1. MemoryBlock
```csharp
namespace MlFramework.Inference.PagedAttention.Models;

/// <summary>
/// Represents a fixed-size block of GPU memory for storing KV cache tokens.
/// </summary>
public class MemoryBlock
{
    /// <summary>
    /// Unique identifier for this block.
    /// </summary>
    public int BlockId { get; }

    /// <summary>
    /// The sequence ID currently using this block.
    /// Null if block is free.
    /// </summary>
    public int? SequenceId { get; set; }

    /// <summary>
    /// Starting token index within the sequence.
    /// </summary>
    public int StartTokenIndex { get; set; }

    /// <summary>
    /// Number of tokens stored in this block (up to BlockSize).
    /// </summary>
    public int TokenCount { get; set; }

    /// <summary>
    /// Flag indicating if this block is currently allocated.
    /// </summary>
    public bool IsAllocated => SequenceId.HasValue;

    /// <summary>
    /// Reference to the actual tensor storing keys/values.
    /// </summary>
    public Tensor KeyTensor { get; set; }
    public Tensor ValueTensor { get; set; }

    public MemoryBlock(int blockId, int blockSize)
    {
        BlockId = blockId;
        TokenCount = 0;
    }

    /// <summary>
    /// Resets the block to free state.
    /// </summary>
    public void Reset()
    {
        SequenceId = null;
        StartTokenIndex = 0;
        TokenCount = 0;
    }
}
```

### 2. BlockManagerStats
```csharp
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
```

### 3. BlockAllocationResult
```csharp
namespace MlFramework.Inference.PagedAttention.Models;

/// <summary>
/// Result of a block allocation operation.
/// </summary>
public class BlockAllocationResult
{
    /// <summary>
    /// The allocated block ID.
    /// </summary>
    public int BlockId { get; set; }

    /// <summary>
    /// Flag indicating if allocation was successful.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if allocation failed.
    /// </summary>
    public string? ErrorMessage { get; set; }

    public static BlockAllocationResult Successful(int blockId)
    {
        return new BlockAllocationResult
        {
            BlockId = blockId,
            Success = true
        };
    }

    public static BlockAllocationResult Failed(string errorMessage)
    {
        return new BlockAllocationResult
        {
            Success = false,
            ErrorMessage = errorMessage
        };
    }
}
```

## Requirements
1. **Namespace**: All models must be in `MlFramework.Inference.PagedAttention.Models`
2. **Dependencies**: Reference `MlFramework.Tensor` for tensor types
3. **Immutability**: BlockId should be immutable once assigned
4. **Thread Safety**: Consider concurrent access for SequenceId and TokenCount
5. **Validation**: Add validation logic in constructors and setters where appropriate

## Testing Requirements
1. Unit tests for MemoryBlock state transitions (allocate, free, reset)
2. Tests for BlockManagerStats calculations
3. Tests for BlockAllocationResult creation patterns

## Estimated Time
30-45 minutes

## Dependencies
- None (foundation models)

## Success Criteria
- All models compile without errors
- Proper encapsulation of block state
- Clear API surface for block management operations
