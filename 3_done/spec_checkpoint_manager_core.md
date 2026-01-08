# Spec: Checkpoint Manager Core

## Overview
Implement the core `CheckpointManager` class that provides the fundamental operations for activation checkpointing: registration, retrieval/recomputation, and clearing of checkpoints.

## Class: CheckpointManager

### Location
`src/MLFramework/Checkpointing/CheckpointManager.cs`

### Dependencies
- `Tensor` class (from core tensor library)
- `MemoryStats` class (from memory tracking module)

### Public Interface

```csharp
namespace MLFramework.Checkpointing;

public class CheckpointManager : IDisposable
{
    /// <summary>
    /// Initializes a new instance of CheckpointManager
    /// </summary>
    public CheckpointManager();

    /// <summary>
    /// Registers a checkpoint for the given layer ID
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">The activation tensor to checkpoint</param>
    /// <exception cref="ArgumentException">Thrown if layerId already exists</exception>
    public void RegisterCheckpoint(string layerId, Tensor activation);

    /// <summary>
    /// Retrieves a checkpointed activation or recomputes it if not stored
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="recomputeFunc">Function to recompute the activation if needed</param>
    /// <returns>The activation tensor</returns>
    /// <exception cref="KeyNotFoundException">Thrown if layerId not found and no recomputeFunc provided</exception>
    public Tensor RetrieveOrRecompute(string layerId, Func<Tensor>? recomputeFunc = null);

    /// <summary>
    /// Checks if a checkpoint exists for the given layer ID
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>True if checkpoint exists, false otherwise</returns>
    public bool HasCheckpoint(string layerId);

    /// <summary>
    /// Clears all stored checkpoints and releases memory
    /// </summary>
    public void ClearCheckpoints();

    /// <summary>
    /// Gets memory statistics for current checkpoints
    /// </summary>
    /// <returns>Memory statistics including size, count, and peak usage</returns>
    public MemoryStats GetMemoryStats();

    /// <summary>
    /// Gets the number of currently stored checkpoints
    /// </summary>
    public int CheckpointCount { get; }

    /// <summary>
    /// Disposes the manager and releases all resources
    /// </summary>
    public void Dispose();
}
```

### Internal Data Structures

```csharp
private class CheckpointEntry
{
    public string LayerId { get; set; }
    public Tensor Activation { get; set; }
    public long MemorySize { get; set; }
    public DateTime CreatedAt { get; set; }
    public int AccessCount { get; set; }
}

private readonly Dictionary<string, CheckpointEntry> _checkpoints;
private readonly object _lock = new object();
private long _totalMemoryUsed;
private long _peakMemoryUsed;
```

## Implementation Details

### RegisterCheckpoint

1. **Validation**:
   - Check if `layerId` already exists in `_checkpoints`
   - If exists, throw `ArgumentException`
   - Validate that `activation` is not null

2. **Memory Calculation**:
   - Calculate memory size of the activation tensor
   - Update `_totalMemoryUsed`
   - Update `_peakMemoryUsed` if current total exceeds peak

3. **Storage**:
   - Create new `CheckpointEntry` with metadata
   - Add to `_checkpoints` dictionary

4. **Thread Safety**:
   - Use lock for thread-safe operations

### RetrieveOrRecompute

1. **Cache Hit**:
   - If checkpoint exists for `layerId`, return stored activation
   - Increment `AccessCount` for the entry

2. **Cache Miss**:
   - If `recomputeFunc` is provided, execute it
   - Call `RegisterCheckpoint` with recomputed result
   - Return the recomputed activation
   - Throw `KeyNotFoundException` if no recomputeFunc and no checkpoint

3. **Thread Safety**:
   - Use lock to ensure thread-safe access

### ClearCheckpoints

1. **Cleanup**:
   - For each checkpoint entry, dispose of the Tensor
   - Clear the `_checkpoints` dictionary
   - Reset `_totalMemoryUsed` to zero
   - Keep `_peakMemoryUsed` for statistics

2. **Thread Safety**:
   - Use lock to ensure exclusive access

### GetMemoryStats

1. **Statistics Collection**:
   - Current memory usage
   - Peak memory usage
   - Number of checkpoints
   - Most frequently accessed checkpoints
   - Least frequently accessed checkpoints

2. **Return**:
   - Return a populated `MemoryStats` object

### Memory Size Calculation

```csharp
private long CalculateMemorySize(Tensor tensor)
{
    // Size = element_count * sizeof(float) * (batch dimensions)
    // Example: For a tensor of shape [batch, seq_len, hidden_dim]
    // Memory = batch * seq_len * hidden_dim * 4 bytes
    return tensor.ElementCount * tensor.DataTypeSize;
}
```

## Class: MemoryStats

### Location
`src/MLFramework/Checkpointing/MemoryStats.cs`

### Public Interface

```csharp
namespace MLFramework.Checkpointing;

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
    /// Memory savings compared to storing all activations (in bytes)
    /// </summary>
    public long MemorySavings { get; set; }
}
```

## Error Handling

### Exception Types

1. **ArgumentException**
   - When trying to register a duplicate layer ID
   - When providing null or invalid parameters

2. **KeyNotFoundException**
   - When trying to retrieve a non-existent checkpoint without recompute function

3. **ObjectDisposedException**
   - When using the manager after it has been disposed

### Validation Rules

1. `layerId` must be non-empty string
2. `activation` must not be null
3. Manager must not be disposed before operations

## Testing Requirements

### Unit Tests

1. **RegisterCheckpoint Tests**
   - [ ] Successfully register a checkpoint
   - [ ] Throw exception when registering duplicate layer ID
   - [ ] Correctly calculate memory size for various tensor shapes
   - [ ] Thread-safe registration from multiple threads

2. **RetrieveOrRecompute Tests**
   - [ ] Successfully retrieve existing checkpoint
   - [ ] Recompute activation when not found
   - [ ] Throw exception when not found and no recompute function
   - [ ] Increment access count correctly
   - [ ] Thread-safe retrieval

3. **ClearCheckpoints Tests**
   - [ ] Successfully clear all checkpoints
   - [ ] Reset current memory to zero
   - [ ] Preserve peak memory statistics
   - [ ] Dispose tensors properly

4. **MemoryStats Tests**
   - [ ] Correctly calculate current memory usage
   - [ ] Correctly track peak memory usage
   - [ ] Correctly calculate average memory per checkpoint
   - [ ] Reset stats appropriately after clearing

5. **Edge Cases**
   - [ ] Handle empty manager state
   - [ ] Handle large number of checkpoints
   - [ ] Handle tensors of various sizes and dimensions
   - [ ] Handle disposal and cleanup

## Implementation Notes

1. **Thread Safety**: All public methods should be thread-safe using proper locking mechanisms.

2. **Memory Management**:
   - Always dispose tensors when clearing checkpoints
   - Track memory accurately for proper statistics
   - Consider using `IDisposable` pattern for cleanup

3. **Performance**:
   - Minimize lock contention for read-heavy operations
   - Consider using `ReaderWriterLockSlim` if read/write patterns are unbalanced
   - Cache frequently accessed statistics

4. **Extensibility**:
   - Design interfaces to allow different storage backends (e.g., disk, GPU)
   - Consider adding event hooks for monitoring

## Dependencies on Other Specs

This spec depends on:
- **Memory Tracking System** (spec_3) for the `MemoryStats` class implementation
- **Recomputation Engine** (spec_4) for advanced recomputation scenarios

## Estimated Implementation Time
45-60 minutes
