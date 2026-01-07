# Spec: Activation Checkpoint Manager

## Overview
Implement activation checkpointing for memory optimization in pipeline parallelism. This stores intermediate activations for backward pass and supports recomputation to save memory.

## Class Design

### CheckpointStrategy
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Strategy for activation checkpointing
    /// </summary>
    public enum CheckpointStrategy
    {
        /// <summary>
        /// Store all activations (maximum memory, fastest backward)
        /// </summary>
        StoreAll,

        /// <summary>
        /// Recompute all activations during backward (minimum memory, slowest)
        /// </summary>
        RecomputeAll,

        /// <summary>
        /// Store every Nth activation (balanced)
        /// </summary>
        Selective,

        /// <summary>
        /// Store activations based on memory threshold
        /// </summary>
        MemoryBased
    }
}
```

### ActivationCheckpointManager
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Manages activation checkpointing for pipeline stages
    /// </summary>
    public class ActivationCheckpointManager : IDisposable
    {
        private readonly CheckpointStrategy _strategy;
        private readonly int _checkpointInterval;
        private readonly long _memoryThreshold;
        private readonly Dictionary<int, Tensor> _checkpoints;
        private readonly PipelineStage _stage;

        /// <summary>
        /// Current checkpoint strategy
        /// </summary>
        public CheckpointStrategy Strategy => _strategy;

        public ActivationCheckpointManager(
            CheckpointStrategy strategy,
            PipelineStage stage,
            int checkpointInterval = 1,
            long memoryThreshold = 1024 * 1024 * 1024); // 1GB default

        /// <summary>
        /// Check if activation should be stored for a given micro-batch
        /// </summary>
        public bool ShouldCheckpoint(int microBatchIndex);

        /// <summary>
        /// Store an activation for later backward pass
        /// </summary>
        public void StoreActivation(int microBatchIndex, Tensor activation);

        /// <summary>
        /// Retrieve a stored activation
        /// </summary>
        public Tensor? GetActivation(int microBatchIndex);

        /// <summary>
        /// Check if activation is available (stored)
        /// </summary>
        public bool HasActivation(int microBatchIndex);

        /// <summary>
        /// Recompute activation for a micro-batch
        /// </summary>
        /// <param name="input">Input to the stage</param>
        /// <param name="microBatchIndex">Micro-batch index</param>
        /// <returns>Recomputed activation</returns>
        public Tensor RecomputeActivation(Tensor input, int microBatchIndex);

        /// <summary>
        /// Get or compute activation (handles both cases)
        /// </summary>
        public Tensor GetOrComputeActivation(Tensor input, int microBatchIndex);

        /// <summary>
        /// Clear all checkpoints
        /// </summary>
        public void Clear();

        /// <summary>
        /// Estimate memory used by checkpoints
        /// </summary>
        public long EstimateMemoryUsage();

        /// <summary>
        /// Get number of stored checkpoints
        /// </summary>
        public int CheckpointCount => _checkpoints.Count;

        public void Dispose();
    }
}
```

### CheckpointMetadata
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Metadata about a stored activation checkpoint
    /// </summary>
    public class CheckpointMetadata
    {
        /// <summary>
        /// Micro-batch index
        /// </summary>
        public int MicroBatchIndex { get; }

        /// <summary>
        /// Memory size in bytes
        /// </summary>
        public long MemorySize { get; }

        /// <summary>
        /// Timestamp when stored
        /// </summary>
        public DateTime Timestamp { get; }

        /// <summary>
        /// Shape of the activation tensor
        /// </summary>
        public long[] Shape { get; }

        public CheckpointMetadata(int microBatchIndex, long memorySize, long[] shape);
    }
}
```

## Implementation Requirements

### Checkpoint Decision Logic

#### StoreAll
- Always return `true` for `ShouldCheckpoint`
- Store every activation

#### RecomputeAll
- Always return `false` for `ShouldCheckpoint`
- Never store activations, always recompute

#### Selective
- Store every Nth activation: `microBatchIndex % checkpointInterval == 0`
- Always checkpoint first and last micro-batch
- Store others based on interval

#### MemoryBased
- Store activation if total memory < memoryThreshold
- Remove oldest checkpoints when memory would exceed threshold
- Always keep first and last checkpoint

### StoreActivation
1. Check if should checkpoint for this micro-batch
2. Clone the activation tensor (to avoid modification)
3. Store in `_checkpoints` dictionary
4. Track metadata (memory, timestamp)

### GetOrComputeActivation
1. Check if activation is stored
2. If stored, return it
3. If not stored, recompute it:
   - Run forward pass of stage module
   - Return computed activation

### RecomputeActivation
1. Run forward pass: `_stage.Forward(input)`
2. Return computed activation
3. Optionally cache result (if strategy changed mid-execution)

### Memory Estimation
1. Sum up memory of all stored activations
2. Use `Tensor.NumElements * sizeof(float)` for estimation
3. Return total in bytes

### Clear and Dispose
1. Dispose all stored tensors
2. Clear dictionary
3. Release any resources

## Testing Requirements

1. **Unit Tests**
   - Test StoreAll strategy checkpoints all
   - Test RecomputeAll strategy checkpoints none
   - Test Selective strategy with interval
   - Test MemoryBased strategy threshold behavior
   - Test store and retrieve activation
   - Test activation not found returns null
   - Test GetOrCompute with stored activation
   - Test GetOrCompute with recomputation
   - Test memory estimation
   - Test clear removes all checkpoints
   - Test dispose cleans up tensors

2. **Integration Tests**
   - Test checkpoint manager with actual pipeline stage
   - Test recomputed activation matches stored activation
   - Test memory usage with real tensors
   - Test with multiple micro-batches

3. **Edge Cases**
   - Test with zero micro-batches
   - Test with very large memory threshold (store all)
   - Test with very small memory threshold (store few)
   - Test checkpoint interval = 0 (should store all)
   - Test duplicate micro-batch indices

## Files to Create
- `src/Pipeline/CheckpointStrategy.cs`
- `src/Pipeline/ActivationCheckpointManager.cs`
- `src/Pipeline/CheckpointMetadata.cs`
- `tests/Pipeline/ActivationCheckpointManagerTests.cs`

## Dependencies
- `PipelineStage` from spec_pipeline_stage_core
- Existing `Tensor` class
- No new external dependencies

## Time Estimate
30-45 minutes for implementation and tests

## Notes
- This is a foundational component for memory optimization
- Integration with GPipeScheduler needed in future spec
- Consider adding automatic strategy selection based on memory pressure
