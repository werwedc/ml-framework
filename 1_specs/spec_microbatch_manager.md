# Spec: Micro-batch Manager

## Overview
Implement the micro-batching logic that splits a large batch into smaller micro-batches, processes them through the pipeline, and accumulates gradients.

## Class Design

### MicroBatchManager
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Manages micro-batching and gradient accumulation for pipeline parallelism
    /// </summary>
    public class MicroBatchManager : IDisposable
    {
        private readonly int _microBatchSize;
        private readonly int _numMicroBatches;
        private readonly int _totalBatchSize;
        private readonly Device _device;
        private readonly List<Tensor?> _accumulatedGradients;
        private int _currentMicroBatch;

        /// <summary>
        /// Size of each micro-batch
        /// </summary>
        public int MicroBatchSize => _microBatchSize;

        /// <summary>
        /// Number of micro-batches per full batch
        /// </summary>
        public int NumMicroBatches => _numMicroBatches;

        /// <summary>
        /// Total batch size (microBatchSize * numMicroBatches)
        /// </summary>
        public int TotalBatchSize => _totalBatchSize;

        public MicroBatchManager(int totalBatchSize, int numMicroBatches, Device device);

        /// <summary>
        /// Split a batch into micro-batches
        /// </summary>
        /// <returns>List of micro-batch tensors</returns>
        public List<Tensor> SplitBatch(Tensor batch);

        /// <summary>
        /// Combine micro-batch outputs into a single batch
        /// </summary>
        public Tensor CombineOutputs(List<Tensor> microBatchOutputs);

        /// <summary>
        /// Accumulate gradients from a micro-batch
        /// </summary>
        public void AccumulateGradients(IEnumerable<Tensor> gradients);

        /// <summary>
        /// Get the accumulated gradients (averaged over micro-batches)
        /// </summary>
        public List<Tensor> GetAccumulatedGradients();

        /// <summary>
        /// Reset accumulated gradients to zero
        /// </summary>
        public void ResetGradients();

        /// <summary>
        /// Check if all micro-batches have been processed
        /// </summary>
        public bool IsComplete => _currentMicroBatch >= _numMicroBatches;

        public void Dispose();
    }
}
```

### MicroBatchInfo
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Metadata about a micro-batch
    /// </summary>
    public class MicroBatchInfo
    {
        /// <summary>
        /// Index of this micro-batch (0 to numMicroBatches-1)
        /// </summary>
        public int Index { get; }

        /// <summary>
        /// Size of this micro-batch
        /// </summary>
        public int Size { get; }

        /// <summary>
        /// Start index in the original batch
        /// </summary>
        public int StartIndex { get; }

        /// <summary>
        /// End index (exclusive) in the original batch
        /// </summary>
        public int EndIndex { get; }

        public MicroBatchInfo(int index, int size, int startIndex, int endIndex);
    }
}
```

## Implementation Requirements

### MicroBatchManager Constructor
1. Calculate micro-batch size: `totalBatchSize / numMicroBatches`
2. Handle remainder: last micro-batch may be larger
3. Validate inputs: `totalBatchSize > 0`, `numMicroBatches > 0`
4. Initialize gradient accumulation buffers (will be allocated on first accumulation)

### SplitBatch
1. Validate that batch dimension 0 equals `_totalBatchSize`
2. Slice batch along dimension 0 into `_numMicroBatches` tensors
3. Create `MicroBatchInfo` for each micro-batch
4. Return list of micro-batches

### CombineOutputs
1. Validate that number of outputs equals `_numMicroBatches`
2. Concatenate micro-batch outputs along dimension 0
3. Handle different micro-batch sizes
4. Return combined tensor

### AccumulateGradients
1. If first micro-batch, allocate gradient buffers
2. Add gradients to accumulated buffers
3. Increment `_currentMicroBatch` counter
4. Validate gradient shapes match parameters

### GetAccumulatedGradients
1. Divide accumulated gradients by `_numMicroBatches` (average)
2. Return list of averaged gradients
3. Throw if not all micro-batches processed (optional, based on design)

### ResetGradients
1. Set accumulated gradients to zero
2. Reset `_currentMicroBatch` to 0
3. Don't deallocate gradient buffers (reuse for efficiency)

## Testing Requirements

1. **Unit Tests**
   - Test splitting batch with even division
   - Test splitting batch with remainder (uneven micro-batches)
   - Test combining outputs back to original shape
   - Test gradient accumulation (single parameter)
   - Test gradient accumulation (multiple parameters)
   - Test gradient averaging is correct
   - Test reset gradients works correctly
   - Test IsComplete property updates correctly

2. **Integration Tests**
   - Test full micro-batch cycle: split -> process -> accumulate -> get gradients
   - Test multiple cycles with reset
   - Test with actual model parameters

3. **Edge Cases**
   - Test single micro-batch (numMicroBatches = 1)
   - Test large number of micro-batches
   - Test with different batch sizes
   - Test GetAccumulatedGradients before complete (should handle gracefully)

## Files to Create
- `src/Pipeline/MicroBatchManager.cs`
- `src/Pipeline/MicroBatchInfo.cs`
- `tests/Pipeline/MicroBatchManagerTests.cs`

## Dependencies
- Existing `Tensor` class with slice/concatenate operations
- Existing `Device` class
- No new external dependencies

## Time Estimate
30-45 minutes for implementation and tests
