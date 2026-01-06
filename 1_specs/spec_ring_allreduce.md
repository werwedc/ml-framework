# Spec: Ring-AllReduce Algorithm

## Overview
Implement the Ring-AllReduce algorithm for bandwidth-efficient gradient aggregation across multiple devices. This is the core communication primitive for distributed training.

## Requirements
- Implement Ring-AllReduce with Reduce-Scatter and AllGather phases
- Optimize for bandwidth: O(N-1) transfers per device for N devices
- Support different reduction operations (Sum, Avg, Max, Min)
- Handle tensors of arbitrary dimensions
- Work with both CPU and GPU tensors

## Classes

### 1. RingAllReduce Class
```csharp
public class RingAllReduce
{
    private readonly IProcessGroup _processGroup;

    public RingAllReduce(IProcessGroup processGroup)
    {
        _processGroup = processGroup;
    }

    /// <summary>
    /// Perform Ring-AllReduce on the given tensor.
    /// Modifies the tensor in-place with the reduced result.
    /// </summary>
    public void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum);

    /// <summary>
    /// Asynchronous Ring-AllReduce.
    /// Returns a task that completes when the operation is done.
    /// </summary>
    public Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum);

    /// <summary>
    /// Reduce-Scatter phase: Each device ends up with a portion of the reduced result.
    /// </summary>
    private void ReduceScatter(Tensor tensor, ReduceOp op);

    /// <summary>
    /// AllGather phase: Each device gathers all portions of the reduced result.
    /// </summary>
    private void AllGather(Tensor tensor);
}
```

### 2. ChunkManager Class (Internal)
```csharp
/// <summary>
/// Helper class to divide tensors into chunks for ring communication.
/// </summary>
internal class ChunkManager
{
    private readonly Tensor _tensor;
    private readonly int _numChunks;
    private readonly long[] _chunkSizes;
    private readonly long[] _chunkOffsets;

    public ChunkManager(Tensor tensor, int numChunks);

    /// <summary>
    /// Get a specific chunk from the tensor.
    /// </summary>
    public Tensor GetChunk(int chunkIndex);

    /// <summary>
    /// Set a specific chunk in the tensor.
    /// </summary>
    public void SetChunk(int chunkIndex, Tensor chunk);

    /// <summary>
    /// Get the chunk index that belongs to a given rank.
    /// </summary>
    public int GetChunkForRank(int rank);

    public int NumChunks => _numChunks;
}
```

## Algorithm Details

### Ring-AllReduce Algorithm

**Topology**: Devices arranged in a ring: 0 -> 1 -> 2 -> ... -> (N-1) -> 0

**Phase 1: Reduce-Scatter**
- For each step i from 0 to N-1:
  - Each device sends its chunk to the next device in the ring
  - Each device receives a chunk from the previous device
  - Device reduces the received chunk with its local chunk
  - After N-1 steps, each device has one chunk of the fully reduced result

**Phase 2: AllGather**
- For each step i from 0 to N-1:
  - Each device sends its reduced chunk to the next device
  - Each device receives a chunk from the previous device
  - After N-1 steps, each device has all chunks of the fully reduced result

### Chunking Strategy

**Number of Chunks**: Equal to world size (one chunk per rank)

**Chunk Size**: `tensor.NumElements / worldSize`
- For uneven division: last chunk gets remaining elements
- Each device is responsible for one chunk (based on rank)

### Communication Pattern

**Send/Recv Pairs**: Each step performs:
```csharp
var sendTo = (rank + 1) % worldSize;
var recvFrom = (rank - 1 + worldSize) % worldSize;
```

**Synchronization**: Use non-blocking send/recv to overlap:
```csharp
// In each step
var sendTask = SendAsync(chunkToSend, sendTo);
var recvTask = RecvAsync(chunkToRecv, recvFrom);
await Task.WhenAll(sendTask, recvTask);
```

## Implementation Notes

### Tensor Operations
- Work with flattened view of tensors for simpler chunking
- Preserve original tensor shape and strides
- Use tensor storage operations for efficient chunk access

### Reduction Operations
- Sum: Element-wise addition
- Avg: Sum / worldSize (applied at the end)
- Max: Element-wise maximum
- Min: Element-wise minimum

### Edge Cases
- **Single device**: Skip communication, tensor is already "reduced"
- **World size = 0**: Throw exception
- **Zero-element tensor**: Skip operation
- **Non-contiguous tensor**: Make contiguous copy before operation

### Performance Optimizations
- Pre-allocate receive buffers to avoid allocations in loop
- Use pinning for GPU tensors to avoid copy overhead
- Consider using CUDA streams for GPU tensors (future optimization)

## Example Usage

```csharp
var processGroup = ProcessGroup.Init(BackendType.NCCL);
var allReduce = new RingAllReduce(processGroup);

// Create gradient tensor (different on each rank)
var gradients = Tensor.Random(1000000);

// AllReduce - after this, all ranks have the same summed gradients
allReduce.AllReduce(gradients, ReduceOp.Sum);

// For average, divide by world size
gradients.Div_(processGroup.WorldSize);
```

## Success Criteria
- [ ] Correctly implements Ring-AllReduce with Reduce-Scatter and AllGather
- [ ] Works with arbitrary tensor shapes and dimensions
- [ ] All devices end up with identical results
- [ ] Results match sequential reduction for correctness
- [ ] Handles edge cases (single device, zero elements, etc.)
- [ ] Supports all reduction operations
- [ ] Both sync and async versions work correctly

## Dependencies
- spec_communication_backend_interface.md (IProcessGroup, Tensor, ReduceOp)
- spec_process_group.md (for testing with real process groups)
- Existing tensor operations (element-wise arithmetic)

## Testing
- Unit tests in spec_ddp_tests.md will verify:
  - Correctness against sequential reduction
  - Works with various tensor shapes
  - All reduction operations produce correct results
  - Async version completes successfully
  - Edge case handling
  - Performance benchmarks (optional)
