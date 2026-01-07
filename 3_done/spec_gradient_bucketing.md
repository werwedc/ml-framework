# Spec: Gradient Bucketing

## Overview
Implement gradient bucketing to reduce the number of AllReduce operations. Gradients are grouped into buckets of similar size, and each bucket is reduced independently to overlap communication with computation.

## Requirements
- Group gradients into buckets of configurable size
- Reduce buckets asynchronously to overlap communication
- Handle tensors of different sizes within a bucket
- Support bucket reuse across iterations (memory optimization)
- Track which gradients have been reduced

## Classes

### 1. GradientBucket Class
```csharp
/// <summary>
/// Represents a bucket of gradients that will be reduced together.
/// </summary>
public class GradientBucket
{
    public int BucketIndex { get; }
    public long SizeInBytes { get; }
    public Tensor[] Gradients { get; }  // Original gradient tensors
    public Tensor BucketTensor { get; }  // Flattened and concatenated view
    public Task ReductionTask { get; private set; }

    public GradientBucket(int bucketIndex, long sizeInBytes, Tensor[] gradients);

    /// <summary>
    /// Prepare bucket for reduction by flattening and concatenating gradients.
    /// </summary>
    public void Prepare();

    /// <summary>
    /// Reduce the bucket asynchronously using the process group.
    /// </summary>
    public Task ReduceAsync(IProcessGroup processGroup, ReduceOp op = ReduceOp.Sum);

    /// <summary>
    /// Copy reduced values back to original gradient tensors.
    /// </summary>
    public void CopyBack();

    /// <summary>
    /// Check if bucket reduction is complete.
    /// </summary>
    public bool IsReduced => ReductionTask?.IsCompleted ?? false;
}
```

### 2. GradientBucketManager Class
```csharp
/// <summary>
/// Manages gradient bucketing and asynchronous reduction.
/// </summary>
public class GradientBucketManager : IDisposable
{
    private readonly IProcessGroup _processGroup;
    private readonly long _bucketSizeInBytes;
    private readonly Dictionary<Tensor, int> _tensorToBucketMap;
    private readonly GradientBucket[] _buckets;

    public GradientBucketManager(
        IProcessGroup processGroup,
        IEnumerable<Tensor> parameters,
        long bucketSizeInBytes = 25 * 1024 * 1024)  // 25 MB default
    {
        _processGroup = processGroup;
        _bucketSizeInBytes = bucketSizeInBytes;
        _tensorToBucketMap = new Dictionary<Tensor, int>();
        _buckets = CreateBuckets(parameters, bucketSizeInBytes);
    }

    /// <summary>
    /// Get the bucket index for a given gradient tensor.
    /// </summary>
    public int GetBucketIndex(Tensor gradient)
    {
        return _tensorToBucketMap[gradient];
    }

    /// <summary>
    /// Reduce a specific bucket asynchronously.
    /// </summary>
    public Task ReduceBucketAsync(int bucketIndex, ReduceOp op = ReduceOp.Sum)
    {
        return _buckets[bucketIndex].ReduceAsync(_processGroup, op);
    }

    /// <summary>
    /// Reduce all buckets asynchronously.
    /// </summary>
    public async Task ReduceAllAsync(ReduceOp op = ReduceOp.Sum)
    {
        var tasks = new Task[_buckets.Length];
        for (int i = 0; i < _buckets.Length; i++)
        {
            _buckets[i].Prepare();
            tasks[i] = _buckets[i].ReduceAsync(_processGroup, op);
        }
        await Task.WhenAll(tasks);
    }

    /// <summary>
    /// Wait for all bucket reductions to complete.
    /// </summary>
    public Task WaitForAllAsync()
    {
        return Task.WhenAll(_buckets.Select(b => b.ReductionTask));
    }

    /// <summary>
    /// Copy reduced values back to all gradient tensors.
    /// </summary>
    public void CopyBackAll()
    {
        foreach (var bucket in _buckets)
        {
            bucket.CopyBack();
        }
    }

    private GradientBucket[] CreateBuckets(IEnumerable<Tensor> parameters, long bucketSize);

    public void Dispose();
}
```

## Implementation Details

### Bucket Creation Algorithm

**Goal**: Group parameters into buckets where each bucket is approximately `_bucketSizeInBytes`

**Process**:
1. Sort parameters by size (largest first helps with balancing)
2. Greedily assign parameters to buckets until bucket size limit is reached
3. Start a new bucket when limit is reached
4. Each bucket tracks the offsets for each parameter's gradient

**Example**:
```
Parameters: [100MB, 50MB, 30MB, 20MB, 15MB]
Bucket size: 100MB

Bucket 0: [100MB]
Bucket 1: [50MB, 30MB, 20MB]
Bucket 2: [15MB]
```

### Bucket Tensor Layout

Each bucket creates a single flattened tensor containing all gradients in the bucket:

```
Bucket Tensor Layout:
[Grad0 | Grad1 | Grad2 | ... | GradN]
^offsets[0] ^offsets[1] ^offsets[2]       ^offsets[N]
```

**Offsets**: Track byte or element offsets for each gradient within the bucket tensor

### Reduction Strategy

**Option 1: Eager Reduction (Simple)**
- Reduce all buckets after backward pass
- Wait for all reductions to complete
- Copy back to original gradients

**Option 2: Overlap with Computation (Advanced)**
- Reduce buckets as soon as their gradients are computed
- Use `RegisterGradHook` to trigger bucket reduction
- Allows overlapping backward computation with gradient reduction

This spec implements Option 1 first. Option 2 can be added as a future optimization.

### Memory Management

**Bucket Reuse**: Reuse the same bucket tensors across iterations to avoid allocations

**Preparation Phase**: Before reduction, flatten and concatenate gradients into bucket tensor

**Copy Back Phase**: After reduction, copy from bucket tensor back to original gradient tensors

### Edge Cases

**Single Bucket**: If all gradients fit in one bucket, behavior reduces to single AllReduce

**Parameter Larger Than Bucket**: Allow buckets to exceed limit if a single parameter is too large

**Empty Buckets**: Skip reduction for buckets with no gradients

## Usage Example

```csharp
// Create bucket manager for model parameters
var parameters = model.GetParameters().Select(p => p.Gradient).ToList();
var bucketManager = new GradientBucketManager(processGroup, parameters, bucketSizeInBytes: 25 * 1024 * 1024);

// In training loop
var loss = model.Forward(input);
loss.Backward();

// Reduce all buckets asynchronously
await bucketManager.ReduceAllAsync();

// Copy reduced gradients back
bucketManager.CopyBackAll();

// Optimizer step (uses reduced gradients)
optimizer.Step();
```

## Success Criteria
- [ ] Correctly creates buckets from parameters
- [ ] Bucket sizes respect the bucket size limit (approximately)
- [ ] Reduced values are correctly copied back to original gradients
- [ ] Async reduction works correctly
- [ ] Handles parameters of various sizes
- [ ] Reduces number of AllReduce calls compared to no bucketing

## Dependencies
- spec_communication_backend_interface.md (IProcessGroup, Tensor, ReduceOp)
- spec_process_group.md (for testing)
- spec_ring_allreduce.md (will be used by process group for reduction)

## Testing
- Unit tests in spec_ddp_tests.md will verify:
  - Correct bucket creation
  - Reduction correctness (matches sequential reduction)
  - Copy back correctness
  - Async reduction completion
  - Edge cases (single bucket, large parameters, etc.)
  - Performance improvement over no bucketing
