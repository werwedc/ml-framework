# Spec: FSDP Reduce-Scatter Communication Primitive

## Overview
Implement the Reduce-Scatter communication primitive used to aggregate and distribute gradients during backward pass.

## Requirements

### 1. ReduceScatterOperation Class
Create a class that performs Reduce-Scatter operations:

```csharp
public class ReduceScatterOperation : IDisposable
{
    private readonly IProcessGroup _processGroup;
    private readonly Tensor _shardedBuffer;
    private readonly int _shardIndex;
    private readonly ReduceOp _reduceOp;

    /// <summary>
    /// Initialize a new Reduce-Scatter operation.
    /// </summary>
    /// <param name="processGroup">Process group for communication</param>
    /// <param name="fullShape">Shape of the full tensor before scattering</param>
    /// <param name="dataType">Data type of the tensor</param>
    /// <param name="shardIndex">Index of the shard to receive</param>
    /// <param name="reduceOp">Reduction operation (Sum, Avg, etc.)</param>
    public ReduceScatterOperation(IProcessGroup processGroup, long[] fullShape, TensorDataType dataType, int shardIndex, ReduceOp reduceOp = ReduceOp.Sum)
    {
        _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
        _shardIndex = shardIndex;
        _reduceOp = reduceOp;

        // Calculate total size
        long totalSize = 1;
        foreach (var dim in fullShape)
            totalSize *= dim;

        // Calculate shard size
        var worldSize = _processGroup.WorldSize;
        var shardSize = (totalSize + worldSize - 1) / worldSize;

        // Allocate buffer for scattered result (only our shard)
        _shardedBuffer = Tensor.Zeros(new[] { shardSize }, dataType);
    }

    /// <summary>
    /// Perform Reduce-Scatter: Reduce all tensors and scatter the result.
    /// Each device gets a reduced portion of the result.
    /// </summary>
    /// <param name="fullTensor">Full tensor to reduce and scatter</param>
    /// <returns>Reduced shard owned by this device</returns>
    public Tensor ReduceScatter(Tensor fullTensor)
    {
        if (fullTensor == null)
            throw new ArgumentNullException(nameof(fullTensor));

        var worldSize = _processGroup.WorldSize;
        var rank = _processGroup.Rank;

        // Edge case: single device
        if (worldSize == 1)
        {
            // No need to scatter, just copy
            Array.Copy(fullTensor.Data, _shardedBuffer.Data, fullTensor.Size);
            return _shardedBuffer;
        }

        // Calculate chunk size for each rank
        var totalSize = fullTensor.Size;
        var chunkSize = (totalSize + worldSize - 1) / worldSize;

        // Phase 1: Reduce-Scatter using ring algorithm
        // Each device reduces its portion and sends to the next device
        for (int step = 0; step < worldSize - 1; step++)
        {
            var sendTo = (rank + 1) % worldSize;
            var recvFrom = (rank - 1 + worldSize) % worldSize;

            // Calculate which chunk to send and receive
            var sendChunkIndex = (rank - step + worldSize) % worldSize;
            var recvChunkIndex = (rank - step - 1 + worldSize) % worldSize;

            // Get chunk to send
            var sendOffset = sendChunkIndex * chunkSize;
            var sendSize = Math.Min(chunkSize, totalSize - sendOffset);
            var sendData = new float[sendSize];
            Array.Copy(fullTensor.Data, sendOffset, sendData, 0, sendSize);
            var sendTensor = Tensor.FromArray(sendData);

            // Get buffer for receive
            var recvOffset = recvChunkIndex * chunkSize;
            var recvSize = Math.Min(chunkSize, totalSize - recvOffset);
            var recvData = new float[recvSize];
            var recvTensor = Tensor.FromArray(recvData);

            // Concurrent send and receive
            var sendTask = _processGroup.SendAsync(sendTensor, sendTo);
            var recvTask = _processGroup.RecvAsync(recvTensor, recvFrom);
            Task.WaitAll(sendTask, recvTask);

            // Reduce received data with local data
            ReduceData(fullTensor.Data, recvData, recvOffset, recvSize);

            // Clean up
            sendTensor.Dispose();
            recvTensor.Dispose();
        }

        // Phase 2: Extract our shard from the reduced result
        var myShardOffset = _shardIndex * chunkSize;
        var myShardSize = Math.Min(chunkSize, totalSize - myShardOffset);
        Array.Copy(fullTensor.Data, myShardOffset, _shardedBuffer.Data, 0, myShardSize);

        // Handle Avg operation
        if (_reduceOp == ReduceOp.Avg)
        {
            for (int i = 0; i < _shardedBuffer.Size; i++)
            {
                _shardedBuffer.Data[i] /= worldSize;
            }
        }

        return _shardedBuffer;
    }

    /// <summary>
    /// Reduce received data with local data in-place.
    /// </summary>
    private void ReduceData(float[] localData, float[] receivedData, int offset, int size)
    {
        switch (_reduceOp)
        {
            case ReduceOp.Sum:
            case ReduceOp.Avg:
                for (int i = 0; i < size; i++)
                {
                    localData[offset + i] += receivedData[i];
                }
                break;

            case ReduceOp.Product:
                for (int i = 0; i < size; i++)
                {
                    localData[offset + i] *= receivedData[i];
                }
                break;

            case ReduceOp.Max:
                for (int i = 0; i < size; i++)
                {
                    localData[offset + i] = Math.Max(localData[offset + i], receivedData[i]);
                }
                break;

            case ReduceOp.Min:
                for (int i = 0; i < size; i++)
                {
                    localData[offset + i] = Math.Min(localData[offset + i], receivedData[i]);
                }
                break;

            default:
                throw new ArgumentException($"Unsupported reduction operation: {_reduceOp}", nameof(_reduceOp));
        }
    }

    /// <summary>
    /// Perform asynchronous Reduce-Scatter.
    /// </summary>
    /// <param name="fullTensor">Full tensor to reduce and scatter</param>
    /// <returns>Task that completes with the reduced shard</returns>
    public Task<Tensor> ReduceScatterAsync(Tensor fullTensor)
    {
        return Task.Run(() => ReduceScatter(fullTensor));
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public void Dispose()
    {
        _shardedBuffer?.Dispose();
    }
}
```

### 2. ReduceScatterHelper Class
Create a helper class for Reduce-Scatter operations:

```csharp
public static class ReduceScatterHelper
{
    /// <summary>
    /// Perform Reduce-Scatter on multiple gradients in parallel.
    /// </summary>
    /// <param name="processGroup">Process group for communication</param>
    /// <param name="gradients">Gradients to reduce and scatter</param>
    /// <param name="shardIndices">Shard index for each gradient</param>
    /// <param name="reduceOp">Reduction operation</param>
    /// <returns>List of reduced and scattered gradients</returns>
    public static Task<List<Tensor>> ReduceScatterMultipleAsync(
        IProcessGroup processGroup,
        List<Tensor> gradients,
        List<int> shardIndices,
        ReduceOp reduceOp = ReduceOp.Sum)
    {
        if (gradients == null || gradients.Count == 0)
            return Task.FromResult(new List<Tensor>());

        if (gradients.Count != shardIndices.Count)
            throw new ArgumentException("Gradients and shard indices must have the same count");

        var tasks = gradients.Zip(shardIndices, (grad, shardIdx) =>
        {
            var op = new ReduceScatterOperation(processGroup, grad.Shape, grad.DataType, shardIdx, reduceOp);
            return op.ReduceScatterAsync(grad);
        }).ToList();

        return Task.WhenAll(tasks).ContinueWith(t => t.Result.ToList());
    }

    /// <summary>
    /// Verify that scattered gradients match the expected reduction.
    /// Used for testing.
    /// </summary>
    /// <param name="fullGradients">Full gradients from all devices</param>
    /// <param name="shardedGradients">Scattered gradients on each device</param>
    /// <param name="worldSize">Number of devices</param>
    /// <param name="reduceOp">Reduction operation</param>
    /// <returns>True if verification passes</returns>
    public static bool VerifyReduceScatter(
        List<Tensor> fullGradients,
        List<Tensor> shardedGradients,
        int worldSize,
        ReduceOp reduceOp = ReduceOp.Sum)
    {
        if (fullGradients.Count != shardedGradients.Count)
            return false;

        for (int i = 0; i < fullGradients.Count; i++)
        {
            var fullGrad = fullGradients[i];
            var shardedGrad = shardedGradients[i];

            var totalSize = fullGrad.Size;
            var chunkSize = (totalSize + worldSize - 1) / worldSize;

            // Each device should have its shard
            for (int rank = 0; rank < worldSize; rank++)
            {
                var offset = rank * chunkSize;
                var size = Math.Min(chunkSize, totalSize - offset);

                for (int j = 0; j < size; j++)
                {
                    var expected = fullGrad.Data[offset + j];
                    var actual = shardedGrad.Data[j];

                    if (reduceOp == ReduceOp.Avg)
                        expected /= worldSize;

                    if (Math.Abs(expected - actual) > 1e-5)
                        return false;
                }
            }
        }

        return true;
    }
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/ReduceScatterOperation.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- `MLFramework.Distributed.IProcessGroup`
- `MLFramework.Distributed.ReduceOp`
- `RitterFramework.Core.Tensor`

## Implementation Notes
1. Use ring algorithm similar to Ring-AllReduce
2. Handle different reduction operations (Sum, Avg, Max, Min, Product)
3. Implement in-place reduction for efficiency
4. Calculate chunk sizes correctly accounting for uneven division
5. Implement proper async/await pattern

## Testing Requirements
- Test Reduce-Scatter with equal-sized shards
- Test Reduce-Scatter with uneven shard sizes
- Test different reduction operations
- Test single device edge case
- Test parallel Reduce-Scatter of multiple tensors
- Test verification logic

## Estimated Time
45 minutes
