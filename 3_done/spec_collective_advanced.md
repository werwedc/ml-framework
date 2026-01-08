# Spec: Advanced Collective Operations

## Overview
Implement advanced collective operations: AllGather, ReduceScatter, and Barrier with sync/async variants.

## Dependencies
- `spec_communication_interfaces.md`
- `spec_collective_basic.md`

## Technical Requirements

### 1. AllGather Operation
Combine data from all ranks and distribute full dataset to all ranks.

```csharp
namespace MLFramework.Communication.Operations
{
    /// <summary>
    /// Synchronous all-gather operation
    /// </summary>
    public static class AllGather
    {
        /// <summary>
        /// Gather tensor data from all ranks and distribute to all ranks
        /// </summary>
        /// <param name="backend">Communication backend</param>
        /// <param name="tensor">Tensor to gather</param>
        /// <param name="group">Process group (default: world)</param>
        /// <returns>Gathered tensor concatenated from all ranks</returns>
        public static Tensor<T> AllGatherTensor<T>(
            ICommunicationBackend backend,
            Tensor<T> tensor,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            return backend.AllGather(tensor);
        }

        /// <summary>
        /// Gather list of tensors from all ranks
        /// </summary>
        /// <returns>List of tensors gathered from each rank</returns>
        public static List<Tensor<T>> AllGatherTensors<T>(
            ICommunicationBackend backend,
            Tensor<T> tensor,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            // Gather all data
            var gathered = backend.AllGather(tensor);

            // Split into per-rank tensors
            return SplitGatheredTensor(gathered, backend.WorldSize);
        }

        private static List<Tensor<T>> SplitGatheredTensor<T>(Tensor<T> gathered, int worldSize)
        {
            // Calculate chunk size
            long totalElements = gathered.Shape.TotalSize;
            long chunkSize = totalElements / worldSize;

            // Split into worldSize chunks
            var result = new List<Tensor<T>>();
            for (int i = 0; i < worldSize; i++)
            {
                var startIdx = i * chunkSize;
                var endIdx = (i == worldSize - 1) ? totalElements : (i + 1) * chunkSize;
                var chunk = gathered.Slice(startIdx, endIdx - startIdx);
                result.Add(chunk);
            }

            return result;
        }
    }
}
```

### 2. ReduceScatter Operation
Combine data from all ranks and scatter chunks to different ranks.

```csharp
namespace MLFramework.Communication.Operations
{
    /// <summary>
    /// Synchronous reduce-scatter operation
    /// </summary>
    public static class ReduceScatter
    {
        /// <summary>
        /// Reduce tensor from all ranks and scatter chunks to different ranks
        /// </summary>
        /// <param name="backend">Communication backend</param>
        /// <param name="tensor">Tensor to reduce and scatter</param>
        /// <param name="operation">Reduction operation</param>
        /// <param name="group">Process group (default: world)</param>
        /// <returns>Reduced chunk for this rank</returns>
        public static Tensor<T> ReduceScatterTensor<T>(
            ICommunicationBackend backend,
            Tensor<T> tensor,
            ReduceOp operation,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            Reduce.ValidateReduceOperation<T>(operation);

            return backend.ReduceScatter(tensor, operation);
        }

        /// <summary>
        /// Reduce-scatter multiple tensors
        /// </summary>
        public static List<Tensor<T>> ReduceScatterTensors<T>(
            ICommunicationBackend backend,
            IEnumerable<Tensor<T>> tensors,
            ReduceOp operation,
            ProcessGroup? group = null)
        {
            var result = new List<Tensor<T>>();
            foreach (var tensor in tensors)
            {
                result.Add(ReduceScatterTensor(backend, tensor, operation, group));
            }
            return result;
        }
    }
}
```

### 3. Barrier Operation
Synchronize all ranks to ensure they reach the same point.

```csharp
namespace MLFramework.Communication.Operations
{
    /// <summary>
    /// Synchronization barrier operation
    /// </summary>
    public static class Barrier
    {
        /// <summary>
        /// Block until all ranks in the process group reach this point
        /// </summary>
        /// <param name="backend">Communication backend</param>
        /// <param name="group">Process group (default: world)</param>
        public static void Synchronize(
            ICommunicationBackend backend,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            backend.Barrier();
        }

        /// <summary>
        /// Barrier with timeout
        /// </summary>
        /// <returns>True if barrier completed, false if timeout</returns>
        public static bool TrySynchronize(
            ICommunicationBackend backend,
            int timeoutMs,
            ProcessGroup? group = null)
        {
            // This would need to be implemented by the backend
            // For now, we'll use a simple implementation
            var task = Task.Run(() => backend.Barrier());

            try
            {
                task.Wait(timeoutMs);
                return true;
            }
            catch (TimeoutException)
            {
                return false;
            }
        }
    }
}
```

### 4. Asynchronous AllGather
Non-blocking all-gather operation.

```csharp
namespace MLFramework.Communication.Operations.Async
{
    /// <summary>
    /// Asynchronous all-gather operation
    /// </summary>
    public static class AllGatherAsync
    {
        public static ICommunicationHandle AllGatherTensorAsync<T>(
            IAsyncCommunicationBackend backend,
            Tensor<T> tensor,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            // Create async task
            var task = Task.Run(() => backend.AllGather(tensor));
            return new PendingOperationHandle(task);
        }

        /// <summary>
        /// Gather list of tensors asynchronously
        /// </summary>
        public static ICommunicationHandle AllGatherTensorsAsync<T>(
            IAsyncCommunicationBackend backend,
            Tensor<T> tensor,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var task = Task.Run(() =>
            {
                var gathered = backend.AllGather(tensor);
                return AllGather.SplitGatheredTensor(gathered, backend.WorldSize);
            });

            return new PendingOperationHandle(task);
        }
    }
}
```

### 5. Asynchronous ReduceScatter
Non-blocking reduce-scatter operation.

```csharp
namespace MLFramework.Communication.Operations.Async
{
    /// <summary>
    /// Asynchronous reduce-scatter operation
    /// </summary>
    public static class ReduceScatterAsync
    {
        public static ICommunicationHandle ReduceScatterTensorAsync<T>(
            IAsyncCommunicationBackend backend,
            Tensor<T> tensor,
            ReduceOp operation,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            Reduce.ValidateReduceOperation<T>(operation);

            var task = Task.Run(() => backend.ReduceScatter(tensor, operation));
            return new PendingOperationHandle(task);
        }
    }
}
```

### 6. Asynchronous Barrier
Non-blocking barrier operation.

```csharp
namespace MLFramework.Communication.Operations.Async
{
    /// <summary>
    /// Asynchronous barrier operation
    /// </summary>
    public static class BarrierAsync
    {
        public static ICommunicationHandle SynchronizeAsync(
            IAsyncCommunicationBackend backend,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            return backend.BarrierAsync();
        }
    }
}
```

### 7. Scatter and Gather Helpers
Provide helper methods for common scatter/gather patterns.

```csharp
namespace MLFramework.Communication.Operations
{
    /// <summary>
    /// Helper methods for scatter and gather patterns
    /// </summary>
    public static class ScatterGatherHelpers
    {
        /// <summary>
        /// Scatter tensor across ranks (each rank gets a contiguous chunk)
        /// </summary>
        public static Tensor<T> Scatter<T>(
            ICommunicationBackend backend,
            Tensor<T> tensor,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            // Use AllGather and then slice for this rank
            var gathered = AllGather.AllGatherTensor(backend, tensor, group);
            return SliceForRank(gathered, backend.Rank, backend.WorldSize);
        }

        /// <summary>
        /// Gather tensor chunks from all ranks (inverse of Scatter)
        /// </summary>
        public static Tensor<T> Gather<T>(
            ICommunicationBackend backend,
            Tensor<T> tensor,
            int rootRank = 0,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            // AllGather returns data on all ranks
            return AllGather.AllGatherTensor(backend, tensor, group);
        }

        private static Tensor<T> SliceForRank<T>(Tensor<T> tensor, int rank, int worldSize)
        {
            long totalElements = tensor.Shape.TotalSize;
            long chunkSize = totalElements / worldSize;

            long startIdx = rank * chunkSize;
            long endIdx = (rank == worldSize - 1) ? totalElements : (rank + 1) * chunkSize;

            return tensor.Slice(startIdx, endIdx - startIdx);
        }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/Operations/AllGather.cs`
   - `src/MLFramework/Communication/Operations/ReduceScatter.cs`
   - `src/MLFramework/Communication/Operations/Barrier.cs`
   - `src/MLFramework/Communication/Operations/Async/AllGatherAsync.cs`
   - `src/MLFramework/Communication/Operations/Async/ReduceScatterAsync.cs`
   - `src/MLFramework/Communication/Operations/Async/BarrierAsync.cs`
   - `src/MLFramework/Communication/Operations/ScatterGatherHelpers.cs`

2. **Design Decisions:**
   - AllGather can return concatenated tensor or list of per-rank tensors
   - ReduceScatter combines reduction and scattering for efficiency
   - Barrier operations are lightweight synchronization primitives
   - Async versions use Task-based implementation

3. **Error Handling:**
   - Validate tensor shapes and types
   - Handle timeout scenarios gracefully
   - Provide clear error messages for rank mismatches

4. **Performance Considerations:**
   - AllGather/ReduceScatter are memory-intensive operations
   - Consider using ring-based algorithms for better scalability
   - Async versions enable compute-communication overlap

## Testing Requirements
- Unit tests for AllGather with different tensor shapes
- Unit tests for ReduceScatter with different reduction operations
- Unit tests for Barrier synchronization
- Tests for async operations with multiple ranks
- Memory usage tests for large tensors

## Success Criteria
- All operations compile and pass tests
- AllGather correctly concatenates data from all ranks
- ReduceScatter correctly reduces and scatters data
- Barrier properly synchronizes all ranks
- Async operations track completion state correctly
