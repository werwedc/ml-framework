# Spec: Basic Collective Operations

## Overview
Implement basic collective operations: Broadcast, Reduce, and AllReduce with synchronous and asynchronous variants.

## Dependencies
- `spec_communication_interfaces.md`
- `spec_process_group.md`

## Technical Requirements

### 1. Broadcast Operation
Send data from root rank to all other ranks.

```csharp
namespace MLFramework.Communication.Operations
{
    /// <summary>
    /// Synchronous broadcast operation
    /// </summary>
    public static class Broadcast
    {
        /// <summary>
        /// Broadcast tensor from root rank to all ranks in the process group
        /// </summary>
        /// <param name="backend">Communication backend</param>
        /// <param name="tensor">Tensor to broadcast</param>
        /// <param name="rootRank">Rank that will broadcast the data</param>
        /// <param name="group">Process group (default: world)</param>
        public static void BroadcastTensor<T>(
            ICommunicationBackend backend,
            Tensor<T> tensor,
            int rootRank,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (rootRank < 0 || rootRank >= backend.WorldSize)
                throw new ArgumentOutOfRangeException(nameof(rootRank));

            if (group != null && !group.ContainsRank(rootRank))
                throw new ArgumentException($"Root rank {rootRank} is not in the group");

            // Broadcast operation
            backend.Broadcast(tensor, rootRank);
        }

        /// <summary>
        /// Broadcast multiple tensors from root rank to all ranks
        /// </summary>
        public static void BroadcastTensors<T>(
            ICommunicationBackend backend,
            IEnumerable<Tensor<T>> tensors,
            int rootRank,
            ProcessGroup? group = null)
        {
            foreach (var tensor in tensors)
            {
                BroadcastTensor(backend, tensor, rootRank, group);
            }
        }
    }
}
```

### 2. Reduce Operation
Combine data from all ranks and store result on root rank.

```csharp
namespace MLFramework.Communication.Operations
{
    /// <summary>
    /// Synchronous reduce operation
    /// </summary>
    public static class Reduce
    {
        /// <summary>
        /// Reduce tensor from all ranks to root rank
        /// </summary>
        /// <param name="backend">Communication backend</param>
        /// <param name="tensor">Tensor to reduce</param>
        /// <param name="operation">Reduction operation</param>
        /// <param name="rootRank">Rank that will receive the result</param>
        /// <param name="group">Process group (default: world)</param>
        /// <returns>Reduced tensor on root rank, null on other ranks</returns>
        public static Tensor<T> ReduceTensor<T>(
            ICommunicationBackend backend,
            Tensor<T> tensor,
            ReduceOp operation,
            int rootRank,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (rootRank < 0 || rootRank >= backend.WorldSize)
                throw new ArgumentOutOfRangeException(nameof(rootRank));

            if (group != null && !group.ContainsRank(rootRank))
                throw new ArgumentException($"Root rank {rootRank} is not in the group");

            // Validate operation is supported for type T
            ValidateReduceOperation<T>(operation);

            return backend.Reduce(tensor, operation, rootRank);
        }

        private static void ValidateReduceOperation<T>(ReduceOp operation)
        {
            // Validate that the operation is valid for type T
            // Some operations like Product may not make sense for all types
            if (operation == ReduceOp.Product && typeof(T) == typeof(bool))
            {
                throw new ArgumentException($"ReduceOp.Product is not supported for type {typeof(T).Name}");
            }
        }

        /// <summary>
        /// Reduce multiple tensors from all ranks to root rank
        /// </summary>
        public static List<Tensor<T>> ReduceTensors<T>(
            ICommunicationBackend backend,
            IEnumerable<Tensor<T>> tensors,
            ReduceOp operation,
            int rootRank,
            ProcessGroup? group = null)
        {
            var result = new List<Tensor<T>>();
            foreach (var tensor in tensors)
            {
                result.Add(ReduceTensor(backend, tensor, operation, rootRank, group));
            }
            return result;
        }
    }
}
```

### 3. AllReduce Operation
Combine data from all ranks and distribute result to all ranks.

```csharp
namespace MLFramework.Communication.Operations
{
    /// <summary>
    /// Synchronous all-reduce operation
    /// </summary>
    public static class AllReduce
    {
        /// <summary>
        /// All-reduce tensor across all ranks
        /// </summary>
        /// <param name="backend">Communication backend</param>
        /// <param name="tensor">Tensor to all-reduce</param>
        /// <param name="operation">Reduction operation</param>
        /// <param name="group">Process group (default: world)</param>
        /// <returns>All-reduced tensor</returns>
        public static Tensor<T> AllReduceTensor<T>(
            ICommunicationBackend backend,
            Tensor<T> tensor,
            ReduceOp operation,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            // Validate operation is supported for type T
            Reduce.ValidateReduceOperation<T>(operation);

            return backend.AllReduce(tensor, operation);
        }

        /// <summary>
        /// All-reduce with automatic division by world size (common pattern)
        /// </summary>
        public static Tensor<T> AverageGradients<T>(
            ICommunicationBackend backend,
            Tensor<T> gradients)
        {
            var summed = AllReduceTensor(backend, gradients, ReduceOp.Sum);
            return summed.Divide(backend.WorldSize);
        }

        /// <summary>
        /// All-reduce multiple tensors
        /// </summary>
        public static List<Tensor<T>> AllReduceTensors<T>(
            ICommunicationBackend backend,
            IEnumerable<Tensor<T>> tensors,
            ReduceOp operation,
            ProcessGroup? group = null)
        {
            var result = new List<Tensor<T>>();
            foreach (var tensor in tensors)
            {
                result.Add(AllReduceTensor(backend, tensor, operation, group));
            }
            return result;
        }
    }
}
```

### 4. Asynchronous Broadcast
Non-blocking broadcast operation.

```csharp
namespace MLFramework.Communication.Operations.Async
{
    /// <summary>
    /// Asynchronous broadcast operation
    /// </summary>
    public static class BroadcastAsync
    {
        public static ICommunicationHandle BroadcastTensorAsync<T>(
            IAsyncCommunicationBackend backend,
            Tensor<T> tensor,
            int rootRank,
            ProcessGroup? group = null)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (rootRank < 0 || rootRank >= backend.WorldSize)
                throw new ArgumentOutOfRangeException(nameof(rootRank));

            if (group != null && !group.ContainsRank(rootRank))
                throw new ArgumentException($"Root rank {rootRank} is not in the group");

            return backend.BroadcastAsync(tensor, rootRank);
        }
    }
}
```

### 5. Asynchronous AllReduce
Non-blocking all-reduce operation.

```csharp
namespace MLFramework.Communication.Operations.Async
{
    /// <summary>
    /// Asynchronous all-reduce operation
    /// </summary>
    public static class AllReduceAsync
    {
        public static ICommunicationHandle AllReduceTensorAsync<T>(
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

            return backend.AllReduceAsync(tensor, operation);
        }

        /// <summary>
        /// Wait for all-reduce to complete and divide by world size
        /// </summary>
        public static Tensor<T> AverageGradientsAsync<T>(
            IAsyncCommunicationBackend backend,
            Tensor<T> gradients)
        {
            var handle = AllReduceTensorAsync(backend, gradients, ReduceOp.Sum);
            handle.Wait();

            var summed = handle.GetResult<T>();
            return summed.Divide(backend.WorldSize);
        }
    }
}
```

### 6. Operation Result Handle
Concrete implementation of ICommunicationHandle.

```csharp
namespace MLFramework.Communication.Operations
{
    /// <summary>
    /// Concrete implementation of ICommunicationHandle for synchronous operations
    /// </summary>
    public class CompletedOperationHandle : ICommunicationHandle
    {
        private readonly Tensor<T> _result;

        public CompletedOperationHandle(Tensor<T> result)
        {
            _result = result;
        }

        public bool IsCompleted { get { return true; } }

        public void Wait()
        {
            // No-op for already completed operation
        }

        public bool TryWait(int timeoutMs)
        {
            return true; // Always completed
        }

        public Tensor<T> GetResult<T>()
        {
            return (Tensor<T>)_result;
        }
    }

    /// <summary>
    /// Pending operation handle for async operations
    /// </summary>
    public class PendingOperationHandle : ICommunicationHandle
    {
        private readonly Task<Tensor<T>> _task;

        public PendingOperationHandle(Task<Tensor<T>> task)
        {
            _task = task ?? throw new ArgumentNullException(nameof(task));
        }

        public bool IsCompleted { get { return _task.IsCompleted; } }

        public void Wait()
        {
            _task.Wait();
        }

        public bool TryWait(int timeoutMs)
        {
            return _task.Wait(timeoutMs);
        }

        public Tensor<T> GetResult<T>()
        {
            if (!IsCompleted)
            {
                throw new InvalidOperationException("Operation has not completed yet");
            }

            return (Tensor<T>)(object)_task.Result;
        }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/Operations/Broadcast.cs`
   - `src/MLFramework/Communication/Operations/Reduce.cs`
   - `src/MLFramework/Communication/Operations/AllReduce.cs`
   - `src/MLFramework/Communication/Operations/Async/BroadcastAsync.cs`
   - `src/MLFramework/Communication/Operations/Async/AllReduceAsync.cs`
   - `src/MLFramework/Communication/Operations/CompletedOperationHandle.cs`
   - `src/MLFramework/Communication/Operations/PendingOperationHandle.cs`

2. **Design Decisions:**
   - Separate sync and async operations into different namespaces
   - Provide helper methods for common patterns (e.g., AverageGradients)
   - Validate tensor shapes and types before operations
   - Support multiple tensor operations for batch processing

3. **Error Handling:**
   - Throw ArgumentNullException for null arguments
   - Throw ArgumentOutOfRangeException for invalid ranks
   - Throw ArgumentException for invalid group configurations
   - Throw InvalidOperationException for accessing results before completion

4. **Performance Considerations:**
   - Minimize allocations in hot paths
   - Use efficient reduction algorithms (ring, tree)
   - Consider in-place operations to reduce memory usage

## Testing Requirements
- Unit tests for each operation with mock backends
- Tests for error conditions (invalid ranks, null tensors)
- Tests for async operations with timeouts
- Integration tests with real tensors
- Performance tests for different tensor sizes

## Success Criteria
- All operations compile and pass unit tests
- Async operations properly track completion state
- Error handling covers all edge cases
- Helper methods provide useful abstractions
