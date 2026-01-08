# Spec: Asynchronous Communication Primitives

## Overview
Implement asynchronous communication primitives to enable overlapping computation with communication for improved performance.

## Dependencies
- `spec_communication_interfaces.md`
- `spec_collective_basic.md`
- `spec_collective_advanced.md`
- `spec_point_to_point.md`

## Technical Requirements

### 1. Async Communication Handle
Enhanced handle for tracking asynchronous operations.

```csharp
namespace MLFramework.Communication.Async
{
    /// <summary>
    /// Enhanced communication handle for async operations
    /// </summary>
    public class AsyncCommunicationHandle : ICommunicationHandle
    {
        private readonly Task<Tensor<T>> _task;
        private readonly CancellationTokenSource _cts;
        private bool _completed;
        private Tensor<T>? _result;

        public AsyncCommunicationHandle(Task<Tensor<T>> task, CancellationTokenSource? cts = null)
        {
            _task = task ?? throw new ArgumentNullException(nameof(task));
            _cts = cts ?? new CancellationTokenSource();
        }

        public bool IsCompleted
        {
            get
            {
                if (_completed) return true;
                _completed = _task.IsCompleted;
                return _completed;
            }
        }

        public void Wait()
        {
            _task.Wait();
            _result = _task.Result;
        }

        public bool TryWait(int timeoutMs)
        {
            if (_task.Wait(timeoutMs))
            {
                _result = _task.Result;
                return true;
            }
            return false;
        }

        public Tensor<T> GetResult<T>()
        {
            if (!_completed)
            {
                throw new InvalidOperationException("Operation has not completed yet");
            }

            if (_result == null)
            {
                throw new InvalidOperationException("Result is not available");
            }

            return (Tensor<T>)(object)_result;
        }

        /// <summary>
        /// Get result as Task for async/await pattern
        /// </summary>
        public Task<Tensor<T>> AsTask()
        {
            return _task;
        }

        /// <summary>
        /// Cancel the operation if possible
        /// </summary>
        public void Cancel()
        {
            _cts.Cancel();
        }

        /// <summary>
        /// Check if the operation was cancelled
        /// </summary>
        public bool IsCancelled
        {
            get { return _cts.IsCancellationRequested; }
        }
    }
}
```

### 2. Operation Queue
Queue multiple async operations and wait for all.

```csharp
namespace MLFramework.Communication.Async
{
    /// <summary>
    /// Queue for managing multiple async communication operations
    /// </summary>
    public class CommunicationOperationQueue : IDisposable
    {
        private readonly List<ICommunicationHandle> _operations;
        private readonly object _lock;
        private bool _disposed;

        public int PendingOperationsCount
        {
            get
            {
                lock (_lock)
                {
                    return _operations.Count(o => !o.IsCompleted);
                }
            }
        }

        public CommunicationOperationQueue()
        {
            _operations = new List<ICommunicationHandle>();
            _lock = new object();
        }

        /// <summary>
        /// Add an operation to the queue
        /// </summary>
        public void Enqueue(ICommunicationHandle handle)
        {
            if (handle == null)
                throw new ArgumentNullException(nameof(handle));

            lock (_lock)
            {
                _operations.Add(handle);
            }
        }

        /// <summary>
        /// Wait for all operations to complete
        /// </summary>
        public void WaitForAll()
        {
            lock (_lock)
            {
                foreach (var operation in _operations)
                {
                    operation.Wait();
                }
            }
        }

        /// <summary>
        /// Wait for all operations with timeout
        /// </summary>
        /// <returns>True if all completed, false if timeout</returns>
        public bool TryWaitForAll(int timeoutMs)
        {
            var stopwatch = Stopwatch.StartNew();
            lock (_lock)
            {
                foreach (var operation in _operations)
                {
                    var remaining = (int)(timeoutMs - stopwatch.ElapsedMilliseconds);
                    if (remaining <= 0)
                    {
                        return false;
                    }

                    if (!operation.TryWait(remaining))
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        /// <summary>
        /// Wait for any operation to complete
        /// </summary>
        /// <returns>Index of completed operation or -1 if timeout</returns>
        public int WaitForAny(int timeoutMs = -1)
        {
            lock (_lock)
            {
                for (int i = 0; i < _operations.Count; i++)
                {
                    if (_operations[i].IsCompleted)
                    {
                        return i;
                    }
                }

                // Poll for completion
                var stopwatch = Stopwatch.StartNew();
                while (timeoutMs == -1 || stopwatch.ElapsedMilliseconds < timeoutMs)
                {
                    for (int i = 0; i < _operations.Count; i++)
                    {
                        if (_operations[i].IsCompleted)
                        {
                            return i;
                        }
                    }
                    Thread.Sleep(1);
                }

                return -1;
            }
        }

        /// <summary>
        /// Clear all completed operations
        /// </summary>
        public void ClearCompleted()
        {
            lock (_lock)
            {
                _operations.RemoveAll(o => o.IsCompleted);
            }
        }

        /// <summary>
        /// Cancel all pending operations
        /// </summary>
        public void CancelAll()
        {
            lock (_lock)
            {
                foreach (var operation in _operations)
                {
                    if (operation is AsyncCommunicationHandle asyncHandle)
                    {
                        asyncHandle.Cancel();
                    }
                }
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                CancelAll();
                lock (_lock)
                {
                    _operations.Clear();
                }
                _disposed = true;
            }
        }
    }
}
```

### 3. Async Operation Wrapper
Wrapper for executing async operations with error handling.

```csharp
namespace MLFramework.Communication.Async
{
    /// <summary>
    /// Wrapper for async operations with error handling
    /// </summary>
    public static class AsyncOperationWrapper
    {
        /// <summary>
        /// Execute async operation with timeout and error handling
        /// </summary>
        public static async Task<Tensor<T>> ExecuteAsync<T>(
            Func<Task<Tensor<T>>> operation,
            int timeoutMs,
            CancellationToken cancellationToken = default)
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            cts.CancelAfter(timeoutMs);

            try
            {
                return await operation().WithCancellation(cts.Token);
            }
            catch (OperationCanceledException) when (cts.Token.IsCancellationRequested)
            {
                throw new CommunicationTimeoutException($"Operation timed out after {timeoutMs}ms", TimeSpan.FromMilliseconds(timeoutMs));
            }
            catch (Exception ex)
            {
                throw new CommunicationException("Async operation failed", ex);
            }
        }

        /// <summary>
        /// Execute multiple async operations in parallel
        /// </summary>
        public static async Task<List<Tensor<T>>> ExecuteAllAsync<T>(
            IEnumerable<Func<Task<Tensor<T>>>> operations,
            int timeoutMs = -1,
            CancellationToken cancellationToken = default)
        {
            var tasks = operations.Select(op => op()).ToList();

            if (timeoutMs > 0)
            {
                using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                cts.CancelAfter(timeoutMs);

                try
                {
                    await Task.WhenAll(tasks).WithCancellation(cts.Token);
                }
                catch (OperationCanceledException) when (cts.Token.IsCancellationRequested)
                {
                    throw new CommunicationTimeoutException($"One or more operations timed out after {timeoutMs}ms", TimeSpan.FromMilliseconds(timeoutMs));
                }
            }
            else
            {
                await Task.WhenAll(tasks);
            }

            return tasks.Select(t => t.Result).ToList();
        }
    }

    /// <summary>
    /// Extension method for Task with cancellation support
    /// </summary>
    public static class TaskExtensions
    {
        public static async Task<T> WithCancellation<T>(this Task<T> task, CancellationToken cancellationToken)
        {
            var tcs = new TaskCompletionSource<bool>();
            using (cancellationToken.Register(s => ((TaskCompletionSource<bool>)s!).TrySetResult(true), tcs))
            {
                if (task != await Task.WhenAny(task, tcs.Task))
                {
                    throw new OperationCanceledException(cancellationToken);
                }
            }

            return await task;
        }
    }
}
```

### 4. Compute-Communication Overlap Helper
Helper methods for overlapping computation with communication.

```csharp
namespace MLFramework.Communication.Async
{
    /// <summary>
    /// Helper for overlapping computation with communication
    /// </summary>
    public static class ComputeCommunicationOverlap
    {
        /// <summary>
        /// Start async communication and immediately return handle
        /// </summary>
        public static ICommunicationHandle StartAllReduce<T>(
            IAsyncCommunicationBackend backend,
            Tensor<T> tensor,
            ReduceOp operation)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var task = Task.Run(() => backend.AllReduce(tensor, operation));
            return new AsyncCommunicationHandle(task);
        }

        /// <summary>
        /// Pattern: Do computation while communication is in progress
        /// </summary>
        public static Tensor<T> ComputeWhileCommunicating<T>(
            IAsyncCommunicationBackend backend,
            Tensor<T> tensorToSync,
            Func<Tensor<T>> computeFunc,
            ReduceOp operation = ReduceOp.Sum)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensorToSync == null)
                throw new ArgumentNullException(nameof(tensorToSync));

            if (computeFunc == null)
                throw new ArgumentNullException(nameof(computeFunc));

            // Start async communication
            var commHandle = StartAllReduce(backend, tensorToSync, operation);

            // Do computation while communicating
            var computed = computeFunc();

            // Wait for communication to finish
            commHandle.Wait();

            return commHandle.GetResult<T>();
        }

        /// <summary>
        /// Pattern: Pipeline multiple compute-communication stages
        /// </summary>
        public static List<Tensor<T>> PipelineComputeCommunicate<T>(
            IAsyncCommunicationBackend backend,
            List<Tensor<T>> tensors,
            Func<Tensor<T>, Tensor<T>> computeFunc,
            ReduceOp operation = ReduceOp.Sum)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensors == null)
                throw new ArgumentNullException(nameof(tensors));

            if (computeFunc == null)
                throw new ArgumentNullException(nameof(computeFunc));

            var results = new List<Tensor<T>>();
            var queue = new CommunicationOperationQueue();

            // Stage 1: Start all communications
            foreach (var tensor in tensors)
            {
                var handle = StartAllReduce(backend, tensor, operation);
                queue.Enqueue(handle);
            }

            // Stage 2: Compute while waiting
            foreach (var tensor in tensors)
            {
                // Do computation
                var computed = computeFunc(tensor);
                results.Add(computed);
            }

            // Stage 3: Wait for all communications
            queue.WaitForAll();

            // Get results
            return queue.GetResults<T>();
        }
    }

    /// <summary>
    /// Extension methods for CommunicationOperationQueue
    /// </summary>
    public static class CommunicationOperationQueueExtensions
    {
        public static List<Tensor<T>> GetResults<T>(this CommunicationOperationQueue queue)
        {
            // This would need access to the internal operations list
            // For now, this is a placeholder
            throw new NotImplementedException("GetResults needs access to queue internals");
        }
    }
}
```

### 5. Async Event Handler
Event-based async communication pattern.

```csharp
namespace MLFramework.Communication.Async
{
    /// <summary>
    /// Event-based async communication handler
    /// </summary>
    public class AsyncCommunicationEventHandler : IDisposable
    {
        private readonly IAsyncCommunicationBackend _backend;
        private readonly Dictionary<int, List<Action<ICommunicationHandle>>> _eventHandlers;
        private readonly object _lock;
        private bool _disposed;

        /// <summary>
        /// Event fired when any operation completes
        /// </summary>
        public event Action<ICommunicationHandle>? OnOperationComplete;

        /// <summary>
        /// Event fired when an operation fails
        /// </summary>
        public event Action<ICommunicationHandle, Exception>? OnOperationError;

        public AsyncCommunicationEventHandler(IAsyncCommunicationBackend backend)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _eventHandlers = new Dictionary<int, List<Action<ICommunicationHandle>>>();
            _lock = new object();
        }

        /// <summary>
        /// Start async operation and register completion handler
        /// </summary>
        public void StartOperation(
            Func<ICommunicationHandle> startFunc,
            int operationId,
            Action<ICommunicationHandle>? onComplete = null)
        {
            var handle = startFunc();

            lock (_lock)
            {
                if (!_eventHandlers.ContainsKey(operationId))
                {
                    _eventHandlers[operationId] = new List<Action<ICommunicationHandle>>();
                }

                if (onComplete != null)
                {
                    _eventHandlers[operationId].Add(onComplete);
                }
            }

            // Start monitoring task
            Task.Run(() => MonitorOperation(handle, operationId));
        }

        private async Task MonitorOperation(ICommunicationHandle handle, int operationId)
        {
            try
            {
                await Task.Run(() => handle.Wait());

                // Fire completion events
                OnOperationComplete?.Invoke(handle);

                lock (_lock)
                {
                    if (_eventHandlers.ContainsKey(operationId))
                    {
                        foreach (var handler in _eventHandlers[operationId])
                        {
                            handler(handle);
                        }
                        _eventHandlers.Remove(operationId);
                    }
                }
            }
            catch (Exception ex)
            {
                OnOperationError?.Invoke(handle, ex);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                lock (_lock)
                {
                    _eventHandlers.Clear();
                }
                _disposed = true;
            }
        }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/Async/AsyncCommunicationHandle.cs`
   - `src/MLFramework/Communication/Async/CommunicationOperationQueue.cs`
   - `src/MLFramework/Communication/Async/AsyncOperationWrapper.cs`
   - `src/MLFramework/Communication/Async/TaskExtensions.cs`
   - `src/MLFramework/Communication/Async/ComputeCommunicationOverlap.cs`
   - `src/MLFramework/Communication/Async/AsyncCommunicationEventHandler.cs`

2. **Design Decisions:**
   - Async handles use Task-based implementation
   - Operation queue manages multiple async operations
   - Helpers provide common patterns for compute-communication overlap
   - Event-based pattern for reactive programming

3. **Error Handling:**
   - Timeout support for all async operations
   - Cancellation support via CancellationToken
   - Event handlers for error notification
   - Proper cleanup in Dispose

4. **Performance Considerations:**
   - Minimize allocations in hot paths
   - Use efficient polling for completion checks
   - Enable compute-communication overlap for better utilization

## Testing Requirements
- Unit tests for async communication handles
- Tests for operation queue with multiple operations
- Tests for compute-communication overlap patterns
- Tests for timeout and cancellation
- Performance tests comparing sync vs async

## Success Criteria
- All async primitives compile and pass tests
- Operation queue correctly manages multiple operations
- Compute-communication overlap improves performance
- Timeout and cancellation work correctly
- Event-based pattern fires callbacks appropriately
