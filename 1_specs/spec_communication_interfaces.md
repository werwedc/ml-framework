# Spec: Communication Core Interfaces and Abstractions

## Overview
Define the foundational interfaces and abstractions for distributed communication primitives in the ML framework.

## Dependencies
- None (this is a foundational spec)

## Technical Requirements

### 1. ICommunicationBackend Interface
Create the base interface that all communication backends must implement.

**Interface Definition:**
```csharp
namespace MLFramework.Communication
{
    /// <summary>
    /// Reduce operations for collective communication
    /// </summary>
    public enum ReduceOp
    {
        Sum,
        Product,
        Max,
        Min,
        Avg
    }

    /// <summary>
    /// Communication device type
    /// </summary>
    public enum DeviceType
    {
        CPU,
        CUDA,
        ROCm
    }

    /// <summary>
    /// Base interface for all communication backends
    /// </summary>
    public interface ICommunicationBackend : IDisposable
    {
        /// <summary>
        /// Gets the rank of this process in the communication group
        /// </summary>
        int Rank { get; }

        /// <summary>
        /// Gets the total number of processes in the communication group
        /// </summary>
        int WorldSize { get; }

        /// <summary>
        /// Gets the backend name for logging/debugging
        /// </summary>
        string BackendName { get; }

        /// <summary>
        /// Broadcast tensor data from root rank to all ranks
        /// </summary>
        void Broadcast<T>(Tensor<T> tensor, int rootRank);

        /// <summary>
        /// Reduce tensor data from all ranks to root rank
        /// </summary>
        Tensor<T> Reduce<T>(Tensor<T> tensor, ReduceOp operation, int rootRank);

        /// <summary>
        /// AllReduce: combine data from all ranks and distribute to all
        /// </summary>
        Tensor<T> AllReduce<T>(Tensor<T> tensor, ReduceOp operation);

        /// <summary>
        /// AllGather: combine data from all ranks and distribute full dataset to all
        /// </summary>
        Tensor<T> AllGather<T>(Tensor<T> tensor);

        /// <summary>
        /// ReduceScatter: combine data from all ranks and scatter chunks
        /// </summary>
        Tensor<T> ReduceScatter<T>(Tensor<T> tensor, ReduceOp operation);

        /// <summary>
        /// Barrier: synchronize all ranks
        /// </summary>
        void Barrier();
    }
}
```

### 2. IAsyncCommunicationBackend Interface
Extend the base interface with async capabilities.

**Interface Definition:**
```csharp
namespace MLFramework.Communication
{
    /// <summary>
    /// Represents a handle to an ongoing communication operation
    /// </summary>
    public interface ICommunicationHandle
    {
        /// <summary>
        /// Returns true if the operation has completed
        /// </summary>
        bool IsCompleted { get; }

        /// <summary>
        /// Wait for the operation to complete
        /// </summary>
        void Wait();

        /// <summary>
        /// Wait for the operation to complete with timeout
        /// </summary>
        bool TryWait(int timeoutMs);

        /// <summary>
        /// Get the result tensor (only valid after completion)
        /// </summary>
        Tensor<T> GetResult<T>();
    }

    /// <summary>
    /// Interface for asynchronous communication operations
    /// </summary>
    public interface IAsyncCommunicationBackend : ICommunicationBackend
    {
        /// <summary>
        /// Non-blocking broadcast operation
        /// </summary>
        ICommunicationHandle BroadcastAsync<T>(Tensor<T> tensor, int rootRank);

        /// <summary>
        /// Non-blocking all-reduce operation
        /// </summary>
        ICommunicationHandle AllReduceAsync<T>(Tensor<T> tensor, ReduceOp operation);

        /// <summary>
        /// Non-blocking barrier operation
        /// </summary>
        ICommunicationHandle BarrierAsync();
    }
}
```

### 3. CommunicationException Class
Create custom exception types for communication errors.

```csharp
namespace MLFramework.Communication
{
    /// <summary>
    /// Base exception for all communication errors
    /// </summary>
    public class CommunicationException : Exception
    {
        public int? Rank { get; }
        public string? BackendName { get; }

        public CommunicationException(string message)
            : base(message) { }

        public CommunicationException(string message, int rank, string backendName)
            : base(message)
        {
            Rank = rank;
            BackendName = backendName;
        }

        public CommunicationException(string message, Exception innerException)
            : base(message, innerException) { }
    }

    /// <summary>
    /// Thrown when a communication operation times out
    /// </summary>
    public class CommunicationTimeoutException : CommunicationException
    {
        public TimeSpan TimeoutDuration { get; }

        public CommunicationTimeoutException(string message, TimeSpan timeout)
            : base(message)
        {
            TimeoutDuration = timeout;
        }
    }

    /// <summary>
    /// Thrown when ranks have inconsistent states (e.g., mismatched tensor shapes)
    /// </summary>
    public class RankMismatchException : CommunicationException
    {
        public int ExpectedRank { get; }
        public int ActualRank { get; }

        public RankMismatchException(string message, int expected, int actual)
            : base(message)
        {
            ExpectedRank = expected;
            ActualRank = actual;
        }
    }
}
```

### 4. Tensor Integration Extensions
Create extension methods to integrate communication with the Tensor class.

```csharp
namespace MLFramework.Communication
{
    /// <summary>
    /// Extension methods for Tensor communication
    /// </summary>
    public static class TensorCommunicationExtensions
    {
        /// <summary>
        /// Broadcast this tensor from root rank to all ranks
        /// </summary>
        public static void BroadcastToAll<T>(this Tensor<T> tensor, ICommunicationBackend backend, int rootRank)
        {
            backend.Broadcast(tensor, rootRank);
        }

        /// <summary>
        /// Perform all-reduce on this tensor in place
        /// </summary>
        public static Tensor<T> AllReduceInPlace<T>(this Tensor<T> tensor, ICommunicationBackend backend, ReduceOp operation)
        {
            return backend.AllReduce(tensor, operation);
        }

        /// <summary>
        /// Scatter tensor across ranks (each rank gets a chunk)
        /// </summary>
        public static Tensor<T> Scatter<T>(this Tensor<T> tensor, ICommunicationBackend backend, int rank)
        {
            // Implementation using AllGather + indexing
            var gathered = backend.AllGather(tensor);
            return SliceForRank(gathered, rank, backend.WorldSize);
        }

        private static Tensor<T> SliceForRank<T>(Tensor<T> tensor, int rank, int worldSize)
        {
            // Calculate chunk size and return appropriate slice
            // This is a placeholder - actual implementation depends on tensor layout
            throw new NotImplementedException();
        }
    }
}
```

### 5. Backend Factory Interface
Define interface for creating communication backends.

```csharp
namespace MLFramework.Communication
{
    /// <summary>
    /// Configuration for communication backend initialization
    /// </summary>
    public class CommunicationConfig
    {
        /// <summary>
        /// Timeout for operations in milliseconds (default: 5 minutes)
        /// </summary>
        public int TimeoutMs { get; set; } = 300000;

        /// <summary>
        /// Enable performance logging
        /// </summary>
        public bool EnableLogging { get; set; } = false;

        /// <summary>
        /// Use pinned memory for transfers
        /// </summary>
        public bool UsePinnedMemory { get; set; } = true;
    }

    /// <summary>
    /// Interface for creating communication backends
    /// </summary>
    public interface ICommunicationBackendFactory
    {
        /// <summary>
        /// Detect if this backend is available on the current system
        /// </summary>
        bool IsAvailable();

        /// <summary>
        /// Create a backend instance with the given configuration
        /// </summary>
        ICommunicationBackend Create(CommunicationConfig config);

        /// <summary>
        /// Get the priority of this backend (higher = preferred)
        /// </summary>
        int Priority { get; }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/ReduceOp.cs`
   - `src/MLFramework/Communication/DeviceType.cs`
   - `src/MLFramework/Communication/ICommunicationBackend.cs`
   - `src/MLFramework/Communication/IAsyncCommunicationBackend.cs`
   - `src/MLFramework/Communication/ICommunicationHandle.cs`
   - `src/MLFramework/Communication/CommunicationException.cs`
   - `src/MLFramework/Communication/CommunicationConfig.cs`
   - `src/MLFramework/Communication/ICommunicationBackendFactory.cs`
   - `src/MLFramework/Communication/TensorCommunicationExtensions.cs`

2. **Tensor Integration:**
   - Assume `Tensor<T>` class exists in the framework
   - The extensions provide a fluent API for common operations
   - In-place operations should modify the original tensor

3. **Error Handling:**
   - All methods should throw appropriate exceptions
   - Include rank and backend information in exceptions for debugging
   - Implement IDisposable for resource cleanup

4. **Performance Considerations:**
   - Use generic type T to support float, double, int, etc.
   - Design for zero-copy operations where possible
   - Minimize allocations in hot paths

## Testing Requirements
- Unit tests for all interfaces (mock implementations)
- Exception handling tests
- Extension method tests with mock backends

## Success Criteria
- All interfaces compile without errors
- Interfaces are well-documented with XML comments
- Extension methods provide intuitive API
- Exception hierarchy covers all error scenarios
