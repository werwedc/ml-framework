# Spec: Pipeline Communication Interface

## Overview
Define and implement the communication interface for sending tensors between pipeline stages. This provides the foundation for asynchronous communication.

## Class Design

### IPipelineCommunicator
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Interface for communication between pipeline stages
    /// </summary>
    public interface IPipelineCommunicator : IDisposable
    {
        /// <summary>
        /// Rank of the current process/device
        /// </summary>
        int CurrentRank { get; }

        /// <summary>
        /// Total number of processes/devices
        /// </summary>
        int WorldSize { get; }

        /// <summary>
        /// Send forward activation to next stage asynchronously
        /// </summary>
        Task<Tensor> SendForwardAsync(Tensor tensor, int destinationRank);

        /// <summary>
        /// Receive forward activation from previous stage asynchronously
        /// </summary>
        Task<Tensor> ReceiveForwardAsync(int sourceRank);

        /// <summary>
        /// Send backward gradient to previous stage asynchronously
        /// </summary>
        Task<Tensor> SendBackwardAsync(Tensor tensor, int destinationRank);

        /// <summary>
        /// Receive backward gradient from next stage asynchronously
        /// </summary>
        Task<Tensor> ReceiveBackwardAsync(int sourceRank);

        /// <summary>
        /// Synchronize all processes (barrier)
        /// </summary>
        Task BarrierAsync();

        /// <summary>
        /// Send a tensor to a specific rank
        /// </summary>
        Task SendAsync(Tensor tensor, int destinationRank);

        /// <summary>
        /// Receive a tensor from a specific rank
        /// </summary>
        Task<Tensor> ReceiveAsync(int sourceRank);
    }
}
```

### LocalPipelineCommunicator
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// In-memory communicator for single-device testing (no actual network)
    /// Uses shared memory buffers for communication
    /// </summary>
    public class LocalPipelineCommunicator : IPipelineCommunicator
    {
        private readonly int _rank;
        private readonly int _worldSize;
        private readonly Dictionary<(int from, int to), BlockingCollection<Tensor?>> _buffers;
        private readonly SemaphoreSlim _barrier;

        public int CurrentRank => _rank;
        public int WorldSize => _worldSize;

        public LocalPipelineCommunicator(int rank, int worldSize);

        public Task<Tensor> SendForwardAsync(Tensor tensor, int destinationRank);
        public Task<Tensor> ReceiveForwardAsync(int sourceRank);
        public Task<Tensor> SendBackwardAsync(Tensor tensor, int destinationRank);
        public Task<Tensor> ReceiveBackwardAsync(int sourceRank);
        public Task BarrierAsync();
        public Task SendAsync(Tensor tensor, int destinationRank);
        public Task<Tensor> ReceiveAsync(int sourceRank);
        public void Dispose();
    }
}
```

### CommunicationDirection
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Direction of communication in pipeline
    /// </summary>
    public enum CommunicationDirection
    {
        /// <summary>
        /// Forward pass: sending activations to next stage
        /// </summary>
        Forward,

        /// <summary>
        /// Backward pass: sending gradients to previous stage
        /// </summary>
        Backward
    }
}
```

## Implementation Requirements

### LocalPipelineCommunicator
1. **Initialization**
   - Create buffers for all possible communication pairs
   - Each buffer is a `BlockingCollection<Tensor?>` for thread-safe communication
   - Use `null` sentinel to signal end of communication

2. **Send Operations**
   - Copy tensor to destination buffer (or move ownership if not needed)
   - Return immediately (non-blocking)
   - Return the same tensor for chaining purposes

3. **Receive Operations**
   - Wait for tensor to be available in source buffer
   - Return the received tensor
   - Time out after 30 seconds and throw `TimeoutException`

4. **Barrier Implementation**
   - Use `SemaphoreSlim` with initial count = worldSize - 1
   - Each process waits until all have arrived at barrier
   - Release all waiting processes simultaneously

### Error Handling
1. Throw `ArgumentOutOfRangeException` if source/destination rank is invalid
2. Throw `InvalidOperationException` if disposed
3. Throw `TimeoutException` on receive timeout
4. Propagate any communication errors to caller

## Testing Requirements

1. **Unit Tests**
   - Test send and receive between two ranks
   - Test multiple sequential sends and receives
   - Test concurrent sends and receives
   - Test barrier synchronization
   - Test proper disposal and cleanup
   - Test timeout behavior

2. **Integration Tests**
   - Test forward communication through multiple stages (3+ stages)
   - Test backward communication through multiple stages
   - Test interleaved forward and backward communication
   - Test with actual tensor data (verify data integrity)

3. **Edge Cases**
   - Test sending to self (should work or throw based on design)
   - Test sending after dispose (should throw)
   - Test with very large tensors

## Files to Create
- `src/Pipeline/IPipelineCommunicator.cs`
- `src/Pipeline/LocalPipelineCommunicator.cs`
- `src/Pipeline/CommunicationDirection.cs`
- `tests/Pipeline/PipelineCommunicatorTests.cs`

## Dependencies
- Existing `Tensor` class
- .NET `System.Threading.Channels` or `BlockingCollection` for buffers
- No new external dependencies

## Notes for Future
- Future implementations: `NCCLCommunicator`, `MPICommunicator`, `GrpcCommunicator`
- This is a test/mock implementation for single-device development
- For production, implement NCCL-based communicator for multi-GPU

## Time Estimate
30-45 minutes for implementation and tests
