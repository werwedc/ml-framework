# Spec: Pipeline Async Execution

## Overview
Implement asynchronous execution support for pipeline parallelism, including CUDA stream management and non-blocking communication to overlap computation and data transfer.

## Class Design

### AsyncPipelineExecutor
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Manages asynchronous execution for pipeline stages
    /// Overlaps computation and communication to maximize throughput
    /// </summary>
    public class AsyncPipelineExecutor : IDisposable
    {
        private readonly PipelineStage _stage;
        private readonly IPipelineCommunicator _communicator;
        private readonly Dictionary<int, CudaStream> _computeStreams;
        private readonly Dictionary<int, CudaStream> _commStreams;
        private readonly int _numStreams;

        /// <summary>
        /// Number of CUDA streams for compute
        /// </summary>
        public int NumComputeStreams => _computeStreams.Count;

        /// <summary>
        /// Number of CUDA streams for communication
        /// </summary>
        public int NumCommStreams => _commStreams.Count;

        public AsyncPipelineExecutor(
            PipelineStage stage,
            IPipelineCommunicator communicator,
            int numStreams = 2);

        /// <summary>
        /// Execute forward pass asynchronously on a specific stream
        /// </summary>
        /// <param name="input">Input tensor</param>
        /// <param name="streamIndex">Stream index</param>
        /// <returns>Task that completes when forward pass is done</returns>
        public Task<Tensor> ForwardAsync(Tensor input, int streamIndex = 0);

        /// <summary>
        /// Execute backward pass asynchronously on a specific stream
        /// </summary>
        /// <param name="gradOutput">Gradient tensor</param>
        /// <param name="streamIndex">Stream index</param>
        /// <returns>Task that completes when backward pass is done</returns>
        public Task<Tensor> BackwardAsync(Tensor gradOutput, int streamIndex = 0);

        /// <summary>
        /// Send tensor asynchronously on communication stream
        /// </summary>
        public Task<Tensor> SendAsync(Tensor tensor, int destinationRank, int streamIndex = 0);

        /// <summary>
        /// Receive tensor asynchronously on communication stream
        /// </summary>
        public Task<Tensor> ReceiveAsync(int sourceRank, int streamIndex = 0);

        /// <summary>
        /// Synchronize all compute streams
        /// </summary>
        public Task SyncComputeAsync();

        /// <summary>
        /// Synchronize all communication streams
        /// </summary>
        public Task SyncCommAsync();

        /// <summary>
        /// Synchronize all streams
        /// </summary>
        public Task SyncAllAsync();

        /// <summary>
        /// Get stream for a specific micro-batch
        /// </summary>
        public CudaStream GetComputeStream(int microBatchIndex);

        /// <summary>
        /// Get communication stream for a specific micro-batch
        /// </summary>
        public CudaStream GetCommStream(int microBatchIndex);

        public void Dispose();
    }
}
```

### StreamManager
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Manages CUDA streams for pipeline execution
    /// </summary>
    public class StreamManager : IDisposable
    {
        private readonly List<CudaStream> _streams;
        private readonly Device _device;
        private int _currentStreamIndex;

        /// <summary>
        /// Number of managed streams
        /// </summary>
        public int Count => _streams.Count;

        public StreamManager(Device device, int numStreams);

        /// <summary>
        /// Get a stream (round-robin)
        /// </summary>
        public CudaStream GetStream();

        /// <summary>
        /// Get a specific stream by index
        /// </summary>
        public CudaStream GetStream(int index);

        /// <summary>
        /// Synchronize all streams
        /// </summary>
        public Task SynchronizeAllAsync();

        /// <summary>
        /// Synchronize a specific stream
        /// </summary>
        public Task SynchronizeAsync(int index);

        /// <summary>
        /// Record an event on a stream
        /// </summary>
        public CudaEvent RecordEvent(int streamIndex);

        /// <summary>
        /// Wait for an event on a stream
        /// </summary>
        public void WaitForEvent(CudaEvent evt, int streamIndex);

        public void Dispose();
    }
}
```

### AsyncOperation
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Represents an asynchronous pipeline operation
    /// </summary>
    public class AsyncOperation
    {
        /// <summary>
        /// Unique ID for this operation
        /// </summary>
        public Guid Id { get; }

        /// <summary>
        /// Type of operation
        /// </summary>
        public OperationType Type { get; }

        /// <summary>
        /// Micro-batch index
        /// </summary>
        public int MicroBatchIndex { get; }

        /// <summary>
        /// Stream index
        /// </summary>
        public int StreamIndex { get; }

        /// <summary>
        /// Task representing the operation
        /// </summary>
        public Task Task { get; }

        /// <summary>
        /// Whether the operation is completed
        /// </summary>
        public bool IsCompleted => Task.IsCompleted;

        public AsyncOperation(
            Guid id,
            OperationType type,
            int microBatchIndex,
            int streamIndex,
            Task task);
    }

    public enum OperationType
    {
        Forward,
        Backward,
        SendForward,
        ReceiveForward,
        SendBackward,
        ReceiveBackward
    }
}
```

## Implementation Requirements

### Stream Management
1. Create separate streams for compute and communication
2. Default: 2 compute streams, 2 communication streams
3. Round-robin assignment for micro-batches
4. Support stream synchronization with events

### ForwardAsync
1. Get or create compute stream for micro-batch
2. Set current stream on device
3. Execute forward pass: `_stage.Forward(input)`
4. Return task that completes when forward pass is done
5. Handle exceptions and propagate to caller

### BackwardAsync
1. Get or create compute stream for micro-batch
2. Set current stream on device
3. Execute backward pass using autograd
4. Return task that completes when backward pass is done
5. Accumulate gradients if needed

### SendAsync/ReceiveAsync
1. Get or create communication stream for micro-batch
2. Execute send/receive on that stream
3. Use communicator async methods
4. Return task that completes when communication is done

### Stream Synchronization
1. Use `CudaEvent` to record completion on streams
2. Wait for events on other streams to synchronize
3. Support individual stream sync and all-stream sync
4. Handle timeout scenarios

### Dependency Tracking
1. Ensure forward pass completes before sending activation
2. Ensure receive completes before forward pass
3. Ensure backward pass completes before sending gradient
4. Use Task continuations or events for dependencies

### Error Handling
1. Catch CUDA errors and convert to .NET exceptions
2. Propagate errors through Task system
3. Ensure proper cleanup on errors
4. Log stream-related errors

## Testing Requirements

1. **Unit Tests**
   - Test stream creation and management
   - Test forward async execution
   - Test backward async execution
   - Test send/receive async operations
   - Test stream synchronization
   - Test multiple concurrent operations
   - Test round-robin stream assignment
   - Test event recording and waiting

2. **Integration Tests**
   - Test overlapped compute and communication
   - Test full pipeline with async execution
   - Test that async results match sync execution
   - Test with actual neural network
   - Test performance improvement (optional)

3. **Edge Cases**
   - Test with single stream
   - Test with many streams
   - Test invalid stream index (should throw)
   - Test operations that fail and propagate exceptions
   - Test disposal with pending operations

## Files to Create
- `src/Pipeline/AsyncPipelineExecutor.cs`
- `src/Pipeline/StreamManager.cs`
- `src/Pipeline/AsyncOperation.cs`
- `tests/Pipeline/AsyncPipelineExecutorTests.cs`

## Dependencies
- `PipelineStage` from spec_pipeline_stage_core
- `IPipelineCommunicator` from spec_pipeline_communication
- Existing `CudaStream`, `CudaEvent`, `Device` classes
- .NET `Task` and `async/await`

## Time Estimate
45-60 minutes for implementation and tests

## Notes
- This is an advanced optimization feature
- Basic pipeline training can work without this
- Critical for production performance
- Ensure correct ordering and synchronization
- Consider adding benchmarks to measure improvement
