# Spec: Continuous Batch Scheduler Core

## Overview
Implement the core scheduler orchestrator that coordinates request queue, batch manager, capacity manager, and completion detection. The scheduler drives the iteration-level batching loop and manages the overall request lifecycle.

## Class: ContinuousBatchScheduler
```csharp
public class ContinuousBatchScheduler : IDisposable
{
    private readonly RequestQueue _requestQueue;
    private readonly BatchManager _batchManager;
    private readonly CompletionDetector _completionDetector;
    private readonly CapacityManager _capacityManager;
    private readonly IModelExecutor _modelExecutor;
    private readonly ISchedulerMetrics _metrics;
    private readonly SchedulerConfiguration _config;
    private readonly CancellationTokenSource _shutdownTokenSource;
    private Task? _schedulerTask;
    private int _iterationCount;

    public bool IsRunning { get; private set; }
    public int ActiveRequestCount => _batchManager.ActiveRequestCount;

    public ContinuousBatchScheduler(
        RequestQueue requestQueue,
        BatchManager batchManager,
        CompletionDetector completionDetector,
        CapacityManager capacityManager,
        IModelExecutor modelExecutor,
        ISchedulerMetrics metrics,
        SchedulerConfiguration config)
    {
        _requestQueue = requestQueue;
        _batchManager = batchManager;
        _completionDetector = completionDetector;
        _capacityManager = capacityManager;
        _modelExecutor = modelExecutor;
        _metrics = metrics;
        _config = config;
        _shutdownTokenSource = new CancellationTokenSource();
        _iterationCount = 0;
        IsRunning = false;
    }

    // Start the scheduler loop
    public void Start();

    // Stop the scheduler gracefully
    public async Task StopAsync(TimeSpan? timeout = null);

    // Enqueue a new request
    public Task<string> EnqueueAsync(Request request, Priority priority = Priority.Normal);

    // Execute single iteration (for testing)
    public Task<IterationResult> ExecuteIterationAsync(
        CancellationToken cancellationToken = default);

    public void Dispose();
}
```

---

## Class: SchedulerConfiguration
```csharp
public record class SchedulerConfiguration(
    int IterationTimeoutMs,              // Timeout per iteration
    int MaxIdleIterations,               // Max idle iterations before throttle
    int MinIterationsPerSecond,          // Target iteration rate
    bool EnableAdaptiveBatching,         // Adjust batch size based on load
    int WarmupIterations,                // Number of warmup iterations
    double TargetUtilization             // Target GPU utilization (0-1)
)
{
    public static readonly SchedulerConfiguration Default = new(
        IterationTimeoutMs: 1000,
        MaxIdleIterations: 10,
        MinIterationsPerSecond: 30,
        EnableAdaptiveBatching: true,
        WarmupIterations: 5,
        TargetUtilization: 0.85
    );
}
```

**Purpose**: Configure scheduler behavior.

---

## Class: IterationResult
```csharp
public record class IterationResult(
    int IterationNumber,
    int RequestCount,
    int TokensGenerated,
    int RequestsCompleted,
    TimeSpan ProcessingTime,
    long MemoryBytesUsed
);
```

**Purpose**: Result of a single iteration execution.

---

## Interface: IModelExecutor
```csharp
public interface IModelExecutor
{
    // Execute forward pass on a batch
    Task<BatchOutput> ExecuteBatchAsync(
        Batch batch,
        CancellationToken cancellationToken = default);
}
```

---

## Class: BatchOutput
```csharp
public record class BatchOutput(
    Dictionary<RequestId, int> GeneratedTokens,
    Dictionary<RequestId, float[]> Logits,
    bool[] IsEosReached
);
```

---

## Interface: ISchedulerMetrics
```csharp
public interface ISchedulerMetrics
{
    void RecordIteration(IterationResult result);
    void RecordRequestCompletion(RequestResult result);
    void RecordBatchUtilization(double utilization);
    void RecordError(string errorType, Exception exception);
}
```

---

## Implementation Details

### Start
```csharp
public void Start()
{
    if (IsRunning)
        throw new InvalidOperationException("Scheduler is already running");

    IsRunning = true;
    _schedulerTask = Task.Run(async () => await SchedulerLoopAsync());
}
```

**Requirements**:
- Prevent multiple starts
- Launch background task
- Set running flag

---

### SchedulerLoopAsync
```csharp
private async Task SchedulerLoopAsync()
{
    int idleIterationCount = 0;
    bool isWarmup = true;
    int warmupCount = 0;

    while (!_shutdownTokenSource.Token.IsCancellationRequested)
    {
        try
        {
            var iterationCts = CancellationTokenSource.CreateLinkedTokenSource(
                _shutdownTokenSource.Token
            );

            iterationCts.CancelAfter(_config.IterationTimeoutMs);

            // Execute iteration
            var result = await ExecuteIterationAsync(iterationCts.Token);

            // Record metrics
            _metrics.RecordIteration(result);

            // Handle warmup
            if (isWarmup)
            {
                warmupCount++;
                if (warmupCount >= _config.WarmupIterations)
                    isWarmup = false;
                continue;
            }

            // Track idle iterations
            if (result.RequestCount == 0)
            {
                idleIterationCount++;
                if (idleIterationCount >= _config.MaxIdleIterations)
                {
                    // Throttle: wait before next iteration
                    await Task.Delay(100, _shutdownTokenSource.Token);
                    idleIterationCount = 0;
                }
            }
            else
            {
                idleIterationCount = 0;
            }

            _iterationCount++;
        }
        catch (OperationCanceledException)
        {
            // Normal shutdown
            break;
        }
        catch (Exception ex)
        {
            _metrics.RecordError("IterationError", ex);
            await Task.Delay(100, _shutdownTokenSource.Token); // Backoff
        }
    }

    IsRunning = false;
}
```

**Requirements**:
- Continuous iteration loop
- Handle cancellation
- Manage idle states
- Record metrics
- Handle errors gracefully

---

### ExecuteIterationAsync
```csharp
public async Task<IterationResult> ExecuteIterationAsync(
    CancellationToken cancellationToken = default)
{
    var stopwatch = Stopwatch.StartNew();

    // Prepare batch for this iteration
    var batch = _batchManager.PrepareNextIteration();

    if (batch.Size == 0)
    {
        // No requests to process
        return new IterationResult(
            _iterationCount,
            0,
            0,
            0,
            stopwatch.Elapsed,
            0
        );
    }

    // Execute model forward pass
    var output = await _modelExecutor.ExecuteBatchAsync(batch, cancellationToken);

    // Process outputs
    int totalTokensGenerated = 0;
    int completedRequests = 0;
    List<RequestId> completedIds = new List<RequestId>();

    foreach (var request in batch.Requests)
    {
        if (output.GeneratedTokens.TryGetValue(request.Id, out int tokenId))
        {
            // Update request state
            request.GeneratedTokenIds.Add(tokenId);
            request.GeneratedTokens++;
            totalTokensGenerated++;

            // Check completion
            var (isCompleted, reason) = _completionDetector.CheckCompletion(request);
            if (isCompleted)
            {
                completedIds.Add(request.Id);
                completedRequests++;

                // Create result
                var result = new RequestResult(
                    request.Id,
                    DecodeTokens(request.GeneratedTokenIds),
                    request.GeneratedTokens,
                    reason,
                    DateTime.UtcNow - request.EnqueuedTime
                );

                // Complete request
                request.CompletionSource.TrySetResult(result.GeneratedText);
                _metrics.RecordRequestCompletion(result);
            }
            else
            {
                // Update capacity usage
                _capacityManager.UpdateUsage(request.Id, 1);
            }
        }
    }

    // Report completions to batch manager
    if (completedIds.Count > 0)
    {
        _batchManager.RemoveCompletedRequests(completedIds);
    }

    // Record batch utilization
    double utilization = (double)batch.Size / _config.MaxBatchSize;
    _metrics.RecordBatchUtilization(utilization);

    return new IterationResult(
        _iterationCount,
        batch.Size,
        totalTokensGenerated,
        completedRequests,
        stopwatch.Elapsed,
        batch.EstimatedMemoryBytes
    );
}
```

**Requirements**:
- Prepare batch
- Execute model
- Process outputs
- Update request state
- Check completion
- Record metrics
- Return detailed results

---

### EnqueueAsync
```csharp
public Task<string> EnqueueAsync(Request request, Priority priority = Priority.Normal)
{
    // Enqueue to request queue
    _requestQueue.Enqueue(request, priority);

    // Return completion task
    return request.CompletionSource.Task;
}
```

**Requirements**:
- Add to queue
- Return completion task
- Support priority

---

### StopAsync
```csharp
public async Task StopAsync(TimeSpan? timeout = null)
{
    if (!IsRunning)
        return;

    _shutdownTokenSource.Cancel();

    var stopTask = _schedulerTask ?? Task.CompletedTask;

    if (timeout.HasValue)
    {
        var timeoutTask = Task.Delay(timeout.Value);
        var completedTask = await Task.WhenAny(stopTask, timeoutTask);

        if (completedTask == timeoutTask)
        {
            // Timeout - force stop
            // Note: May leave some requests incomplete
        }
    }
    else
    {
        await stopTask;
    }

    IsRunning = false;
}
```

**Requirements**:
- Signal shutdown
- Wait for completion
- Support timeout
- Handle graceful shutdown

---

### Dispose
```csharp
public void Dispose()
{
    _shutdownTokenSource.Cancel();
    _shutdownTokenSource.Dispose();
    _schedulerTask?.Dispose();
}
```

**Requirements**:
- Clean up resources
- Cancel background task

---

## Files to Create

### Implementation
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/ContinuousBatchScheduler.cs`
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/SchedulerConfiguration.cs`
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/IterationResult.cs`

### Tests
- `tests/MLFramework.Tests/Inference/ContinuousBatching/ContinuousBatchSchedulerTests.cs`

---

## Dependencies
- `spec_request_queue.md` (RequestQueue)
- `spec_batch_manager.md` (BatchManager)
- `spec_completion_detector.md` (CompletionDetector)
- `spec_capacity_manager.md` (CapacityManager)
- `spec_continuous_batching_models.md` (Request, RequestResult, Batch)

---

## Testing Requirements

### Unit Tests (with Mocks)
1. **Basic Operations**:
   - Start scheduler successfully
   - Stop scheduler gracefully
   - Enqueue requests successfully

2. **Iteration Execution**:
   - Execute iteration with requests
   - Execute iteration with no requests
   - Process outputs correctly
   - Update request state

3. **Completion Handling**:
   - Completed requests removed from batch
   - Completion task set correctly
   - Completion reason accurate

4. **Capacity Integration**:
   - Check capacity before adding requests
   - Update capacity after token generation
   - Respect capacity limits

5. **Idle Handling**:
   - Handle idle iterations correctly
   - Throttle after max idle iterations
   - Resume when requests arrive

6. **Error Handling**:
   - Handle iteration errors gracefully
   - Record error metrics
   - Continue after errors

7. **Shutdown**:
   - Graceful shutdown completes pending requests
   - Timeout forces shutdown
   - Dispose cleans up resources

8. **Metrics Recording**:
   - Record iteration metrics
   - Record request completion metrics
   - Record batch utilization

---

## Success Criteria
- [ ] All public methods implemented
- [ ] Scheduler loop runs continuously
- [ ] Request lifecycle managed correctly
- [ ] Batch execution integrated with model executor
- [ ] Completion detection works end-to-end
- [ ] Capacity management enforced
- [ ] Metrics recorded correctly
- [ ] Shutdown works gracefully
- [ ] Unit tests cover all scenarios
