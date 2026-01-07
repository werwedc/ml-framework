# Spec: Continuous Batching Scheduler API

## Overview
Implement the public API surface for the continuous batching scheduler. The API provides a clean, easy-to-use interface for enqueueing requests, monitoring status, and managing scheduler lifecycle.

## Class: ContinuousBatchSchedulerClient
```csharp
public class ContinuousBatchSchedulerClient : IDisposable
{
    private readonly ContinuousBatchScheduler _scheduler;
    private readonly SchedulerApiClientConfiguration _config;
    private readonly ILogger _logger;

    public ContinuousBatchSchedulerClient(
        ContinuousBatchScheduler scheduler,
        SchedulerApiClientConfiguration config,
        ILogger logger)
    {
        _scheduler = scheduler;
        _config = config;
        _logger = logger;
    }

    // Enqueue a text generation request
    public Task<string> GenerateTextAsync(
        string prompt,
        int maxTokens = 256,
        CancellationToken cancellationToken = default,
        Priority priority = Priority.Normal,
        GenerationOptions? options = null);

    // Enqueue a request with full control
    public Task<string> EnqueueRequestAsync(
        Request request,
        Priority priority = Priority.Normal);

    // Enqueue multiple requests concurrently
    public Task<List<GenerationResult>> GenerateBatchAsync(
        List<GenerationRequest> requests,
        CancellationToken cancellationToken = default);

    // Get scheduler status
    public SchedulerStatus GetStatus();

    // Get detailed statistics
    public SchedulerStatistics GetStatistics();

    // Cancel a specific request
    public bool CancelRequest(RequestId requestId);

    // Cancel all pending requests
    public int CancelAllRequests();

    // Get estimated wait time for a new request
    public TimeSpan? EstimateWaitTime(Priority priority = Priority.Normal);
}
```

---

## Class: SchedulerApiClientConfiguration
```csharp
public record class SchedulerApiClientConfiguration(
    int DefaultMaxTokens,
    double TimeoutMultiplier,
    bool EnableRequestLogging,
    bool EnableStatisticsCollection,
    int MaxConcurrentEnqueue
)
{
    public static readonly SchedulerApiClientConfiguration Default = new(
        DefaultMaxTokens: 256,
        TimeoutMultiplier: 1.5,
        EnableRequestLogging: true,
        EnableStatisticsCollection: true,
        MaxConcurrentEnqueue: 100
    );
}
```

**Purpose**: Configure client API behavior.

---

## Class: GenerationOptions
```csharp
public record class GenerationOptions(
    float? Temperature,                    // Sampling temperature (0-2)
    float? TopP,                          // Nucleus sampling threshold
    int? TopK,                            // Top-K sampling
    float? FrequencyPenalty,              // Repetition penalty
    float? PresencePenalty,               // Presence penalty
    List<string>? StopSequences,          // Stop sequences
    int? Seed,                            // Random seed for reproducibility
    bool? EchoPrompt,                     // Include prompt in output
    string? Grammar,                      // Structured output grammar
    Dictionary<string, object>? Metadata // Additional metadata
)
{
    public static readonly GenerationOptions Default = new(
        Temperature: 1.0f,
        TopP: 1.0f,
        TopK: null,
        FrequencyPenalty: 0.0f,
        PresencePenalty: 0.0f,
        StopSequences: null,
        Seed: null,
        EchoPrompt: false,
        Grammar: null,
        Metadata: null
    );
}
```

**Purpose**: Generation parameters.

---

## Class: GenerationRequest
```csharp
public record class GenerationRequest(
    string Prompt,
    int MaxTokens,
    Priority Priority,
    GenerationOptions? Options,
    CancellationToken CancellationToken,
    Dictionary<string, object>? Metadata
)
{
    public GenerationRequest(
        string prompt,
        int maxTokens = 256,
        Priority priority = Priority.Normal,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default,
        Dictionary<string, object>? metadata = null)
        : this(prompt, maxTokens, priority, options, cancellationToken, metadata)
    {
    }
}
```

**Purpose**: Encapsulate a generation request.

---

## Class: GenerationResult
```csharp
public record class GenerationResult(
    RequestId RequestId,
    string GeneratedText,
    int TokensGenerated,
    CompletionReason Reason,
    TimeSpan ProcessingTime,
    TimeSpan QueueTime,
    Dictionary<string, object>? Metadata
)
```

**Purpose**: Result of a generation request.

---

## Class: SchedulerStatus
```csharp
public record class SchedulerStatus(
    bool IsRunning,
    int ActiveRequests,
    int QueuedRequests,
    int CompletedRequests,
    double GpuUtilization,
    double MemoryUtilization,
    DateTime LastUpdateTime
)
```

**Purpose**: Current scheduler status.

---

## Class: SchedulerStatistics
```csharp
public record class SchedulerStatistics(
    int TotalRequests,
    int TotalCompletedRequests,
    int TotalFailedRequests,
    int TotalCancelledRequests,
    TimeSpan TotalProcessingTime,
    double AverageRequestsPerSecond,
    double AverageTokensPerSecond,
    double P50Latency,
    double P95Latency,
    double P99Latency,
    double AverageBatchUtilization,
    DateTime StartTime
)
```

**Purpose**: Historical scheduler statistics.

---

## Implementation Details

### GenerateTextAsync
```csharp
public async Task<string> GenerateTextAsync(
    string prompt,
    int maxTokens = 256,
    CancellationToken cancellationToken = default,
    Priority priority = Priority.Normal,
    GenerationOptions? options = null)
{
    if (_config.EnableRequestLogging)
    {
        _logger.LogInformation("Enqueueing generation request: prompt length={PromptLength}, maxTokens={MaxTokens}, priority={Priority}",
            prompt.Length, maxTokens, priority);
    }

    // Create request
    var requestId = RequestId.New();
    var request = new Request(
        requestId,
        prompt,
        maxTokens,
        cancellationToken,
        priority
    );

    // Apply options if provided
    if (options != null)
    {
        // Options would be attached to request metadata
        // For now, we'll store them
        request.Metadata = new Dictionary<string, object>
        {
            ["options"] = options
        };
    }

    try
    {
        // Enqueue and wait for completion
        var result = await _scheduler.EnqueueAsync(request, priority);
        return result;
    }
    catch (OperationCanceledException)
    {
        _logger.LogInformation("Generation request cancelled: {RequestId}", requestId);
        throw;
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Generation request failed: {RequestId}", requestId);
        throw;
    }
}
```

**Requirements**:
- Create request from parameters
- Apply generation options
- Enqueue to scheduler
- Handle exceptions
- Log if enabled

---

### EnqueueRequestAsync
```csharp
public Task<string> EnqueueRequestAsync(
    Request request,
    Priority priority = Priority.Normal)
{
    if (_config.EnableRequestLogging)
    {
        _logger.LogInformation("Enqueueing request: {RequestId}, priority={Priority}",
            request.Id, priority);
    }

    return _scheduler.EnqueueAsync(request, priority);
}
```

**Requirements**:
- Forward request to scheduler
- Log if enabled

---

### GenerateBatchAsync
```csharp
public async Task<List<GenerationResult>> GenerateBatchAsync(
    List<GenerationRequest> requests,
    CancellationToken cancellationToken = default)
{
    if (requests.Count == 0)
        return new List<GenerationResult>();

    if (_config.EnableRequestLogging)
    {
        _logger.LogInformation("Enqueueing batch of {Count} requests", requests.Count);
    }

    // Create tasks for all requests
    var tasks = requests.Select(async req =>
    {
        try
        {
            var requestId = RequestId.New();
            var request = new Request(
                requestId,
                req.Prompt,
                req.MaxTokens,
                req.CancellationToken,
                req.Priority
            );

            // Attach metadata
            if (req.Metadata != null)
            {
                request.Metadata = req.Metadata;
            }

            // Enqueue
            await _scheduler.EnqueueAsync(request, req.Priority);

            // Wait for completion
            var result = await request.CompletionSource.Task;

            return new GenerationResult(
                requestId,
                result,
                request.GeneratedTokens,
                CompletionReason.EosTokenReached, // Simplified
                TimeSpan.Zero, // Simplified
                TimeSpan.Zero, // Simplified
                req.Metadata
            );
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Batch request failed");
            // Return error result
            return new GenerationResult(
                RequestId.Empty,
                string.Empty,
                0,
                CompletionReason.Cancelled,
                TimeSpan.Zero,
                TimeSpan.Zero,
                null
            );
        }
    }).ToArray();

    await Task.WhenAll(tasks);

    return tasks.Select(t => t.Result).ToList();
}
```

**Requirements**:
- Handle multiple requests concurrently
- Respect individual cancellation tokens
- Collect results from all requests
- Handle failures gracefully

---

### GetStatus
```csharp
public SchedulerStatus GetStatus()
{
    // Get status from scheduler components
    var capacityUtil = _capacityManager.GetUtilization(); // Would need access
    var batchStats = _batchManager.GetStats(); // Would need access

    return new SchedulerStatus(
        IsRunning: _scheduler.IsRunning,
        ActiveRequests: _scheduler.ActiveRequestCount,
        QueuedRequests: _requestQueue.Count, // Would need access
        CompletedRequests: 0, // Would need to track
        GpuUtilization: 0.0, // Would need GPU manager
        MemoryUtilization: capacityUtil.MemoryUtilization,
        LastUpdateTime: DateTime.UtcNow
    );
}
```

**Requirements**:
- Collect status from all components
- Return current state
- Thread-safe

---

### GetStatistics
```csharp
public SchedulerStatistics GetStatistics()
{
    // This would be implemented with a statistics collector
    // For now, return placeholder
    return new SchedulerStatistics(
        TotalRequests: 0,
        TotalCompletedRequests: 0,
        TotalFailedRequests: 0,
        TotalCancelledRequests: 0,
        TotalProcessingTime: TimeSpan.Zero,
        AverageRequestsPerSecond: 0.0,
        AverageTokensPerSecond: 0.0,
        P50Latency: 0.0,
        P95Latency: 0.0,
        P99Latency: 0.0,
        AverageBatchUtilization: 0.0,
        StartTime: DateTime.UtcNow
    );
}
```

**Requirements**:
- Collect historical statistics
- Calculate percentiles
- Return comprehensive metrics

---

### CancelRequest
```csharp
public bool CancelRequest(RequestId requestId)
{
    // Implementation would involve:
    // 1. Check if request is in queue
    // 2. If in queue, remove it
    // 3. If in batch, mark for cancellation
    // 4. Cancel completion task

    if (_config.EnableRequestLogging)
    {
        _logger.LogInformation("Cancelling request: {RequestId}", requestId);
    }

    // Placeholder implementation
    return true;
}
```

**Requirements**:
- Cancel request wherever it is
- Handle queue and batch cases
- Return success status

---

### CancelAllRequests
```csharp
public int CancelAllRequests()
{
    // Cancel all queued requests
    int cancelledCount = 0;

    if (_config.EnableRequestLogging)
    {
        _logger.LogInformation("Cancelling all pending requests");
    }

    // Placeholder implementation
    return cancelledCount;
}
```

**Requirements**:
- Cancel all pending requests
- Return count cancelled
- Log if enabled

---

### EstimateWaitTime
```csharp
public TimeSpan? EstimateWaitTime(Priority priority = Priority.Normal)
{
    // Estimate based on:
    // 1. Queue size at priority level
    // 2. Average tokens per request
    // 3. Current throughput

    // Placeholder implementation
    return TimeSpan.FromSeconds(1.0);
}
```

**Requirements**:
- Estimate based on current load
- Consider priority
- Return null if cannot estimate

---

## Files to Create

### Implementation
- `src/MLFramework/Inference/ContinuousBatching/API/ContinuousBatchSchedulerClient.cs`
- `src/MLFramework/Inference/ContinuousBatching/API/SchedulerApiClientConfiguration.cs`
- `src/MLFramework/Inference/ContinuousBatching/API/GenerationOptions.cs`
- `src/MLFramework/Inference/ContinuousBatching/API/GenerationRequest.cs`
- `src/MLFramework/Inference/ContinuousBatching/API/GenerationResult.cs`
- `src/MLFramework/Inference/ContinuousBatching/API/SchedulerStatus.cs`
- `src/MLFramework/Inference/ContinuousBatching/API/SchedulerStatistics.cs`

### Tests
- `tests/MLFramework.Tests/Inference/ContinuousBatching/API/ContinuousBatchSchedulerClientTests.cs`

---

## Dependencies
- `spec_continuous_scheduler_core.md` (ContinuousBatchScheduler)
- `spec_continuous_batching_models.md` (Request, RequestId, Priority, CompletionReason)

---

## Testing Requirements

### Unit Tests (with Mocks)
1. **GenerateTextAsync**:
   - Enqueue request successfully
   - Return generated text
   - Handle cancellation
   - Handle errors

2. **EnqueueRequestAsync**:
   - Forward request to scheduler
   - Return completion task

3. **GenerateBatchAsync**:
   - Handle multiple requests
   - Return all results
   - Handle partial failures
   - Handle batch cancellation

4. **GetStatus**:
   - Return current status
   - Include all fields
   - Thread-safe

5. **GetStatistics**:
   - Return historical statistics
   - Calculate percentiles correctly
   - Include all metrics

6. **CancelRequest**:
   - Cancel request successfully
   - Handle non-existent request
   - Return correct status

7. **CancelAllRequests**:
   - Cancel all pending requests
   - Return correct count

8. **EstimateWaitTime**:
   - Return reasonable estimate
   - Handle different priorities
   - Return null if unknown

---

## Success Criteria
- [ ] All public methods implemented
- [ ] API is clean and intuitive
- [ ] GenerateTextAsync works correctly
- [ ] Batch operations work
- [ ] Status and statistics accurate
- [ ] Cancellation works
- [ ] Wait time estimation reasonable
- [ ] Logging works when enabled
- [ ] Unit tests cover all scenarios
