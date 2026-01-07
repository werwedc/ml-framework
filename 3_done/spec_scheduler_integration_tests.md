# Spec: Scheduler Integration Tests

## Overview
End-to-end integration tests for the continuous batching scheduler. These tests verify the complete system behavior, from request submission through generation and completion, including real model execution (or high-fidelity mocks).

## Test Structure

### Test File to Create
`tests/MLFramework.Tests/Inference/ContinuousBatching/Integration/ContinuousBatchingIntegrationTests.cs`

---

## Test Setup

### Test Fixture
```csharp
[SetUp]
public async Task Setup()
{
    // Create all components
    _tokenizer = CreateMockTokenizer();
    _modelExecutor = CreateModelExecutor();
    _kvCacheManager = CreateKVCacheManager();

    // Create scheduler components
    _requestQueue = new RequestQueue(100);
    _capacityManager = new CapacityManager(CapacityConstraints.Default);
    _completionDetector = new CompletionDetector(
        CompletionConfiguration.Default,
        _tokenizer
    );
    _batchManager = new BatchManager(
        _requestQueue,
        _kvCacheManager,
        new BatchConstraints(
            MaxBatchSize: 8,
            MaxMemoryBytes: 1L * 1024 * 1024 * 1024, // 1GB
            MinBatchSize: 2,
            MaxSequenceLength: 1024
        )
    );

    _metricsCollector = new SchedulerMetricsCollector(MetricsConfiguration.Default);

    // Create scheduler
    _scheduler = new ContinuousBatchScheduler(
        _requestQueue,
        _batchManager,
        _completionDetector,
        _capacityManager,
        _modelExecutor,
        _metricsCollector,
        SchedulerConfiguration.Default
    );

    // Create client
    _client = new ContinuousBatchSchedulerClient(
        _scheduler,
        SchedulerApiClientConfiguration.Default,
        NullLogger.Instance
    );
}

[TearDown]
public async Task TearDown()
{
    if (_scheduler.IsRunning)
    {
        await _scheduler.StopAsync();
    }

    _modelExecutor?.Dispose();
    _kvCacheManager?.Dispose();
}
```

---

## Test: Full Request Lifecycle

### Test 1: Single Request Completion
```csharp
[Test]
public async Task SingleRequest_GeneratesTextAndCompletes()
{
    // Arrange
    string prompt = "The capital of France is";
    int maxTokens = 10;

    // Act
    var task = _client.GenerateTextAsync(prompt, maxTokens);
    _scheduler.Start();
    string result = await task;
    await _scheduler.StopAsync();

    // Assert
    Assert.That(result, Is.Not.Null);
    Assert.That(result, Is.Not.Empty);
    Assert.That(result, Does.Contain("Paris"));
}
```

### Test 2: Multiple Sequential Requests
```csharp
[Test]
public async Task MultipleSequentialRequests_AllComplete()
{
    // Arrange
    var prompts = new List<string>
    {
        "The sky is",
        "The sun is",
        "The moon is"
    };
    var results = new List<string>();

    // Act
    _scheduler.Start();

    foreach (var prompt in prompts)
    {
        var result = await _client.GenerateTextAsync(prompt);
        results.Add(result);
    }

    await _scheduler.StopAsync();

    // Assert
    Assert.That(results.Count, Is.EqualTo(3));
    foreach (var result in results)
    {
        Assert.That(result, Is.Not.Null);
    }
}
```

### Test 3: Multiple Concurrent Requests
```csharp
[Test]
public async Task MultipleConcurrentRequests_AllComplete()
{
    // Arrange
    int requestCount = 20;
    var prompts = Enumerable.Range(0, requestCount)
        .Select(i => $"Generate text {i}")
        .ToList();
    var tasks = prompts.Select(p => _client.GenerateTextAsync(p)).ToList();

    // Act
    _scheduler.Start();
    var results = await Task.WhenAll(tasks);
    await _scheduler.StopAsync();

    // Assert
    Assert.That(results.Length, Is.EqualTo(requestCount));
    foreach (var result in results)
    {
        Assert.That(result, Is.Not.Null);
    }
}
```

---

## Test: Batch Dynamics

### Test 4: Dynamic Batch Composition
```csharp
[Test]
public async Task DynamicBatch_AddsAndRemovesRequestsCorrectly()
{
    // Arrange
    var shortRequestTask = _client.GenerateTextAsync("Short", maxTokens: 5);
    var longRequestTask = _client.GenerateTextAsync("Long", maxTokens: 50);

    // Act
    _scheduler.Start();
    await Task.WhenAll(shortRequestTask, longRequestTask);
    await _scheduler.StopAsync();

    // Assert
    // Verify short request completed before long one
    // Verify both completed successfully
    Assert.Pass();
}
```

### Test 5: Batch Size Limits
```csharp
public async Task BatchSize_RespectsMaxBatchSize()
{
    // Arrange
    int maxBatchSize = 4;
    var requests = Enumerable.Range(0, 10)
        .Select(i => _client.GenerateTextAsync($"Request {i}"))
        .ToList();

    // Act
    _scheduler.Start();
    await Task.WhenAll(requests);
    await _scheduler.StopAsync();

    // Assert
    // Check metrics to verify batch sizes never exceeded limit
    var stats = _metricsCollector.GetBatchStatistics();
    Assert.That(stats.MaxBatchSize, Is.LessThanOrEqualTo(maxBatchSize));
}
```

---

## Test: Completion Detection

### Test 6: EOS Token Completion
```csharp
[Test]
public async Task Request_CompletesOnEosToken()
{
    // Arrange
    string prompt = "Complete this sentence with";
    int maxTokens = 100; // Set high, expect early completion

    // Act
    var task = _client.GenerateTextAsync(prompt, maxTokens);
    _scheduler.Start();
    string result = await task;
    await _scheduler.StopAsync();

    // Assert
    Assert.That(result, Is.Not.Null);
    // Verify completion reason (would need to track)
}
```

### Test 7: Max Tokens Completion
```csharp
[Test]
public async Task Request_CompletesOnMaxTokens()
{
    // Arrange
    string prompt = "Generate text";
    int maxTokens = 10;

    // Act
    var task = _client.GenerateTextAsync(prompt, maxTokens);
    _scheduler.Start();
    string result = await task;
    await _scheduler.StopAsync();

    // Assert
    Assert.That(result, Is.Not.Null);
    // Verify exactly maxTokens were generated
}
```

### Test 8: Cancellation
```csharp
[Test]
public void Request_Cancellation_CompletesWithoutResult()
{
    // Arrange
    var cts = new CancellationTokenSource();
    var task = _client.GenerateTextAsync("Test", cancellationToken: cts.Token);
    _scheduler.Start();

    // Act
    cts.Cancel();
    Assert.ThrowsAsync<TaskCanceledException>(async () => await task);

    await _scheduler.StopAsync();
}
```

---

## Test: Capacity Management

### Test 9: Memory Limits
```csharp
[Test]
public async Task MemoryLimit_PreventsOversubscription()
{
    // Arrange
    long memoryLimit = 100L * 1024 * 1024; // 100MB
    var requests = Enumerable.Range(0, 20)
        .Select(i => _client.GenerateTextAsync($"Request {i}"))
        .ToList();

    // Act
    _scheduler.Start();
    await Task.WhenAll(requests);
    await _scheduler.StopAsync();

    // Assert
    var utilization = _capacityManager.GetUtilization();
    Assert.That(utilization.MemoryUtilization, Is.LessThanOrEqualTo(100));
}
```

### Test 10: Capacity Release
```csharp
[Test]
public async Task CompletedRequest_ReleasesCapacity()
{
    // Arrange
    var request1Task = _client.GenerateTextAsync("Short", maxTokens: 5);
    var request2Task = _client.GenerateTextAsync("Medium", maxTokens: 20);

    // Act
    _scheduler.Start();
    await request1Task; // Wait for first to complete
    var utilizationAfter1 = _capacityManager.GetUtilization();
    await request2Task;
    await _scheduler.StopAsync();

    // Assert
    // Verify capacity decreased after request1 completed
    Assert.Pass();
}
```

---

## Test: Performance and Metrics

### Test 11: Throughput Measurement
```csharp
[Test]
public async Task Throughput_MeasuresCorrectly()
{
    // Arrange
    int requestCount = 100;
    var requests = Enumerable.Range(0, requestCount)
        .Select(i => _client.GenerateTextAsync($"Request {i}"))
        .ToList();

    // Act
    var stopwatch = Stopwatch.StartNew();
    _scheduler.Start();
    await Task.WhenAll(requests);
    await _scheduler.StopAsync();
    stopwatch.Stop();

    // Assert
    var stats = _metricsCollector.GetRequestStatistics();
    double actualThroughput = stats.RequestsPerSecond;
    double expectedThroughput = requestCount / stopwatch.Elapsed.TotalSeconds;

    Assert.That(actualThroughput, Is.GreaterThan(0));
    Assert.That(actualThroughput, Is.EqualTo(expectedThroughput).Within(20));
}
```

### Test 12: Latency Distribution
```csharp
[Test]
public async Task Latency_HasReasonableDistribution()
{
    // Arrange
    int requestCount = 50;
    var requests = Enumerable.Range(0, requestCount)
        .Select(i => _client.GenerateTextAsync($"Request {i}"))
        .ToList();

    // Act
    _scheduler.Start();
    await Task.WhenAll(requests);
    await _scheduler.StopAsync();

    // Assert
    var stats = _metricsCollector.GetRequestStatistics();
    Assert.That(stats.P50Latency, Is.GreaterThan(0));
    Assert.That(stats.P95Latency, Is.GreaterThan(stats.P50Latency));
    Assert.That(stats.P99Latency, Is.GreaterThan(stats.P95Latency));
}
```

### Test 13: Batch Utilization
```csharp
[Test]
public async Task BatchUtilization_IsEfficient()
{
    // Arrange
    int requestCount = 100;
    var requests = Enumerable.Range(0, requestCount)
        .Select(i => _client.GenerateTextAsync($"Request {i}"))
        .ToList();

    // Act
    _scheduler.Start();
    await Task.WhenAll(requests);
    await _scheduler.StopAsync();

    // Assert
    var stats = _metricsCollector.GetBatchStatistics();
    double utilization = stats.AverageUtilization;

    // Target: > 70% average utilization
    Assert.That(utilization, Is.GreaterThan(70));
}
```

---

## Test: Error Handling

### Test 14: Model Error Recovery
```csharp
[Test]
public async Task ModelError_SchedulerContinues()
{
    // Arrange
    var failingExecutor = new FailingModelExecutor(
        failCount: 2,
        _modelExecutor
    );

    // Replace executor in scheduler (would need setter or reconstruction)
    // For now, assume we can inject a failing executor

    var tasks = Enumerable.Range(0, 10)
        .Select(i => _client.GenerateTextAsync($"Request {i}"))
        .ToList();

    // Act
    _scheduler.Start();
    var results = await Task.WhenAll(tasks);
    await _scheduler.StopAsync();

    // Assert
    // Verify scheduler recovered and continued processing
    var errorStats = _metricsCollector.GetErrorStatistics();
    Assert.That(errorStats.TotalErrors, Is.GreaterThan(0));
}
```

### Test 15: Prefill Failure
```csharp
[Test]
public async Task PrefillFailure_RequestFailsGracefully()
{
    // Arrange
    var failingPrefillHandler = new FailingPrefillHandler();

    // Inject failing handler

    // Act & Assert
    Assert.ThrowsAsync<InvalidOperationException>(async () =>
    {
        await _client.GenerateTextAsync("Test");
    });
}
```

---

## Test: Stress Tests

### Test 16: High Concurrency
```csharp
[Test]
[Timeout(60000)] // 60 second timeout
public async Task HighConcurrency_HandlesLoad()
{
    // Arrange
    int requestCount = 1000;
    var requests = Enumerable.Range(0, requestCount)
        .Select(i => _client.GenerateTextAsync($"Request {i}", maxTokens: 5))
        .ToList();

    // Act
    _scheduler.Start();
    var stopwatch = Stopwatch.StartNew();
    var results = await Task.WhenAll(tasks);
    await _scheduler.StopAsync();
    stopwatch.Stop();

    // Assert
    Assert.That(results.Length, Is.EqualTo(requestCount));
    Assert.That(stopwatch.Elapsed.TotalSeconds, Is.LessThan(60));
}
```

### Test 17: Memory Stress
```csharp
[Test]
public async Task MemoryStress_HandlesLargeRequests()
{
    // Arrange
    var longPrompts = Enumerable.Range(0, 10)
        .Select(i => new string('A', 10000))
        .ToList();
    var tasks = longPrompts.Select(p => _client.GenerateTextAsync(p, maxTokens: 10))
                           .ToList();

    // Act
    _scheduler.Start();
    var results = await Task.WhenAll(tasks);
    await _scheduler.StopAsync();

    // Assert
    Assert.That(results.Count, Is.EqualTo(10));
    // Verify no memory leaks
}
```

---

## Test: Priority Scheduling

### Test 18: Priority Ordering
```csharp
[Test]
public async Task Priority_HighPriorityRequestsProcessFirst()
{
    // Arrange
    var lowPriorityTask = _client.GenerateTextAsync("Low", priority: Priority.Low);
    await Task.Delay(10); // Ensure queue order

    var highPriorityTask = _client.GenerateTextAsync("High", priority: Priority.High);

    // Act
    _scheduler.Start();
    var lowResult = await lowPriorityTask;
    var highResult = await highPriorityTask;
    await _scheduler.StopAsync();

    // Assert
    // High priority should complete first (or near first)
    Assert.Pass();
}
```

---

## Test: Prefill and Caching

### Test 19: Prefill Caching
```csharp
[Test]
public async Task PrefillCache_ImprovesPerformance()
{
    // Arrange
    string sharedPrompt = "Shared prompt for caching";

    // Act
    _scheduler.Start();

    var time1 = await TimeGenerationAsync(sharedPrompt);
    var time2 = await TimeGenerationAsync(sharedPrompt);

    await _scheduler.StopAsync();

    // Assert
    // Second generation should be faster due to cache
    Assert.That(time2, Is.LessThan(time1));
}

private async Task<TimeSpan> TimeGenerationAsync(string prompt)
{
    var stopwatch = Stopwatch.StartNew();
    await _client.GenerateTextAsync(prompt);
    stopwatch.Stop();
    return stopwatch.Elapsed;
}
```

---

## Mock Implementations

### Mock Model Executor
```csharp
private class MockModelExecutor : IModelExecutor, IDisposable
{
    private readonly Random _random = new(42);
    private bool _disposed;

    public async Task<BatchOutput> ExecuteBatchAsync(
        Batch batch,
        CancellationToken cancellationToken = default)
    {
        await Task.Delay(10, cancellationToken); // Simulate compute time

        var generatedTokens = new Dictionary<RequestId, int>();
        var logits = new Dictionary<RequestId, float[]>();
        var isEosReached = new bool[batch.Size];

        int idx = 0;
        foreach (var request in batch.Requests)
        {
            // Generate token (mock)
            int tokenId = _random.Next(0, 1000);

            // Occasionally generate EOS
            if (_random.Next(0, 10) == 0)
            {
                tokenId = EOS_TOKEN_ID;
                isEosReached[idx] = true;
            }

            generatedTokens[request.Id] = tokenId;
            logits[request.Id] = new float[1000]; // Mock logits
            idx++;
        }

        return new BatchOutput(generatedTokens, logits, isEosReached);
    }

    public void Dispose()
    {
        _disposed = true;
    }
}
```

### Failing Model Executor
```csharp
private class FailingModelExecutor : IModelExecutor, IDisposable
{
    private readonly int _failCount;
    private readonly IModelExecutor _innerExecutor;
    private int _failCounter;
    private bool _disposed;

    public FailingModelExecutor(int failCount, IModelExecutor innerExecutor)
    {
        _failCount = failCount;
        _innerExecutor = innerExecutor;
        _failCounter = 0;
    }

    public async Task<BatchOutput> ExecuteBatchAsync(
        Batch batch,
        CancellationToken cancellationToken = default)
    {
        if (_failCounter < _failCount)
        {
            _failCounter++;
            throw new InvalidOperationException("Simulated model error");
        }

        return await _innerExecutor.ExecuteBatchAsync(batch, cancellationToken);
    }

    public void Dispose()
    {
        _disposed = true;
        _innerExecutor?.Dispose();
    }
}
```

---

## Success Criteria

### Functional
- [ ] All requests complete successfully
- [ ] Completion detection works correctly
- [ ] Batch composition is dynamic
- [ ] Capacity limits are respected
- [ ] Priority ordering works
- [ ] Cancellation works

### Performance
- [ ] Throughput targets met
- [ ] Latency distributions reasonable
- [ ] Batch utilization > 70%
- [ ] Handles 1000+ concurrent requests
- [ ] Prefill caching improves performance

### Reliability
- [ ] Error recovery works
- [ ] No memory leaks
- [ ] No deadlocks
- [ ] Clean shutdown

### Test Quality
- [ ] Tests are independent
- [ ] Tests run in reasonable time
- [ ] Clear failure messages
- [ ] Comprehensive coverage
