# Spec: Scheduler Unit Tests

## Overview
Comprehensive unit tests for the continuous batching scheduler components. Tests verify correctness, thread-safety, edge case handling, and integration between components.

## Test Structure

### Test Files to Create
1. `tests/MLFramework.Tests/Inference/ContinuousBatching/ContinuousBatchSchedulerTests.cs`
2. `tests/MLFramework.Tests/Inference/ContinuousBatching/RequestQueueTests.cs`
3. `tests/MLFramework.Tests/Inference/ContinuousBatching/BatchManagerTests.cs`
4. `tests/MLFramework.Tests/Inference/ContinuousBatching/CompletionDetectorTests.cs`
5. `tests/MLFramework.Tests/Inference/ContinuousBatching/CapacityManagerTests.cs`
6. `tests/MLFramework.Tests/Inference/ContinuousBatching/KVCache/ContinuousBatchKVCacheManagerTests.cs`
7. `tests/MLFramework.Tests/Inference/ContinuousBatching/Prefill/PrefillHandlerTests.cs`
8. `tests/MLFramework.Tests/Inference/ContinuousBatching/API/ContinuousBatchSchedulerClientTests.cs`
9. `tests/MLFramework.Tests/Inference/ContinuousBatching/Metrics/SchedulerMetricsCollectorTests.cs`
10. `tests/MLFramework.Tests/Inference/ContinuousBatching/Integration/ContinuousBatchingIntegrationTests.cs`

---

## Test: ContinuousBatchSchedulerTests

### Test Categories

#### 1. Lifecycle Tests
```csharp
[Test]
public async Task Start_StartsSchedulerTask()
{
    // Arrange
    var scheduler = CreateScheduler();

    // Act
    scheduler.Start();

    // Assert
    Assert.That(scheduler.IsRunning, Is.True);
    await scheduler.StopAsync();
}

[Test]
public async Task Stop_StopsSchedulerGracefully()
{
    // Arrange
    var scheduler = CreateScheduler();
    scheduler.Start();

    // Act
    await scheduler.StopAsync();

    // Assert
    Assert.That(scheduler.IsRunning, Is.False);
}

[Test]
public void Start_WhenAlreadyRunning_ThrowsInvalidOperationException()
{
    // Arrange
    var scheduler = CreateScheduler();
    scheduler.Start();

    // Act & Assert
    Assert.Throws<InvalidOperationException>(() => scheduler.Start());
}
```

#### 2. Request Enqueue Tests
```csharp
[Test]
public async Task EnqueueAsync_AddsRequestToQueue()
{
    // Arrange
    var scheduler = CreateScheduler();
    var request = CreateTestRequest();

    // Act
    var task = scheduler.EnqueueAsync(request, Priority.Normal);

    // Assert
    Assert.That(task, Is.Not.Null);
}
```

#### 3. Iteration Execution Tests
```csharp
[Test]
public async Task ExecuteIterationAsync_WithRequests_ProcessesBatch()
{
    // Arrange
    var scheduler = CreateScheduler();
    var request = CreateTestRequest();

    // Act
    var result = await scheduler.ExecuteIterationAsync();

    // Assert
    Assert.That(result, Is.Not.Null);
}

[Test]
public async Task ExecuteIterationAsync_WithNoRequests_ReturnsEmptyResult()
{
    // Arrange
    var scheduler = CreateScheduler();

    // Act
    var result = await scheduler.ExecuteIterationAsync();

    // Assert
    Assert.That(result.RequestCount, Is.EqualTo(0));
}
```

#### 4. Completion Handling Tests
```csharp
[Test]
public async Task ExecuteIterationAsync_WithCompletedRequest_RemovesFromBatch()
{
    // Arrange
    var scheduler = CreateScheduler();
    var request = CreateTestRequest();
    // Setup mock to return completion

    // Act
    var result = await scheduler.ExecuteIterationAsync();

    // Assert
    Assert.That(result.RequestsCompleted, Is.EqualTo(1));
}
```

---

## Test: RequestQueueTests

### Test Categories

#### 1. Basic Operations
```csharp
[Test]
public void Enqueue_AddsRequestToQueue()
{
    // Arrange
    var queue = CreateRequestQueue();
    var request = CreateTestRequest();

    // Act
    queue.Enqueue(request, Priority.Normal);

    // Assert
    Assert.That(queue.Count, Is.EqualTo(1));
    Assert.That(queue.IsEmpty, Is.False);
}

[Test]
public void Dequeue_RemovesRequestFromQueue()
{
    // Arrange
    var queue = CreateRequestQueue();
    var request = CreateTestRequest();
    queue.Enqueue(request, Priority.Normal);

    // Act
    var dequeued = queue.Dequeue();

    // Assert
    Assert.That(dequeued, Is.Not.Null);
    Assert.That(queue.Count, Is.EqualTo(0));
}
```

#### 2. Priority Ordering Tests
```csharp
[Test]
public void GetRequests_ReturnsInPriorityOrder()
{
    // Arrange
    var queue = CreateRequestQueue();
    var highPriority = CreateTestRequest();
    var lowPriority = CreateTestRequest();

    queue.Enqueue(lowPriority, Priority.Low);
    queue.Enqueue(highPriority, Priority.High);

    // Act
    var requests = queue.GetRequests(2, long.MaxValue);

    // Assert
    Assert.That(requests[0].Priority, Is.EqualTo(Priority.High));
    Assert.That(requests[1].Priority, Is.EqualTo(Priority.Low));
}
```

#### 3. Cancellation Tests
```csharp
[Test]
public void Dequeue_SkipsCancelledRequests()
{
    // Arrange
    var queue = CreateRequestQueue();
    var cts = new CancellationTokenSource();
    var request = CreateTestRequest(cts.Token);
    queue.Enqueue(request, Priority.Normal);

    cts.Cancel();

    // Act
    var dequeued = queue.Dequeue();

    // Assert
    Assert.That(dequeued, Is.Null);
}
```

#### 4. Thread Safety Tests
```csharp
[Test]
public async Task ConcurrentEnqueue_ThreadSafe()
{
    // Arrange
    var queue = CreateRequestQueue();
    var tasks = new List<Task>();

    // Act
    for (int i = 0; i < 100; i++)
    {
        tasks.Add(Task.Run(() =>
        {
            var request = CreateTestRequest();
            queue.Enqueue(request, Priority.Normal);
        }));
    }

    await Task.WhenAll(tasks);

    // Assert
    Assert.That(queue.Count, Is.EqualTo(100));
}
```

---

## Test: BatchManagerTests

### Test Categories

#### 1. Batch Construction Tests
```csharp
[Test]
public void PrepareNextIteration_CreatesBatchFromQueue()
{
    // Arrange
    var manager = CreateBatchManager();
    // Setup request queue with requests

    // Act
    var batch = manager.PrepareNextIteration();

    // Assert
    Assert.That(batch, Is.Not.Null);
}
```

#### 2. Completion Removal Tests
```csharp
[Test]
public void PrepareNextIteration_RemovesCompletedRequests()
{
    // Arrange
    var manager = CreateBatchManager();
    // Setup batch with completed request

    // Act
    var batch = manager.PrepareNextIteration();

    // Assert
    Assert.That(batch.Requests.Count, Is.EqualTo(0));
}
```

#### 3. Capacity Management Tests
```csharp
[Test]
public void PrepareNextIteration_RespectsBatchSizeLimit()
{
    // Arrange
    var manager = CreateBatchManager(
        new BatchConstraints(MaxBatchSize: 2, ...));
    // Setup queue with many requests

    // Act
    var batch = manager.PrepareNextIteration();

    // Assert
    Assert.That(batch.Size, Is.LessThanOrEqualTo(2));
}
```

---

## Test: CompletionDetectorTests

### Test Categories

#### 1. EOS Token Detection
```csharp
[Test]
public void CheckCompletion_WithEosToken_ReturnsTrue()
{
    // Arrange
    var detector = CreateCompletionDetector();
    var request = CreateTestRequest();
    request.GeneratedTokenIds.Add(EOS_TOKEN_ID);

    // Act
    var (isCompleted, reason) = detector.CheckCompletion(request);

    // Assert
    Assert.That(isCompleted, Is.True);
    Assert.That(reason, Is.EqualTo(CompletionReason.EosTokenReached));
}
```

#### 2. Max Tokens Detection
```csharp
[Test]
public void CheckCompletion_WithMaxTokensReached_ReturnsTrue()
{
    // Arrange
    var detector = CreateCompletionDetector();
    var request = CreateTestRequest(maxTokens: 5);
    request.GeneratedTokens = 5;

    // Act
    var (isCompleted, reason) = detector.CheckCompletion(request);

    // Assert
    Assert.That(isCompleted, Is.True);
    Assert.That(reason, Is.EqualTo(CompletionReason.MaxTokensReached));
}
```

#### 3. Priority Order Tests
```csharp
[Test]
public void CheckCompletion_Cancellation_CheckedFirst()
{
    // Arrange
    var detector = CreateCompletionDetector();
    var cts = new CancellationTokenSource();
    var request = CreateTestRequest(cts.Token);
    cts.Cancel();
    request.GeneratedTokenIds.Add(EOS_TOKEN_ID);

    // Act
    var (isCompleted, reason) = detector.CheckCompletion(request);

    // Assert
    Assert.That(isCompleted, Is.True);
    Assert.That(reason, Is.EqualTo(CompletionReason.Cancelled));
}
```

---

## Test: CapacityManagerTests

### Test Categories

#### 1. Allocation Tests
```csharp
[Test]
public void TryAllocate_WithCapacity_ReturnsTrue()
{
    // Arrange
    var manager = CreateCapacityManager();
    var request = CreateTestRequest();

    // Act
    var success = manager.TryAllocate(request, out var allocation);

    // Assert
    Assert.That(success, Is.True);
    Assert.That(allocation, Is.Not.Null);
}
```

#### 2. Capacity Limit Tests
```csharp
[Test]
public void TryAllocate_WhenCapacityExceeded_ReturnsFalse()
{
    // Arrange
    var manager = CreateCapacityManager(
        new CapacityConstraints(MaxBatchSize: 1, ...));
    var request1 = CreateTestRequest();
    var request2 = CreateTestRequest();

    manager.TryAllocate(request1, out _);

    // Act
    var success = manager.TryAllocate(request2, out _);

    // Assert
    Assert.That(success, Is.False);
}
```

#### 3. Release Tests
```csharp
[Test]
public void Release_FreesAllocatedResources()
{
    // Arrange
    var manager = CreateCapacityManager();
    var request = CreateTestRequest();
    manager.TryAllocate(request, out var allocation);

    // Act
    manager.Release(request.Id);

    // Assert
    var utilization = manager.GetUtilization();
    Assert.That(utilization.SlotUtilization, Is.EqualTo(0));
}
```

---

## Test: ContinuousBatchKVCacheManagerTests

### Test Categories

#### 1. Allocation Tests
```csharp
[Test]
public void AllocateCache_SuccessfullyAllocatesBlocks()
{
    // Arrange
    var manager = CreateKVCacheManager();
    var requestId = RequestId.New();

    // Act
    var bytes = manager.AllocateCache(requestId, 256);

    // Assert
    Assert.That(bytes, Is.GreaterThan(0));
}
```

#### 2. Release Tests
```csharp
[Test]
public void ReleaseCache_FreesBlocks()
{
    // Arrange
    var manager = CreateKVCacheManager();
    var requestId = RequestId.New();
    manager.AllocateCache(requestId, 256);

    // Act
    manager.ReleaseCache(requestId);

    // Assert
    Assert.That(manager.GetCurrentUsageBytes(), Is.EqualTo(0));
}
```

---

## Test: PrefillHandlerTests

### Test Categories

#### 1. Basic Prefill Tests
```csharp
[Test]
public async Task ProcessPrefillAsync_ProcessesPromptSuccessfully()
{
    // Arrange
    var handler = CreatePrefillHandler();
    var request = CreateTestRequest("Hello world");

    // Act
    var result = await handler.ProcessPrefillAsync(request);

    // Assert
    Assert.That(result.Success, Is.True);
    Assert.That(result.ProcessedTokens, Is.GreaterThan(0));
}
```

#### 2. Chunking Tests
```csharp
[Test]
public async Task ProcessPrefillAsync_WithLongPrompt_ChunksCorrectly()
{
    // Arrange
    var handler = CreatePrefillHandler(chunkSize: 10);
    var longPrompt = new string('A', 1000);
    var request = CreateTestRequest(longPrompt);

    // Act
    var result = await handler.ProcessPrefillAsync(request);

    // Assert
    Assert.That(result.ProcessedTokens, Is.GreaterThan(0));
}
```

---

## Test: ContinuousBatchSchedulerClientTests

### Test Categories

#### 1. GenerateText Tests
```csharp
[Test]
public async Task GenerateTextAsync_ReturnsGeneratedText()
{
    // Arrange
    var client = CreateSchedulerClient();

    // Act
    var result = await client.GenerateTextAsync("Hello");

    // Assert
    Assert.That(result, Is.Not.Null);
}
```

#### 2. Batch Generation Tests
```csharp
[Test]
public async Task GenerateBatchAsync_ProcessesMultipleRequests()
{
    // Arrange
    var client = CreateSchedulerClient();
    var requests = new List<GenerationRequest>
    {
        new("Hello"),
        new("World")
    };

    // Act
    var results = await client.GenerateBatchAsync(requests);

    // Assert
    Assert.That(results.Count, Is.EqualTo(2));
}
```

---

## Test: SchedulerMetricsCollectorTests

### Test Categories

#### 1. Recording Tests
```csharp
[Test]
public void RecordIteration_SavesMetrics()
{
    // Arrange
    var collector = CreateMetricsCollector();
    var result = CreateIterationResult();

    // Act
    collector.RecordIteration(result);

    // Assert
    var stats = collector.GetIterationStatistics();
    Assert.That(stats.TotalIterations, Is.EqualTo(1));
}
```

#### 2. Statistics Tests
```csharp
[Test]
public void GetRequestStatistics_CalculatesPercentiles()
{
    // Arrange
    var collector = CreateMetricsCollector();
    // Record multiple requests with different latencies

    // Act
    var stats = collector.GetRequestStatistics();

    // Assert
    Assert.That(stats.P50Latency, Is.GreaterThan(0));
    Assert.That(stats.P95Latency, Is.GreaterThanOrEqualTo(stats.P50Latency));
}
```

---

## Test: ContinuousBatchingIntegrationTests

### Test Categories

#### 1. End-to-End Tests
```csharp
[Test]
public async Task FullRequestLifecycle_CompletesSuccessfully()
{
    // Arrange
    var scheduler = CreateScheduler();
    var client = CreateSchedulerClient(scheduler);

    // Act
    var task = client.GenerateTextAsync("Hello world");
    scheduler.Start();

    var result = await task;
    await scheduler.StopAsync();

    // Assert
    Assert.That(result, Is.Not.Null);
}
```

#### 2. Multiple Requests Tests
```csharp
[Test]
public async Task MultipleConcurrentRequests_AllCompleteSuccessfully()
{
    // Arrange
    var scheduler = CreateScheduler();
    var client = CreateSchedulerClient(scheduler);
    var tasks = new List<Task<string>>();

    // Act
    for (int i = 0; i < 10; i++)
    {
        tasks.Add(client.GenerateTextAsync($"Request {i}"));
    }

    scheduler.Start();
    var results = await Task.WhenAll(tasks);
    await scheduler.StopAsync();

    // Assert
    Assert.That(results.Length, Is.EqualTo(10));
}
```

#### 3. Error Recovery Tests
```csharp
[Test]
public async Task ErrorInIteration_ContinuesProcessing()
{
    // Arrange
    var scheduler = CreateScheduler();
    // Setup mock to throw error then succeed

    // Act
    scheduler.Start();
    await Task.Delay(100);
    await scheduler.StopAsync();

    // Assert
    // Verify scheduler continued after error
}
```

---

## Common Test Helpers

### Helper Methods
```csharp
private ContinuousBatchScheduler CreateScheduler(
    Mock<RequestQueue>? queueMock = null,
    Mock<BatchManager>? batchManagerMock = null,
    Mock<CompletionDetector>? detectorMock = null,
    Mock<CapacityManager>? capacityMock = null,
    Mock<IModelExecutor>? executorMock = null,
    Mock<ISchedulerMetrics>? metricsMock = null)
{
    return new ContinuousBatchScheduler(
        queueMock?.Object ?? new Mock<RequestQueue>().Object,
        batchManagerMock?.Object ?? new Mock<BatchManager>().Object,
        detectorMock?.Object ?? new Mock<CompletionDetector>().Object,
        capacityMock?.Object ?? new Mock<CapacityManager>().Object,
        executorMock?.Object ?? new Mock<IModelExecutor>().Object,
        metricsMock?.Object ?? new Mock<ISchedulerMetrics>().Object,
        SchedulerConfiguration.Default
    );
}

private Request CreateTestRequest(
    string prompt = "Test prompt",
    int maxTokens = 100,
    CancellationToken? token = null,
    Priority priority = Priority.Normal)
{
    return new Request(
        RequestId.New(),
        prompt,
        maxTokens,
        token ?? CancellationToken.None,
        priority
    );
}

private IterationResult CreateIterationResult(
    int requestCount = 5,
    int tokensGenerated = 100)
{
    return new IterationResult(
        0,
        requestCount,
        tokensGenerated,
        1,
        TimeSpan.FromMilliseconds(100),
        1000
    );
}
```

---

## Test Framework Requirements

### Dependencies
- NUnit (or xUnit/MSTest)
- Moq (or NSubstitute)
- FluentAssertions (for readable assertions)

### Test Attributes
- `[Test]` - Unit test
- `[Test]` + `[TestCase(...)]` - Parameterized tests
- `[SetUp]` - Test setup
- `[TearDown]` - Test cleanup
- `[Timeout(...)]` - Timeout for long-running tests

### Assertions
- Use FluentAssertions for readable code:
  ```csharp
  result.Should().NotBeNull();
  result.RequestCount.Should().Be(5);
  ```

---

## Success Criteria

### Coverage
- [ ] All public methods tested
- [ ] All error paths tested
- [ ] Edge cases covered
- [ ] Thread safety verified
- [ ] Integration scenarios tested

### Quality
- [ ] Tests are independent and repeatable
- [ ] Tests run quickly (unit tests < 5s)
- [ ] Clear test names
- [ ] Good test documentation
- [ ] Proper mocking of dependencies

### Maintenance
- [ ] Tests are easy to understand
- [ ] Tests are easy to extend
- [ ] Tests provide good failure messages
- [ ] Tests use helper methods for common setup
