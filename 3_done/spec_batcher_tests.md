# Spec: Batcher Tests

## Overview
Implement comprehensive unit tests for the dynamic batching system components.

## Technical Requirements

### Test File Structure
```
tests/Serving/
├── BatchingConfigurationTests.cs
├── BoundedRequestQueueTests.cs
├── BatchBuilderTests.cs
├── TensorOperationsTests.cs
├── DynamicBatcherTests.cs
├── ResponseScattererTests.cs
└── BatchingMetricsTests.cs
```

### Test Requirements

#### 1. BatchingConfigurationTests.cs
```csharp
namespace MLFramework.Tests.Serving;

[TestClass]
public class BatchingConfigurationTests
{
    [TestMethod]
    public void DefaultConfiguration_HasValidValues()
    {
        var config = BatchingConfiguration.Default();
        Assert.AreEqual(32, config.MaxBatchSize);
        Assert.AreEqual(TimeSpan.FromMilliseconds(5), config.MaxWaitTime);
        Assert.AreEqual(16, config.PreferBatchSize);
        Assert.AreEqual(100, config.MaxQueueSize);
        Assert.AreEqual(TimeoutStrategy.DispatchPartial, config.TimeoutStrategy);
    }

    [TestMethod]
    public void Validate_WithValidConfig_DoesNotThrow()
    {
        var config = BatchingConfiguration.Default();
        config.Validate(); // Should not throw
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void Validate_WithInvalidMaxBatchSize_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = 0;
        config.Validate();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void Validate_WithPreferBatchSizeGreaterThanMax_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = 16;
        config.PreferBatchSize = 32;
        config.Validate();
    }

    [TestMethod]
    public void AllTimeoutStrategyValues_AreAccessible()
    {
        Assert.AreEqual(3, Enum.GetValues(typeof(TimeoutStrategy)).Length);
    }
}
```

#### 2. BoundedRequestQueueTests.cs
```csharp
namespace MLFramework.Tests.Serving;

[TestClass]
public class BoundedRequestQueueTests
{
    [TestMethod]
    public async Task EnqueueAsync_WithCapacity_Success()
    {
        var queue = new BoundedRequestQueue<string, int>(10);
        var request = new QueuedRequest<string>("test");

        var result = await queue.TryEnqueueAsync(request);

        Assert.IsTrue(result);
        Assert.AreEqual(1, queue.Count);
    }

    [TestMethod]
    public async Task EnqueueAsync_WhenFull_ReturnsFalse()
    {
        var queue = new BoundedRequestQueue<string, int>(1);
        var request1 = new QueuedRequest<string>("test1");
        var request2 = new QueuedRequest<string>("test2");

        await queue.TryEnqueueAsync(request1);
        var result = await queue.TryEnqueueAsync(request2);

        Assert.IsFalse(result);
    }

    [TestMethod]
    public async Task Dequeue_WithMultipleItems_ReturnsCorrectCount()
    {
        var queue = new BoundedRequestQueue<string, int>(10);
        for (int i = 0; i < 5; i++)
        {
            await queue.TryEnqueueAsync(new QueuedRequest<string>($"test{i}"));
        }

        var items = queue.Dequeue(3);

        Assert.AreEqual(3, items.Count);
        Assert.AreEqual(2, queue.Count);
    }

    [TestMethod]
    public void IsEmpty_Initially_ReturnsTrue()
    {
        var queue = new BoundedRequestQueue<string, int>(10);
        Assert.IsTrue(queue.IsEmpty);
    }

    [TestMethod]
    public async Task IsFull_WhenAtCapacity_ReturnsTrue()
    {
        var queue = new BoundedRequestQueue<string, int>(2);
        await queue.TryEnqueueAsync(new QueuedRequest<string>("test1"));
        await queue.TryEnqueueAsync(new QueuedRequest<string>("test2"));

        Assert.IsTrue(queue.IsFull);
    }
}
```

#### 3. BatchBuilderTests.cs
```csharp
namespace MLFramework.Tests.Serving;

[TestClass]
public class BatchBuilderTests
{
    [TestMethod]
    public async Task BuildBatchAsync_WithFullBatch_ReturnsMaxSize()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(100),
            PreferBatchSize = 2
        };
        var queue = new BoundedRequestQueue<string, int>(10);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Enqueue more than MaxBatchSize
        for (int i = 0; i < 10; i++)
        {
            await queue.TryEnqueueAsync(new QueuedRequest<string>($"test{i}"));
        }

        var batch = await builder.BuildBatchAsync();

        Assert.AreEqual(4, batch.Requests.Count);
    }

    [TestMethod]
    public async Task BuildBatchAsync_WithTimeout_ReturnsPartialBatch()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 10,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            PreferBatchSize = 5,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };
        var queue = new BoundedRequestQueue<string, int>(10);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Enqueue only 2 items
        await queue.TryEnqueueAsync(new QueuedRequest<string>("test1"));
        await queue.TryEnqueueAsync(new QueuedRequest<string>("test2"));

        var batch = await builder.BuildBatchAsync();

        Assert.AreEqual(2, batch.Requests.Count);
    }

    [TestMethod]
    public async Task BuildBatchAsync_WithCancelationToken_CanCancel()
    {
        var config = new BatchingConfiguration.Default();
        var queue = new BoundedRequestQueue<string, int>(10);
        var builder = new BatchBuilder<string, int>(queue, config);
        var cts = new CancellationTokenSource();

        cts.CancelAfter(TimeSpan.FromMilliseconds(10));

        await Assert.ThrowsExceptionAsync<OperationCanceledException>(
            () => builder.BuildBatchAsync(cts.Token));
    }
}
```

#### 4. TensorOperationsTests.cs
```csharp
namespace MLFramework.Tests.Serving;

[TestClass]
public class TensorOperationsTests
{
    [TestMethod]
    public void StackWithPadding_WithSameLengthTensors_NoPadding()
    {
        var tensors = new List<Tensor>
        {
            new Tensor(new long[] { 3 }),
            new Tensor(new long[] { 3 }),
            new Tensor(new long[] { 3 })
        };

        var result = TensorOperations.StackWithPadding(tensors);

        Assert.AreEqual(new long[] { 3, 3 }, result.StackedTensor.Shape);
        Assert.AreEqual(3, result.OriginalLengths.Length);
    }

    [TestMethod]
    public void StackWithPadding_WithVariableLengthTensors_PostPads()
    {
        var tensors = new List<Tensor>
        {
            new Tensor(new long[] { 2 }),
            new Tensor(new long[] { 4 }),
            new Tensor(new long[] { 3 })
        };

        var result = TensorOperations.StackWithPadding(
            tensors,
            paddingValue: -1f,
            strategy: PaddingStrategy.Post);

        Assert.AreEqual(new long[] { 3, 4 }, result.StackedTensor.Shape);
        CollectionAssert.AreEqual(new[] { 2, 4, 3 }, result.OriginalLengths);
    }

    [TestMethod]
    public void Unstack_WithOriginalLengths_ReturnsCorrectSizes()
    {
        var stacked = new Tensor(new long[] { 3, 5 });
        var lengths = new[] { 2, 5, 3 };

        var result = TensorOperations.Unstack(stacked, lengths);

        Assert.AreEqual(3, result.Count);
        Assert.AreEqual(new long[] { 2 }, result[0].Shape);
        Assert.AreEqual(new long[] { 5 }, result[1].Shape);
        Assert.AreEqual(new long[] { 3 }, result[2].Shape);
    }
}
```

#### 5. ResponseScattererTests.cs
```csharp
namespace MLFramework.Tests.Serving;

[TestClass]
public class ResponseScattererTests
{
    [TestMethod]
    public void Scatter_WithAllResponses_SetsAllTasks()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string>("test1"),
            new QueuedRequest<string>("test2"),
            new QueuedRequest<string>("test3")
        };
        var responses = new List<int> { 1, 2, 3 };

        scatterer.Scatter(requests, responses);

        Assert.IsTrue(requests[0].ResponseSource.Task.IsCompleted);
        Assert.IsTrue(requests[1].ResponseSource.Task.IsCompleted);
        Assert.IsTrue(requests[2].ResponseSource.Task.IsCompleted);
    }

    [TestMethod]
    public void Scatter_WithException_SetsAllTasksToFaulted()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string>("test1"),
            new QueuedRequest<string>("test2")
        };
        var exception = new InvalidOperationException("Batch failed");

        scatterer.Scatter(requests, null, exception);

        Assert.IsTrue(requests[0].ResponseSource.Task.IsFaulted);
        Assert.IsTrue(requests[1].ResponseSource.Task.IsFaulted);
    }

    [TestMethod]
    public void ScatterWithPartialFailures_HandlesMixedResults()
    {
        var scatterer = new ResponseScatterer<int>();
        var requests = new[]
        {
            new QueuedRequest<string>("test1"),
            new QueuedRequest<string>("test2"),
            new QueuedRequest<string>("test3")
        };
        var responses = new List<int> { 1, 0, 3 };
        var exceptions = new List<Exception>
        {
            null,
            new InvalidOperationException("Failed"),
            null
        };

        scatterer.ScatterWithPartialFailures(requests, responses, exceptions);

        Assert.IsTrue(requests[0].ResponseSource.Task.IsCompletedSuccessfully);
        Assert.IsTrue(requests[1].ResponseSource.Task.IsFaulted);
        Assert.IsTrue(requests[2].ResponseSource.Task.IsCompletedSuccessfully);
    }
}
```

#### 6. BatchingMetricsTests.cs
```csharp
namespace MLFramework.Tests.Serving;

[TestClass]
public class BatchingMetricsTests
{
    [TestMethod]
    public void RecordBatch_UpdatesMetricsCorrectly()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordBatch(batchSize: 4, queueWaitMs: 10.5, processingMs: 25.3);

        var snapshot = collector.GetSnapshot(currentQueueDepth: 0);

        Assert.AreEqual(1, snapshot.TotalBatches);
        Assert.AreEqual(4, snapshot.AverageBatchSize);
        Assert.AreEqual(10.5, snapshot.AverageQueueWaitMs);
        Assert.AreEqual(25.3, snapshot.AverageBatchProcessingMs);
    }

    [TestMethod]
    public void GetBatchSizeDistribution_CategorizesCorrectly()
    {
        var collector = new BatchingMetricsCollector();
        collector.RecordBatch(3, 10, 20);  // Very small
        collector.RecordBatch(10, 10, 20); // Small
        collector.RecordBatch(20, 10, 20); // Medium
        collector.RecordBatch(40, 10, 20); // Large
        collector.RecordBatch(70, 10, 20); // Very large

        var dist = collector.GetBatchSizeDistribution();

        Assert.AreEqual(1, dist.VerySmall);
        Assert.AreEqual(1, dist.Small);
        Assert.AreEqual(1, dist.Medium);
        Assert.AreEqual(1, dist.Large);
        Assert.AreEqual(1, dist.VeryLarge);
    }

    [TestMethod]
    public void Reset_ClearsAllMetrics()
    {
        var collector = new BatchingMetricsCollector();
        collector.RecordBatch(4, 10, 20);
        collector.RecordRequestEnqueued(5);
        collector.RecordQueueRejection();

        collector.Reset();

        var snapshot = collector.GetSnapshot(0);
        Assert.AreEqual(0, snapshot.TotalBatches);
        Assert.AreEqual(0, snapshot.QueueFullRejections);
    }
}
```

#### 7. DynamicBatcherTests.cs (Integration Tests)
```csharp
namespace MLFramework.Tests.Serving;

[TestClass]
public class DynamicBatcherTests
{
    [TestMethod]
    public async Task ProcessAsync_WithSingleRequest_ProcessesCorrectly()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            PreferBatchSize = 2
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            return Task.FromResult(requests.Select(r => r.Length).ToList());
        });

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var result = await batcher.ProcessAsyncAsync("test");

        Assert.AreEqual(4, result);
    }

    [TestMethod]
    public async Task ProcessAsync_WithMultipleConcurrentRequests_BatchesCorrectly()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            PreferBatchSize = 2
        };

        int batchCount = 0;
        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            Interlocked.Increment(ref batchCount);
            return Task.FromResult(requests.Select(r => r.Length).ToList());
        });

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var tasks = Enumerable.Range(0, 8)
            .Select(i => batcher.ProcessAsyncAsync($"test{i}"))
            .ToArray();

        await Task.WhenAll(tasks);

        Assert.AreEqual(2, batchCount); // Should be 2 batches of 4
    }

    [TestMethod]
    public async Task ProcessAsync_WhenQueueFull_Throws()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 2
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
            Task.FromResult(requests.Select(r => r.Length).ToList()));

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        // Enqueue 3 requests (queue size is 2)
        var task1 = batcher.ProcessAsyncAsync("test1");
        var task2 = batcher.ProcessAsyncAsync("test2");

        // Third request should fail
        await Assert.ThrowsExceptionAsync<InvalidOperationException>(
            () => batcher.ProcessAsyncAsync("test3"));
    }

    [TestMethod]
    public async Task GetStatistics_ReturnsCorrectValues()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 10
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
            Task.Delay(10).ContinueWith(_ => requests.Select(r => r.Length).ToList()));

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var tasks = Enumerable.Range(0, 5)
            .Select(i => batcher.ProcessAsyncAsync($"test{i}"))
            .ToArray();

        var stats = batcher.GetStatistics();

        Assert.IsTrue(stats.CurrentQueueSize >= 0);
    }
}
```

## File Locations
- **Tests:** `tests/Serving/*.cs`

## Dependencies
- xUnit or MSTest framework
- `src/Serving/` components

## Key Testing Principles
1. **Unit Tests**: Test each component in isolation
2. **Concurrency Tests**: Verify thread safety
3. **Integration Tests**: Test end-to-end scenarios
4. **Edge Cases**: Test boundary conditions and error cases
5. **Performance Tests**: (Optional) Measure throughput and latency

## Success Criteria
- All unit tests pass
- Test coverage > 80% for batching components
- Tests verify all edge cases
- Integration tests validate end-to-end flow

## Notes
- Use async/await in all async test methods
- Use proper assertion messages for debugging
- Consider test data builders for complex test setup
- Mock external dependencies (e.g., tensor operations) if needed
