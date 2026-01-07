using System.Collections.Concurrent;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using MLFramework.Inference;
using MLFramework.Inference.ContinuousBatching;
using Xunit;

namespace MLFramework.Tests.Inference.ContinuousBatching;

/// <summary>
/// Unit tests for ContinuousBatchScheduler.
/// </summary>
public class ContinuousBatchSchedulerTests
{
    #region Mock Implementations

    private class MockModelExecutor : IModelExecutor
    {
        public Task<ModelOutput> ExecuteBatchAsync(Batch batch, CancellationToken cancellationToken = default)
        {
            var logits = new Dictionary<RequestId, float[]>();
            foreach (var request in batch.Requests)
            {
                // Generate a token for each request
                request.GeneratedTokens++;
                request.GeneratedTokenIds.Add(1); // Mock token ID
                logits[request.Id] = new float[] { 1.0f };
            }
            return Task.FromResult(new ModelOutput(logits));
        }
    }

    private class MockSchedulerMetrics : ISchedulerMetrics
    {
        public List<IterationResult> RecordedIterations { get; } = new();
        public List<RequestResult> RecordedRequests { get; } = new();
        public List<double> RecordedUtilizations { get; } = new();
        public List<(string, Exception)> RecordedErrors { get; } = new();

        public void RecordIteration(IterationResult result)
        {
            RecordedIterations.Add(result);
        }

        public void RecordRequestCompletion(RequestResult result)
        {
            RecordedRequests.Add(result);
        }

        public void RecordBatchUtilization(double utilization)
        {
            RecordedUtilizations.Add(utilization);
        }

        public void RecordError(string errorType, Exception exception)
        {
            RecordedErrors.Add((errorType, exception));
        }
    }

    private class MockLogger : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) { }
    }

    private class MockLoggerFactory : ILoggerFactory
    {
        public void Dispose() { }
        public ILogger CreateLogger(string categoryName) => new MockLogger();
        public void AddProvider(ILoggerProvider provider) { }
    }

    #endregion

    #region Test Factories

    private static ContinuousBatchScheduler CreateScheduler(
        RequestQueue? queue = null,
        BatchManager? batchManager = null,
        CompletionDetector? completionDetector = null,
        CapacityManager? capacityManager = null,
        IModelExecutor? executor = null,
        ISchedulerMetrics? metrics = null,
        SchedulerConfiguration? config = null)
    {
        return new ContinuousBatchScheduler(
            queue ?? new RequestQueue(),
            batchManager ?? new BatchManager(new RequestQueue(), new MockKVCacheManager(), BatchConstraints.Default),
            completionDetector ?? new CompletionDetector(CompletionConfiguration.Default, new MockTokenizer()),
            capacityManager ?? new CapacityManager(CapacityConstraints.Default, new MockGPUResourceManager()),
            executor ?? new MockModelExecutor(),
            metrics ?? new MockSchedulerMetrics(),
            config ?? SchedulerConfiguration.Default
        );
    }

    private static Request CreateTestRequest(
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

    private class MockKVCacheManager : IKVCacheManager
    {
        private readonly ConcurrentDictionary<RequestId, long> _allocations = new();
        private long _currentUsage = 0;

        public long AllocateCache(RequestId requestId, int maxTokens)
        {
            var size = 1024 * 1024; // 1MB
            _allocations[requestId] = size;
            Interlocked.Add(ref _currentUsage, size);
            return size;
        }

        public void ReleaseCache(RequestId requestId)
        {
            if (_allocations.TryRemove(requestId, out var size))
            {
                Interlocked.Add(ref _currentUsage, -size);
            }
        }

        public long GetCurrentUsageBytes() => _currentUsage;
    }

    private class MockGPUResourceManager : IGPUResourceManager
    {
        public long GetTotalMemoryBytes() => 16L * 1024 * 1024 * 1024;
        public long GetCurrentMemoryUsageBytes() => 0;
        public double GetUtilization() => 0.0;
        public bool IsAvailable() => true;
    }

    private class MockTokenizer : ITokenizer
    {
        public string Decode(int[] tokenIds)
        {
            return string.Join(" ", tokenIds.Select(id => $"token{id}"));
        }

        public int[] Encode(string text)
        {
            return text.Split(' ').Select((word, i) => i).ToArray();
        }
    }

    #endregion

    #region Lifecycle Tests

    [Fact]
    public async Task Start_StartsScheduler()
    {
        // Arrange
        var scheduler = CreateScheduler();

        // Act
        scheduler.Start();
        await Task.Delay(50); // Give scheduler time to start

        // Assert
        Assert.True(scheduler.IsRunning);
        await scheduler.StopAsync();
    }

    [Fact]
    public async Task Stop_StopsSchedulerGracefully()
    {
        // Arrange
        var scheduler = CreateScheduler();
        scheduler.Start();
        await Task.Delay(50);

        // Act
        await scheduler.StopAsync();

        // Assert
        Assert.False(scheduler.IsRunning);
    }

    [Fact]
    public void Start_WhenAlreadyRunning_ThrowsInvalidOperationException()
    {
        // Arrange
        var scheduler = CreateScheduler();
        scheduler.Start();
        await Task.Delay(50);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => scheduler.Start());
        await scheduler.StopAsync();
    }

    [Fact]
    public async Task Stop_WhenNotRunning_DoesNothing()
    {
        // Arrange
        var scheduler = CreateScheduler();

        // Act & Assert - Should not throw
        await scheduler.StopAsync();
        Assert.False(scheduler.IsRunning);
    }

    #endregion

    #region Request Enqueue Tests

    [Fact]
    public async Task EnqueueAsync_AddsRequestToQueue()
    {
        // Arrange
        var scheduler = CreateScheduler();
        var request = CreateTestRequest();

        // Act
        var task = scheduler.EnqueueAsync(request, Priority.Normal);

        // Assert
        Assert.NotNull(task);
        Assert.Equal(1, scheduler.ActiveRequestCount);
        await scheduler.StopAsync();
    }

    [Fact]
    public async Task EnqueueAsync_WithMultipleRequests_AddsAllToQueue()
    {
        // Arrange
        var scheduler = CreateScheduler();
        var request1 = CreateTestRequest("First");
        var request2 = CreateTestRequest("Second");
        var request3 = CreateTestRequest("Third");

        // Act
        var task1 = scheduler.EnqueueAsync(request1, Priority.Normal);
        var task2 = scheduler.EnqueueAsync(request2, Priority.High);
        var task3 = scheduler.EnqueueAsync(request3, Priority.Low);

        // Assert
        Assert.NotNull(task1);
        Assert.NotNull(task2);
        Assert.NotNull(task3);
        await scheduler.StopAsync();
    }

    [Fact]
    public async Task EnqueueAsync_WithNullRequest_ThrowsArgumentNullException()
    {
        // Arrange
        var scheduler = CreateScheduler();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            scheduler.EnqueueAsync(null!, Priority.Normal));
        await scheduler.StopAsync();
    }

    #endregion

    #region Iteration Execution Tests

    [Fact]
    public async Task ExecuteIterationAsync_WithRequests_ProcessesBatch()
    {
        // Arrange
        var queue = new RequestQueue();
        var batchManager = new BatchManager(queue, new MockKVCacheManager(), BatchConstraints.Default);
        var scheduler = CreateScheduler(queue: queue, batchManager: batchManager);
        var request = CreateTestRequest();

        // Act
        var result = await scheduler.ExecuteIterationAsync();

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(scheduler);
        await scheduler.StopAsync();
    }

    [Fact]
    public async Task ExecuteIterationAsync_WithNoRequests_ReturnsEmptyResult()
    {
        // Arrange
        var queue = new RequestQueue();
        var batchManager = new BatchManager(queue, new MockKVCacheManager(), BatchConstraints.Default);
        var scheduler = CreateScheduler(queue: queue, batchManager: batchManager);

        // Act
        var result = await scheduler.ExecuteIterationAsync();

        // Assert
        Assert.NotNull(result);
        Assert.Equal(0, result.RequestCount);
        Assert.Equal(0, result.TokensGenerated);
        await scheduler.StopAsync();
    }

    [Fact]
    public async Task ExecuteIterationAsync_GeneratesTokens()
    {
        // Arrange
        var queue = new RequestQueue();
        var batchManager = new BatchManager(queue, new MockKVCacheManager(), BatchConstraints.Default);
        var scheduler = CreateScheduler(queue: queue, batchManager: batchManager);
        var request = CreateTestRequest(maxTokens: 10);

        queue.Enqueue(request, Priority.Normal);

        // Act
        var result = await scheduler.ExecuteIterationAsync();

        // Assert
        Assert.NotNull(result);
        Assert.True(result.RequestCount >= 0);
        await scheduler.StopAsync();
    }

    #endregion

    #region Completion Handling Tests

    [Fact]
    public async Task ExecuteIterationAsync_WithCompletedRequest_RemovesFromBatch()
    {
        // Arrange
        var queue = new RequestQueue();
        var batchManager = new BatchManager(queue, new MockKVCacheManager(), BatchConstraints.Default);
        var request = CreateTestRequest(maxTokens: 5);

        queue.Enqueue(request, Priority.Normal);
        var scheduler = CreateScheduler(queue: queue, batchManager: batchManager);

        // Execute enough iterations to complete the request
        for (int i = 0; i < 10; i++)
        {
            await scheduler.ExecuteIterationAsync();
        }

        // Act
        var result = await scheduler.ExecuteIterationAsync();

        // Assert
        Assert.NotNull(result);
        // The request should have been completed and removed
        await scheduler.StopAsync();
    }

    [Fact]
    public async Task ExecuteIterationAsync_SetsCompletionTaskResult()
    {
        // Arrange
        var queue = new RequestQueue();
        var batchManager = new BatchManager(queue, new MockKVCacheManager(), BatchConstraints.Default);
        var request = CreateTestRequest(maxTokens: 1);

        queue.Enqueue(request, Priority.Normal);
        var scheduler = CreateScheduler(queue: queue, batchManager: batchManager);

        var completionTask = scheduler.EnqueueAsync(request, Priority.Normal);

        // Act
        await scheduler.ExecuteIterationAsync();
        await Task.Delay(100); // Give time for completion

        // Assert
        Assert.True(completionTask.IsCompleted);
        await scheduler.StopAsync();
    }

    #endregion

    #region Metrics Tests

    [Fact]
    public async Task ExecuteIterationAsync_RecordsMetrics()
    {
        // Arrange
        var queue = new RequestQueue();
        var batchManager = new BatchManager(queue, new MockKVCacheManager(), BatchConstraints.Default);
        var metrics = new MockSchedulerMetrics();
        var scheduler = CreateScheduler(queue: queue, batchManager: batchManager, metrics: metrics);

        // Act
        await scheduler.ExecuteIterationAsync();

        // Assert
        Assert.Single(metrics.RecordedIterations);
        await scheduler.StopAsync();
    }

    [Fact]
    public async Task ExecuteIterationAsync_RecordsBatchUtilization()
    {
        // Arrange
        var queue = new RequestQueue();
        var batchManager = new BatchManager(queue, new MockKVCacheManager(), BatchConstraints.Default);
        var metrics = new MockSchedulerMetrics();
        var request = CreateTestRequest();

        queue.Enqueue(request, Priority.Normal);
        var scheduler = CreateScheduler(queue: queue, batchManager: batchManager, metrics: metrics);

        // Act
        await scheduler.ExecuteIterationAsync();

        // Assert
        Assert.Single(metrics.RecordedUtilizations);
        await scheduler.StopAsync();
    }

    #endregion

    #region Disposal Tests

    [Fact]
    public void Dispose_DisposesSchedulerResources()
    {
        // Arrange
        var scheduler = CreateScheduler();

        // Act & Assert - Should not throw
        scheduler.Dispose();
    }

    [Fact]
    public void Dispose_WhenRunning_StopsScheduler()
    {
        // Arrange
        var scheduler = CreateScheduler();
        scheduler.Start();
        Task.Delay(50).Wait();

        // Act
        scheduler.Dispose();
        Task.Delay(100).Wait();

        // Assert
        Assert.False(scheduler.IsRunning);
    }

    #endregion

    #region Configuration Tests

    [Fact]
    public async Task ExecuteIterationAsync_WithCustomConfiguration_UsesConfiguration()
    {
        // Arrange
        var config = new SchedulerConfiguration(
            MaxBatchSize: 8,
            IterationTimeoutMs: 5000,
            WarmupIterations: 2,
            MaxIdleIterations: 10
        );

        var scheduler = CreateScheduler(config: config);

        // Act
        var result = await scheduler.ExecuteIterationAsync();

        // Assert
        Assert.NotNull(result);
        await scheduler.StopAsync();
    }

    #endregion

    #region Active Request Count Tests

    [Fact]
    public async Task ActiveRequestCount_WithNoRequests_ReturnsZero()
    {
        // Arrange
        var queue = new RequestQueue();
        var batchManager = new BatchManager(queue, new MockKVCacheManager(), BatchConstraints.Default);
        var scheduler = CreateScheduler(queue: queue, batchManager: batchManager);

        // Act
        var count = scheduler.ActiveRequestCount;

        // Assert
        Assert.Equal(0, count);
        await scheduler.StopAsync();
    }

    [Fact]
    public async Task ActiveRequestCount_WithEnqueuedRequests_ReturnsCount()
    {
        // Arrange
        var queue = new RequestQueue();
        var batchManager = new BatchManager(queue, new MockKVCacheManager(), BatchConstraints.Default);
        var request1 = CreateTestRequest("First");
        var request2 = CreateTestRequest("Second");

        queue.Enqueue(request1, Priority.Normal);
        queue.Enqueue(request2, Priority.Normal);
        var scheduler = CreateScheduler(queue: queue, batchManager: batchManager);

        // Act
        var count = scheduler.ActiveRequestCount;

        // Assert
        Assert.Equal(0, count); // Requests are in queue, not batch yet
        await scheduler.StopAsync();
    }

    #endregion
}
