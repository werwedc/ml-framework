using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using MLFramework.Inference;
using MLFramework.Inference.ContinuousBatching;
using Xunit;

namespace MLFramework.Tests.Inference.ContinuousBatching;

/// <summary>
/// Unit tests for ContinuousBatchSchedulerClient.
/// </summary>
public class ContinuousBatchSchedulerClientTests
{
    #region Mock Implementations

    private class MockScheduler : ContinuousBatchScheduler
    {
        public List<Request> EnqueuedRequests { get; } = new();
        public TaskCompletionSource<string>? LastCompletionSource { get; set; }

        public MockScheduler(
            RequestQueue queue,
            BatchManager batchManager,
            CompletionDetector completionDetector,
            CapacityManager capacityManager,
            IModelExecutor executor,
            ISchedulerMetrics metrics,
            SchedulerConfiguration config) : base(
                queue, batchManager, completionDetector,
                capacityManager, executor, metrics, config)
        {
        }

        public new Task<string> EnqueueAsync(Request request, Priority priority = Priority.Normal)
        {
            EnqueuedRequests.Add(request);
            LastCompletionSource = new TaskCompletionSource<string>();
            return LastCompletionSource.Task;
        }
    }

    private class MockModelExecutor : IModelExecutor
    {
        public Task<ModelOutput> ExecuteBatchAsync(Batch batch, CancellationToken cancellationToken = default)
        {
            var logits = new Dictionary<RequestId, float[]>();
            foreach (var request in batch.Requests)
            {
                request.GeneratedTokens++;
                request.GeneratedTokenIds.Add(1);
                logits[request.Id] = new float[] { 1.0f };
            }
            return Task.FromResult(new ModelOutput(logits));
        }
    }

    private class MockSchedulerMetrics : ISchedulerMetrics
    {
        public void RecordIteration(IterationResult result) { }
        public void RecordRequestCompletion(RequestResult result) { }
        public void RecordBatchUtilization(double utilization) { }
        public void RecordError(string errorType, Exception exception) { }
    }

    private class MockLogger : ILogger
    {
        public List<(LogLevel, string)> LoggedMessages { get; } = new();

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            LoggedMessages.Add((logLevel, formatter(state, exception)));
        }
    }

    private class MockLoggerFactory : ILoggerFactory
    {
        private readonly MockLogger _logger = new();

        public MockLogger Logger => _logger;

        public void Dispose() { }
        public ILogger CreateLogger(string categoryName) => _logger;
        public void AddProvider(ILoggerProvider provider) { }
    }

    private class MockKVCacheManager : IKVCacheManager
    {
        private readonly ConcurrentDictionary<RequestId, long> _allocations = new();
        private long _currentUsage = 0;

        public long AllocateCache(RequestId requestId, int maxTokens)
        {
            var size = 1024 * 1024;
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

    #region Test Factories

    private static ContinuousBatchSchedulerClient CreateClient(
        MockScheduler? scheduler = null,
        SchedulerApiClientConfiguration? config = null,
        ILogger? logger = null)
    {
        return new ContinuousBatchSchedulerClient(
            scheduler ?? CreateMockScheduler(),
            config ?? new SchedulerApiClientConfiguration(EnableRequestLogging: false),
            logger ?? new MockLogger()
        );
    }

    private static MockScheduler CreateMockScheduler()
    {
        var queue = new RequestQueue();
        var batchManager = new BatchManager(queue, new MockKVCacheManager(), BatchConstraints.Default);
        var completionDetector = new CompletionDetector(CompletionConfiguration.Default, new MockTokenizer());
        var capacityManager = new CapacityManager(CapacityConstraints.Default, new MockGPUResourceManager());
        var executor = new MockModelExecutor();
        var metrics = new MockSchedulerMetrics();
        var config = SchedulerConfiguration.Default;

        return new MockScheduler(
            queue, batchManager, completionDetector,
            capacityManager, executor, metrics, config
        );
    }

    private static SchedulerApiClientConfiguration CreateConfig(bool enableLogging = false)
    {
        return new SchedulerApiClientConfiguration(EnableRequestLogging: enableLogging);
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidParameters_CreatesClient()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var config = CreateConfig();
        var logger = new MockLogger();

        // Act
        var client = new ContinuousBatchSchedulerClient(scheduler, config, logger);

        // Assert
        Assert.NotNull(client);
    }

    [Fact]
    public void Constructor_WithNullScheduler_ThrowsArgumentNullException()
    {
        // Arrange
        var config = CreateConfig();
        var logger = new MockLogger();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ContinuousBatchSchedulerClient(null!, config, logger));
    }

    [Fact]
    public void Constructor_WithNullConfig_ThrowsArgumentNullException()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var logger = new MockLogger();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ContinuousBatchSchedulerClient(scheduler, null!, logger));
    }

    [Fact]
    public void Constructor_WithNullLogger_ThrowsArgumentNullException()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var config = CreateConfig();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ContinuousBatchSchedulerClient(scheduler, config, null!));
    }

    #endregion

    #region GenerateTextAsync Tests

    [Fact]
    public async Task GenerateTextAsync_WithValidPrompt_ReturnsTask()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());

        // Act
        var task = client.GenerateTextAsync("Hello world", 100);

        // Assert
        Assert.NotNull(task);
        Assert.Single(scheduler.EnqueuedRequests);
        Assert.Equal("Hello world", scheduler.EnqueuedRequests[0].Prompt);
    }

    [Fact]
    public async Task GenerateTextAsync_WithCustomMaxTokens_UsesCustomValue()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());

        // Act
        var task = client.GenerateTextAsync("Hello", 256);

        // Assert
        Assert.Single(scheduler.EnqueuedRequests);
        Assert.Equal(256, scheduler.EnqueuedRequests[0].MaxTokens);
    }

    [Fact]
    public async Task GenerateTextAsync_WithDefaultMaxTokens_UsesDefault()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());

        // Act
        var task = client.GenerateTextAsync("Hello");

        // Assert
        Assert.Single(scheduler.EnqueuedRequests);
        Assert.Equal(256, scheduler.EnqueuedRequests[0].MaxTokens); // Default value
    }

    [Fact]
    public async Task GenerateTextAsync_WithCustomPriority_UsesPriority()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());

        // Act
        var task = client.GenerateTextAsync("Hello", 100, Priority.High);

        // Assert
        Assert.Single(scheduler.EnqueuedRequests);
        Assert.Equal(Priority.High, scheduler.EnqueuedRequests[0].Priority);
    }

    [Fact]
    public async Task GenerateTextAsync_WithDefaultPriority_UsesNormal()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());

        // Act
        var task = client.GenerateTextAsync("Hello", 100);

        // Assert
        Assert.Single(scheduler.EnqueuedRequests);
        Assert.Equal(Priority.Normal, scheduler.EnqueuedRequests[0].Priority);
    }

    [Fact]
    public async Task GenerateTextAsync_WithLogging_LogsRequest()
    {
        // Arrange
        var loggerFactory = new MockLoggerFactory();
        var logger = loggerFactory.Logger;
        var scheduler = CreateMockScheduler();
        var config = CreateConfig(enableLogging: true);
        var client = new ContinuousBatchSchedulerClient(scheduler, config, logger);

        // Act
        var task = client.GenerateTextAsync("Hello world", 100);

        // Assert
        Assert.NotEmpty(logger.LoggedMessages);
        Assert.True(logger.LoggedMessages.Any(m =>
            m.Item1 == LogLevel.Information &&
            m.Item2.Contains("Enqueueing generation request")));
    }

    [Fact]
    public async Task GenerateTextAsync_WithoutLogging_DoesNotLog()
    {
        // Arrange
        var loggerFactory = new MockLoggerFactory();
        var logger = loggerFactory.Logger;
        var scheduler = CreateMockScheduler();
        var config = CreateConfig(enableLogging: false);
        var client = new ContinuousBatchSchedulerClient(scheduler, config, logger);

        // Act
        var task = client.GenerateTextAsync("Hello world", 100);

        // Assert
        Assert.Empty(logger.LoggedMessages);
    }

    [Fact]
    public async Task GenerateTextAsync_WithCancellation_PropagatesCancellation()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());
        var cts = new CancellationTokenSource();

        // Act
        var task = client.GenerateTextAsync("Hello", 100, cts.Token);
        cts.Cancel();

        // Assert
        await Assert.ThrowsAsync<TaskCanceledException>(() => task);
    }

    [Fact]
    public async Task GenerateTextAsync_WithEmptyPrompt_CreatesRequest()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());

        // Act
        var task = client.GenerateTextAsync("", 100);

        // Assert
        Assert.Single(scheduler.EnqueuedRequests);
        Assert.Equal("", scheduler.EnqueuedRequests[0].Prompt);
    }

    [Fact]
    public async Task GenerateTextAsync_WithLongPrompt_CreatesRequest()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());
        var longPrompt = new string('A', 10000);

        // Act
        var task = client.GenerateTextAsync(longPrompt, 100);

        // Assert
        Assert.Single(scheduler.EnqueuedRequests);
        Assert.Equal(longPrompt, scheduler.EnqueuedRequests[0].Prompt);
    }

    #endregion

    #region EnqueueRequestAsync Tests

    [Fact]
    public async Task EnqueueRequestAsync_WithValidRequest_ReturnsTask()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());
        var request = new Request(
            RequestId.New(),
            "Test prompt",
            100,
            CancellationToken.None,
            Priority.Normal
        );

        // Act
        var task = client.EnqueueRequestAsync(request);

        // Assert
        Assert.NotNull(task);
        Assert.Single(scheduler.EnqueuedRequests);
    }

    [Fact]
    public async Task EnqueueRequestAsync_WithCustomPriority_UsesPriority()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());
        var request = new Request(
            RequestId.New(),
            "Test prompt",
            100,
            CancellationToken.None,
            Priority.Normal
        );

        // Act
        var task = client.EnqueueRequestAsync(request, Priority.High);

        // Assert
        Assert.Single(scheduler.EnqueuedRequests);
    }

    [Fact]
    public async Task EnqueueRequestAsync_WithNullRequest_ThrowsArgumentNullException()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            client.EnqueueRequestAsync(null!));
    }

    [Fact]
    public async Task EnqueueRequestAsync_WithLogging_LogsRequest()
    {
        // Arrange
        var loggerFactory = new MockLoggerFactory();
        var logger = loggerFactory.Logger;
        var scheduler = CreateMockScheduler();
        var config = CreateConfig(enableLogging: true);
        var client = new ContinuousBatchSchedulerClient(scheduler, config, logger);
        var request = new Request(
            RequestId.New(),
            "Test prompt",
            100,
            CancellationToken.None,
            Priority.Normal
        );

        // Act
        var task = client.EnqueueRequestAsync(request);

        // Assert
        Assert.NotEmpty(logger.LoggedMessages);
        Assert.True(logger.LoggedMessages.Any(m =>
            m.Item1 == LogLevel.Information &&
            m.Item2.Contains("Enqueueing request")));
    }

    #endregion

    #region CancelRequest Tests

    [Fact]
    public void CancelRequest_WithValidRequestId_LogsCancellation()
    {
        // Arrange
        var loggerFactory = new MockLoggerFactory();
        var logger = loggerFactory.Logger;
        var scheduler = CreateMockScheduler();
        var config = CreateConfig(enableLogging: true);
        var client = new ContinuousBatchSchedulerClient(scheduler, config, logger);
        var requestId = RequestId.New();

        // Act
        var result = client.CancelRequest(requestId);

        // Assert
        Assert.True(result); // Placeholder always returns true
        Assert.NotEmpty(logger.LoggedMessages);
        Assert.True(logger.LoggedMessages.Any(m =>
            m.Item1 == LogLevel.Information &&
            m.Item2.Contains("Cancelling request")));
    }

    [Fact]
    public void CancelRequest_WithoutLogging_DoesNotLog()
    {
        // Arrange
        var loggerFactory = new MockLoggerFactory();
        var logger = loggerFactory.Logger;
        var scheduler = CreateMockScheduler();
        var config = CreateConfig(enableLogging: false);
        var client = new ContinuousBatchSchedulerClient(scheduler, config, logger);
        var requestId = RequestId.New();

        // Act
        var result = client.CancelRequest(requestId);

        // Assert
        Assert.Empty(logger.LoggedMessages);
    }

    #endregion

    #region CancelAllRequests Tests

    [Fact]
    public void CancelAllRequests_LogsCancellation()
    {
        // Arrange
        var loggerFactory = new MockLoggerFactory();
        var logger = loggerFactory.Logger;
        var scheduler = CreateMockScheduler();
        var config = CreateConfig(enableLogging: true);
        var client = new ContinuousBatchSchedulerClient(scheduler, config, logger);

        // Act
        var count = client.CancelAllRequests();

        // Assert
        Assert.Equal(0, count); // Placeholder returns 0
        Assert.NotEmpty(logger.LoggedMessages);
        Assert.True(logger.LoggedMessages.Any(m =>
            m.Item1 == LogLevel.Information &&
            m.Item2.Contains("Cancelling all pending requests")));
    }

    [Fact]
    public void CancelAllRequests_WithoutLogging_DoesNotLog()
    {
        // Arrange
        var loggerFactory = new MockLoggerFactory();
        var logger = loggerFactory.Logger;
        var scheduler = CreateMockScheduler();
        var config = CreateConfig(enableLogging: false);
        var client = new ContinuousBatchSchedulerClient(scheduler, config, logger);

        // Act
        var count = client.CancelAllRequests();

        // Assert
        Assert.Empty(logger.LoggedMessages);
    }

    #endregion

    #region EstimateWaitTime Tests

    [Fact]
    public void EstimateWaitTime_WithNormalPriority_ReturnsEstimate()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());

        // Act
        var waitTime = client.EstimateWaitTime(Priority.Normal);

        // Assert
        Assert.NotNull(waitTime);
        Assert.True(waitTime.Value.TotalSeconds > 0);
    }

    [Fact]
    public void EstimateWaitTime_WithHighPriority_ReturnsEstimate()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());

        // Act
        var waitTime = client.EstimateWaitTime(Priority.High);

        // Assert
        Assert.NotNull(waitTime);
        Assert.True(waitTime.Value.TotalSeconds > 0);
    }

    [Fact]
    public void EstimateWaitTime_WithLowPriority_ReturnsEstimate()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());

        // Act
        var waitTime = client.EstimateWaitTime(Priority.Low);

        // Assert
        Assert.NotNull(waitTime);
        Assert.True(waitTime.Value.TotalSeconds > 0);
    }

    #endregion

    #region Dispose Tests

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var client = CreateClient();

        // Act & Assert - Should not throw
        client.Dispose();
        client.Dispose();
    }

    [Fact]
    public void Dispose_DoesNotThrow()
    {
        // Arrange
        var client = CreateClient();

        // Act & Assert
        client.Dispose();
    }

    #endregion

    #region Integration Tests

    [Fact]
    public async Task GenerateTextAsync_CompletesSuccessfully()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());
        var task = client.GenerateTextAsync("Hello world", 10);

        // Act
        scheduler.LastCompletionSource?.SetResult("Generated text");
        var result = await task;

        // Assert
        Assert.Equal("Generated text", result);
    }

    [Fact]
    public async Task GenerateTextAsync_FailsWithException_PropagatesException()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());
        var task = client.GenerateTextAsync("Hello world", 10);

        // Act
        scheduler.LastCompletionSource?.SetException(new InvalidOperationException("Test error"));

        // Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() => task);
    }

    [Fact]
    public async Task GenerateTextAsync_Cancelled_ThrowsTaskCanceledException()
    {
        // Arrange
        var scheduler = CreateMockScheduler();
        var client = CreateClient(scheduler, CreateConfig());
        var task = client.GenerateTextAsync("Hello world", 10);

        // Act
        scheduler.LastCompletionSource?.SetCanceled();

        // Assert
        await Assert.ThrowsAsync<TaskCanceledException>(() => task);
    }

    #endregion
}
