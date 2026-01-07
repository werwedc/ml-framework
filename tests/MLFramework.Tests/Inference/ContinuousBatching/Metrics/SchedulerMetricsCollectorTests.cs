using MLFramework.Inference.ContinuousBatching;
using Xunit;

namespace MLFramework.Tests.Inference.ContinuousBatching;

/// <summary>
/// Unit tests for SchedulerMetricsCollector.
/// </summary>
public class SchedulerMetricsCollectorTests
{
    #region Test Factories

    private static SchedulerMetricsCollector CreateCollector(
        MetricsConfiguration? config = null)
    {
        return new SchedulerMetricsCollector(
            config ?? new MetricsConfiguration(
                MaxRequestSamples: 100,
                MaxIterationSamples: 100,
                MaxBatchSamples: 100,
                CounterWindowSeconds: 60
            )
        );
    }

    private static IterationResult CreateIterationResult(
        int iterationNumber = 1,
        int requestCount = 5,
        int tokensGenerated = 10,
        int requestsCompleted = 1,
        TimeSpan? processingTime = null,
        long memoryBytesUsed = 1024)
    {
        return new IterationResult(
            iterationNumber,
            requestCount,
            tokensGenerated,
            requestsCompleted,
            processingTime ?? TimeSpan.FromMilliseconds(100),
            memoryBytesUsed
        );
    }

    private static RequestResult CreateRequestResult(
        RequestId? requestId = null,
        int tokensGenerated = 10,
        CompletionReason reason = CompletionReason.EosTokenReached,
        TimeSpan? processingTime = null)
    {
        return new RequestResult(
            requestId ?? RequestId.New(),
            "Generated text",
            tokensGenerated,
            reason,
            processingTime ?? TimeSpan.FromMilliseconds(100)
        );
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidConfig_CreatesCollector()
    {
        // Arrange
        var config = new MetricsConfiguration(
            MaxRequestSamples: 100,
            MaxIterationSamples: 100,
            MaxBatchSamples: 100,
            CounterWindowSeconds: 60
        );

        // Act
        var collector = new SchedulerMetricsCollector(config);

        // Assert
        Assert.NotNull(collector);
    }

    [Fact]
    public void Constructor_WithNullConfig_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new SchedulerMetricsCollector(null!));
    }

    #endregion

    #region RecordIteration Tests

    [Fact]
    public void RecordIteration_WithValidResult_SavesMetrics()
    {
        // Arrange
        var collector = CreateCollector();
        var result = CreateIterationResult();

        // Act
        collector.RecordIteration(result);

        // Assert
        var stats = collector.GetIterationStatistics();
        Assert.Equal(1, stats.TotalIterations);
    }

    [Fact]
    public void RecordIteration_MultipleCalls_SavesAllMetrics()
    {
        // Arrange
        var collector = CreateCollector();
        var result1 = CreateIterationResult(iterationNumber: 1);
        var result2 = CreateIterationResult(iterationNumber: 2);
        var result3 = CreateIterationResult(iterationNumber: 3);

        // Act
        collector.RecordIteration(result1);
        collector.RecordIteration(result2);
        collector.RecordIteration(result3);

        // Assert
        var stats = collector.GetIterationStatistics();
        Assert.Equal(3, stats.TotalIterations);
    }

    [Fact]
    public void RecordIteration_UpdatesTokenCounter()
    {
        // Arrange
        var collector = CreateCollector();
        var result = CreateIterationResult(tokensGenerated: 50);

        // Act
        collector.RecordIteration(result);

        // Assert
        var stats = collector.GetIterationStatistics();
        Assert.True(stats.AverageTokensPerIteration > 0);
    }

    [Fact]
    public void RecordIteration_UpdatesProcessingTimeStats()
    {
        // Arrange
        var collector = CreateCollector();
        var result = CreateIterationResult(processingTime: TimeSpan.FromMilliseconds(200));

        // Act
        collector.RecordIteration(result);

        // Assert
        var stats = collector.GetIterationStatistics();
        Assert.True(stats.AverageProcessingTimeMs > 0);
    }

    #endregion

    #region RecordRequestCompletion Tests

    [Fact]
    public void RecordRequestCompletion_WithValidResult_SavesMetrics()
    {
        // Arrange
        var collector = CreateCollector();
        var result = CreateRequestResult();

        // Act
        collector.RecordRequestCompletion(result);

        // Assert
        var stats = collector.GetRequestStatistics();
        Assert.Equal(1, stats.TotalRequests);
    }

    [Fact]
    public void RecordRequestCompletion_MultipleCalls_SavesAllMetrics()
    {
        // Arrange
        var collector = CreateCollector();
        var result1 = CreateRequestResult();
        var result2 = CreateRequestResult();
        var result3 = CreateRequestResult();

        // Act
        collector.RecordRequestCompletion(result1);
        collector.RecordRequestCompletion(result2);
        collector.RecordRequestCompletion(result3);

        // Assert
        var stats = collector.GetRequestStatistics();
        Assert.Equal(3, stats.TotalRequests);
    }

    [Fact]
    public void RecordRequestCompletion_WithCancelledRequest_IncrementsCancelledCount()
    {
        // Arrange
        var collector = CreateCollector();
        var result = CreateRequestResult(reason: CompletionReason.Cancelled);

        // Act
        collector.RecordRequestCompletion(result);

        // Assert
        var stats = collector.GetRequestStatistics();
        Assert.Equal(1, stats.TotalRequests);
        Assert.Equal(1, stats.CancelledRequests);
    }

    [Fact]
    public void RecordRequestCompletion_ExcludesCancelledFromCompleted()
    {
        // Arrange
        var collector = CreateCollector();
        var completedResult = CreateRequestResult(reason: CompletionReason.EosTokenReached);
        var cancelledResult = CreateRequestResult(reason: CompletionReason.Cancelled);

        // Act
        collector.RecordRequestCompletion(completedResult);
        collector.RecordRequestCompletion(cancelledResult);

        // Assert
        var stats = collector.GetRequestStatistics();
        Assert.Equal(2, stats.TotalRequests);
        Assert.Equal(1, stats.CompletedRequests);
        Assert.Equal(1, stats.CancelledRequests);
    }

    #endregion

    #region RecordBatchUtilization Tests

    [Fact]
    public void RecordBatchUtilization_WithValidValue_SavesMetrics()
    {
        // Arrange
        var collector = CreateCollector();

        // Act
        collector.RecordBatchUtilization(0.5);

        // Assert
        var stats = collector.GetBatchStatistics();
        Assert.Equal(1, stats.TotalBatches);
    }

    [Fact]
    public void RecordBatchUtilization_MultipleValues_CalculatesAverage()
    {
        // Arrange
        var collector = CreateCollector();

        // Act
        collector.RecordBatchUtilization(0.5);
        collector.RecordBatchUtilization(0.7);
        collector.RecordBatchUtilization(0.9);

        // Assert
        var stats = collector.GetBatchStatistics();
        Assert.Equal(3, stats.TotalBatches);
        Assert.Equal(0.7, stats.AverageUtilization, 0.01);
    }

    [Fact]
    public void RecordBatchUtilization_WithZeroValue_SavesMetrics()
    {
        // Arrange
        var collector = CreateCollector();

        // Act
        collector.RecordBatchUtilization(0.0);

        // Assert
        var stats = collector.GetBatchStatistics();
        Assert.Equal(1, stats.TotalBatches);
    }

    [Fact]
    public void RecordBatchUtilization_WithFullValue_SavesMetrics()
    {
        // Arrange
        var collector = CreateCollector();

        // Act
        collector.RecordBatchUtilization(1.0);

        // Assert
        var stats = collector.GetBatchStatistics();
        Assert.Equal(1, stats.TotalBatches);
    }

    #endregion

    #region RecordError Tests

    [Fact]
    public void RecordError_WithValidError_SavesMetrics()
    {
        // Arrange
        var collector = CreateCollector();
        var exception = new Exception("Test error");

        // Act
        collector.RecordError("TestErrorType", exception);

        // Assert
        var stats = collector.GetErrorStatistics();
        Assert.Equal(1, stats.TotalErrors);
        Assert.True(stats.ErrorsByType.ContainsKey("TestErrorType"));
    }

    [Fact]
    public void RecordError_MultipleErrors_IncrementsCount()
    {
        // Arrange
        var collector = CreateCollector();
        var exception1 = new Exception("Error 1");
        var exception2 = new Exception("Error 2");

        // Act
        collector.RecordError("ErrorType1", exception1);
        collector.RecordError("ErrorType1", exception2);

        // Assert
        var stats = collector.GetErrorStatistics();
        Assert.Equal(2, stats.TotalErrors);
        Assert.Equal(2, stats.ErrorsByType["ErrorType1"]);
    }

    [Fact]
    public void RecordError_DifferentErrorTypes_GroupsByType()
    {
        // Arrange
        var collector = CreateCollector();

        // Act
        collector.RecordError("ErrorType1", new Exception("Error 1"));
        collector.RecordError("ErrorType2", new Exception("Error 2"));
        collector.RecordError("ErrorType1", new Exception("Error 3"));

        // Assert
        var stats = collector.GetErrorStatistics();
        Assert.Equal(3, stats.TotalErrors);
        Assert.Equal(2, stats.ErrorsByType["ErrorType1"]);
        Assert.Equal(1, stats.ErrorsByType["ErrorType2"]);
    }

    [Fact]
    public void RecordError_UpdatesLastErrorTime()
    {
        // Arrange
        var collector = CreateCollector();
        var beforeRecording = DateTime.UtcNow;

        // Act
        collector.RecordError("TestError", new Exception("Test"));

        // Assert
        var stats = collector.GetErrorStatistics();
        Assert.True(stats.LastErrorTime >= beforeRecording);
    }

    #endregion

    #region GetRequestStatistics Tests

    [Fact]
    public void GetRequestStatistics_WithNoRequests_ReturnsEmptyStats()
    {
        // Arrange
        var collector = CreateCollector();

        // Act
        var stats = collector.GetRequestStatistics();

        // Assert
        Assert.Equal(0, stats.TotalRequests);
        Assert.Equal(0, stats.CompletedRequests);
        Assert.Equal(0, stats.CancelledRequests);
        Assert.Equal(0, stats.AverageTokensPerRequest);
    }

    [Fact]
    public void GetRequestStatistics_CalculatesAverageTokens()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordRequestCompletion(CreateRequestResult(tokensGenerated: 10));
        collector.RecordRequestCompletion(CreateRequestResult(tokensGenerated: 20));
        collector.RecordRequestCompletion(CreateRequestResult(tokensGenerated: 30));

        // Act
        var stats = collector.GetRequestStatistics();

        // Assert
        Assert.Equal(20.0, stats.AverageTokensPerRequest, 0.01);
    }

    [Fact]
    public void GetRequestStatistics_CalculatesPercentiles()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordRequestCompletion(CreateRequestResult(processingTime: TimeSpan.FromMilliseconds(100)));
        collector.RecordRequestCompletion(CreateRequestResult(processingTime: TimeSpan.FromMilliseconds(200)));
        collector.RecordRequestCompletion(CreateRequestResult(processingTime: TimeSpan.FromMilliseconds(300)));
        collector.RecordRequestCompletion(CreateRequestResult(processingTime: TimeSpan.FromMilliseconds(400)));
        collector.RecordRequestCompletion(CreateRequestResult(processingTime: TimeSpan.FromMilliseconds(500)));

        // Act
        var stats = collector.GetRequestStatistics();

        // Assert
        Assert.True(stats.P50Latency > 0);
        Assert.True(stats.P95Latency >= stats.P50Latency);
        Assert.True(stats.P99Latency >= stats.P95Latency);
    }

    [Fact]
    public void GetRequestStatistics_CalculatesRequestRate()
    {
        // Arrange
        var collector = CreateCollector();
        for (int i = 0; i < 10; i++)
        {
            collector.RecordRequestCompletion(CreateRequestResult());
        }

        // Act
        var stats = collector.GetRequestStatistics();

        // Assert
        Assert.True(stats.RequestsPerSecond >= 0);
    }

    [Fact]
    public void GetRequestStatistics_CalculatesTokenRate()
    {
        // Arrange
        var collector = CreateCollector();
        for (int i = 0; i < 10; i++)
        {
            collector.RecordRequestCompletion(CreateRequestResult(tokensGenerated: 50));
        }

        // Act
        var stats = collector.GetRequestStatistics();

        // Assert
        Assert.True(stats.TokensPerSecond >= 0);
    }

    #endregion

    #region GetIterationStatistics Tests

    [Fact]
    public void GetIterationStatistics_WithNoIterations_ReturnsEmptyStats()
    {
        // Arrange
        var collector = CreateCollector();

        // Act
        var stats = collector.GetIterationStatistics();

        // Assert
        Assert.Equal(0, stats.TotalIterations);
        Assert.Equal(0, stats.AverageRequestsPerIteration);
        Assert.Equal(0, stats.AverageTokensPerIteration);
    }

    [Fact]
    public void GetIterationStatistics_CalculatesAverageRequests()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordIteration(CreateIterationResult(requestCount: 5));
        collector.RecordIteration(CreateIterationResult(requestCount: 10));
        collector.RecordIteration(CreateIterationResult(requestCount: 15));

        // Act
        var stats = collector.GetIterationStatistics();

        // Assert
        Assert.Equal(10, stats.AverageRequestsPerIteration);
    }

    [Fact]
    public void GetIterationStatistics_CalculatesAverageTokens()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordIteration(CreateIterationResult(tokensGenerated: 10));
        collector.RecordIteration(CreateIterationResult(tokensGenerated: 20));
        collector.RecordIteration(CreateIterationResult(tokensGenerated: 30));

        // Act
        var stats = collector.GetIterationStatistics();

        // Assert
        Assert.Equal(20, stats.AverageTokensPerIteration);
    }

    [Fact]
    public void GetIterationStatistics_CalculatesAverageProcessingTime()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordIteration(CreateIterationResult(processingTime: TimeSpan.FromMilliseconds(100)));
        collector.RecordIteration(CreateIterationResult(processingTime: TimeSpan.FromMilliseconds(200)));

        // Act
        var stats = collector.GetIterationStatistics();

        // Assert
        Assert.Equal(150.0, stats.AverageProcessingTimeMs, 0.01);
    }

    [Fact]
    public void GetIterationStatistics_CalculatesUtilization()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordIteration(CreateIterationResult(requestCount: 16));
        collector.RecordIteration(CreateIterationResult(requestCount: 24));

        // Act
        var stats = collector.GetIterationStatistics();

        // Assert
        Assert.True(stats.AverageUtilization > 0);
    }

    #endregion

    #region GetBatchStatistics Tests

    [Fact]
    public void GetBatchStatistics_WithNoBatches_ReturnsEmptyStats()
    {
        // Arrange
        var collector = CreateCollector();

        // Act
        var stats = collector.GetBatchStatistics();

        // Assert
        Assert.Equal(0, stats.TotalBatches);
        Assert.Equal(0.0, stats.AverageBatchSize);
        Assert.Equal(0, stats.MaxBatchSize);
    }

    [Fact]
    public void GetBatchStatistics_CalculatesAverageBatchSize()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordBatchUtilization(0.5);
        collector.RecordBatchUtilization(0.75);
        collector.RecordBatchUtilization(1.0);

        // Act
        var stats = collector.GetBatchStatistics();

        // Assert
        Assert.Equal(3, stats.TotalBatches);
        Assert.Equal(0.75, stats.AverageUtilization, 0.01);
    }

    [Fact]
    public void GetBatchStatistics_FindsMaxAndMinSize()
    {
        // Arrange
        var collector = CreateCollector();

        // Act
        collector.RecordBatchUtilization(0.25);
        collector.RecordBatchUtilization(0.5);
        collector.RecordBatchUtilization(0.75);
        collector.RecordBatchUtilization(1.0);

        var stats = collector.GetBatchStatistics();

        // Assert
        Assert.Equal(4, stats.TotalBatches);
        Assert.Equal(0.25, stats.AverageUtilization, 0.01);
    }

    #endregion

    #region GetErrorStatistics Tests

    [Fact]
    public void GetErrorStatistics_WithNoErrors_ReturnsEmptyStats()
    {
        // Arrange
        var collector = CreateCollector();

        // Act
        var stats = collector.GetErrorStatistics();

        // Assert
        Assert.Equal(0, stats.TotalErrors);
        Assert.Empty(stats.ErrorsByType);
        Assert.Equal(0.0, stats.ErrorRate);
    }

    [Fact]
    public void GetErrorStatistics_CalculatesErrorRate()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordRequestCompletion(CreateRequestResult());
        collector.RecordRequestCompletion(CreateRequestResult());
        collector.RecordError("TestError", new Exception("Test"));

        // Act
        var stats = collector.GetErrorStatistics();

        // Assert
        Assert.Equal(1, stats.TotalErrors);
        Assert.Equal(0.5, stats.ErrorRate, 0.01);
    }

    #endregion

    #region GetSnapshot Tests

    [Fact]
    public void GetSnapshot_ReturnsCompleteSnapshot()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordIteration(CreateIterationResult());
        collector.RecordRequestCompletion(CreateRequestResult());
        collector.RecordBatchUtilization(0.5);

        // Act
        var snapshot = collector.GetSnapshot();

        // Assert
        Assert.NotNull(snapshot);
        Assert.NotNull(snapshot.RequestStats);
        Assert.NotNull(snapshot.IterationStats);
        Assert.NotNull(snapshot.BatchStats);
        Assert.NotNull(snapshot.ErrorStats);
        Assert.True(snapshot.SnapshotTime > DateTime.MinValue);
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Reset_ClearsAllMetrics()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordIteration(CreateIterationResult());
        collector.RecordRequestCompletion(CreateRequestResult());
        collector.RecordBatchUtilization(0.5);
        collector.RecordError("TestError", new Exception("Test"));

        // Act
        collector.Reset();

        // Assert
        Assert.Equal(0, collector.GetIterationStatistics().TotalIterations);
        Assert.Equal(0, collector.GetRequestStatistics().TotalRequests);
        Assert.Equal(0, collector.GetBatchStatistics().TotalBatches);
        Assert.Equal(0, collector.GetErrorStatistics().TotalErrors);
    }

    [Fact]
    public void Reset_CanRecordNewMetricsAfterReset()
    {
        // Arrange
        var collector = CreateCollector();
        collector.RecordIteration(CreateIterationResult());

        // Act
        collector.Reset();
        collector.RecordIteration(CreateIterationResult());

        // Assert
        Assert.Equal(1, collector.GetIterationStatistics().TotalIterations);
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public void ConcurrentRecordIteration_ThreadSafe()
    {
        // Arrange
        var collector = CreateCollector();
        var tasks = new List<Task>();

        // Act
        for (int i = 0; i < 100; i++)
        {
            int index = i;
            tasks.Add(Task.Run(() =>
            {
                collector.RecordIteration(CreateIterationResult(iterationNumber: index));
            }));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        var stats = collector.GetIterationStatistics();
        Assert.Equal(100, stats.TotalIterations);
    }

    [Fact]
    public void ConcurrentRecordRequestCompletion_ThreadSafe()
    {
        // Arrange
        var collector = CreateCollector();
        var tasks = new List<Task>();

        // Act
        for (int i = 0; i < 100; i++)
        {
            tasks.Add(Task.Run(() =>
            {
                collector.RecordRequestCompletion(CreateRequestResult());
            }));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        var stats = collector.GetRequestStatistics();
        Assert.Equal(100, stats.TotalRequests);
    }

    [Fact]
    public void ConcurrentRecordAndStatistics_ThreadSafe()
    {
        // Arrange
        var collector = CreateCollector();
        var tasks = new List<Task>();

        // Act
        for (int i = 0; i < 50; i++)
        {
            int index = i;
            tasks.Add(Task.Run(() =>
            {
                if (index % 2 == 0)
                {
                    collector.RecordIteration(CreateIterationResult());
                    collector.GetIterationStatistics();
                }
                else
                {
                    collector.RecordRequestCompletion(CreateRequestResult());
                    collector.GetRequestStatistics();
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert - Should complete without exceptions
        Assert.True(collector.GetIterationStatistics().TotalIterations > 0);
        Assert.True(collector.GetRequestStatistics().TotalRequests > 0);
    }

    #endregion

    #region Circular Buffer Tests

    [Fact]
    public void RecordIteration_BeyondMaxSamples_DropsOldest()
    {
        // Arrange
        var config = new MetricsConfiguration(
            MaxRequestSamples: 10,
            MaxIterationSamples: 5,
            MaxBatchSamples: 10,
            CounterWindowSeconds: 60
        );
        var collector = new SchedulerMetricsCollector(config);

        // Act - Record more than max samples
        for (int i = 0; i < 10; i++)
        {
            collector.RecordIteration(CreateIterationResult(iterationNumber: i));
        }

        // Assert
        var stats = collector.GetIterationStatistics();
        Assert.Equal(5, stats.TotalIterations); // Limited to max samples
    }

    #endregion
}
