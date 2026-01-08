using Xunit;
using MLFramework.Data;

namespace MLFramework.Tests.Data;

/// <summary>
/// Unit tests for error handling components including WorkerError, ErrorAggregator, and related classes.
/// </summary>
public class ErrorHandlingTests
{
    #region WorkerError Tests

    [Fact]
    public void WorkerError_Constructor_ValidParameters_CreatesInstance()
    {
        // Arrange
        var exception = new InvalidOperationException("Test error");
        int workerId = 1;
        string context = "TestContext";

        // Act
        var error = new WorkerError(workerId, exception, context);

        // Assert
        Assert.Equal(workerId, error.WorkerId);
        Assert.Equal(exception, error.Exception);
        Assert.Equal(context, error.Context);
        Assert.True((DateTime.UtcNow - error.Timestamp) < TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void WorkerError_Constructor_NullException_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new WorkerError(1, null!));
    }

    [Fact]
    public void WorkerError_ToString_ReturnsFormattedString()
    {
        // Arrange
        var exception = new InvalidOperationException("Test error");
        var error = new WorkerError(1, exception, "TestContext");

        // Act
        var result = error.ToString();

        // Assert
        Assert.Contains("WorkerError", result);
        Assert.Contains("WorkerId: 1", result);
        Assert.Contains("InvalidOperationException", result);
        Assert.Contains("Test error", result);
        Assert.Contains("TestContext", result);
    }

    [Fact]
    public void WorkerError_GetStackTrace_ReturnsStackTrace()
    {
        // Arrange
        try
        {
            throw new InvalidOperationException("Test error");
        }
        catch (InvalidOperationException ex)
        {
            var error = new WorkerError(1, ex);

            // Act
            var stackTrace = error.GetStackTrace();

            // Assert
            Assert.NotNull(stackTrace);
            Assert.NotEmpty(stackTrace);
        }
    }

    [Fact]
    public void WorkerError_GetFormattedMessage_ReturnsFormattedMessage()
    {
        // Arrange
        var exception = new InvalidOperationException("Test error");
        var error = new WorkerError(1, exception, "TestContext");

        // Act
        var message = error.GetFormattedMessage();

        // Assert
        Assert.Contains("Worker 1", message);
        Assert.Contains("[TestContext]", message);
        Assert.Contains("InvalidOperationException", message);
        Assert.Contains("Test error", message);
    }

    #endregion

    #region ErrorAggregator Tests

    [Fact]
    public void ErrorAggregator_Constructor_CreatesEmptyAggregator()
    {
        // Act
        var aggregator = new ErrorAggregator();

        // Assert
        Assert.Equal(0, aggregator.TotalErrors);
        Assert.False(aggregator.HasErrors());
    }

    [Fact]
    public void ErrorAggregator_AddError_ValidError_IncreasesCount()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        var error = new WorkerError(1, new InvalidOperationException("Test error"));

        // Act
        aggregator.AddError(error);

        // Assert
        Assert.Equal(1, aggregator.TotalErrors);
        Assert.True(aggregator.HasErrors());
    }

    [Fact]
    public void ErrorAggregator_AddError_NullError_ThrowsArgumentNullException()
    {
        // Arrange
        var aggregator = new ErrorAggregator();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => aggregator.AddError(null!));
    }

    [Fact]
    public void ErrorAggregator_GetErrors_ReturnsAllErrors()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        var error1 = new WorkerError(1, new InvalidOperationException("Error 1"));
        var error2 = new WorkerError(2, new InvalidOperationException("Error 2"));

        aggregator.AddError(error1);
        aggregator.AddError(error2);

        // Act
        var errors = aggregator.GetErrors();

        // Assert
        Assert.Equal(2, errors.Count);
        Assert.Equal(error1, errors[0]);
        Assert.Equal(error2, errors[1]);
    }

    [Fact]
    public void ErrorAggregator_GetLastError_ReturnsLastAddedError()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        var error1 = new WorkerError(1, new InvalidOperationException("Error 1"));
        var error2 = new WorkerError(2, new InvalidOperationException("Error 2"));

        aggregator.AddError(error1);
        aggregator.AddError(error2);

        // Act
        var lastError = aggregator.GetLastError();

        // Assert
        Assert.NotNull(lastError);
        Assert.Equal(error2, lastError);
    }

    [Fact]
    public void ErrorAggregator_GetErrorCount_ReturnsCorrectCount()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        var error1 = new WorkerError(1, new InvalidOperationException("Error 1"));
        var error2 = new WorkerError(1, new InvalidOperationException("Error 2"));
        var error3 = new WorkerError(2, new InvalidOperationException("Error 3"));

        aggregator.AddError(error1);
        aggregator.AddError(error2);
        aggregator.AddError(error3);

        // Act
        var count1 = aggregator.GetErrorCount(1);
        var count2 = aggregator.GetErrorCount(2);
        var count3 = aggregator.GetErrorCount(3);

        // Assert
        Assert.Equal(2, count1);
        Assert.Equal(1, count2);
        Assert.Equal(0, count3);
    }

    [Fact]
    public void ErrorAggregator_GetErrorsByWorker_ReturnsCorrectErrors()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        var error1 = new WorkerError(1, new InvalidOperationException("Error 1"));
        var error2 = new WorkerError(1, new InvalidOperationException("Error 2"));
        var error3 = new WorkerError(2, new InvalidOperationException("Error 3"));

        aggregator.AddError(error1);
        aggregator.AddError(error2);
        aggregator.AddError(error3);

        // Act
        var worker1Errors = aggregator.GetErrorsByWorker(1);
        var worker2Errors = aggregator.GetErrorsByWorker(2);

        // Assert
        Assert.Equal(2, worker1Errors.Count);
        Assert.Equal(1, worker2Errors.Count);
        Assert.Equal(error1, worker1Errors[0]);
        Assert.Equal(error2, worker1Errors[1]);
        Assert.Equal(error3, worker2Errors[0]);
    }

    [Fact]
    public void ErrorAggregator_ClearErrors_RemovesAllErrors()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        aggregator.AddError(new WorkerError(1, new InvalidOperationException("Error 1")));

        // Act
        aggregator.ClearErrors();

        // Assert
        Assert.Equal(0, aggregator.TotalErrors);
        Assert.False(aggregator.HasErrors());
    }

    [Fact]
    public void ErrorAggregator_GetUniqueWorkerCount_ReturnsCorrectCount()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        aggregator.AddError(new WorkerError(1, new InvalidOperationException("Error 1")));
        aggregator.AddError(new WorkerError(1, new InvalidOperationException("Error 2")));
        aggregator.AddError(new WorkerError(2, new InvalidOperationException("Error 3")));

        // Act
        var uniqueCount = aggregator.GetUniqueWorkerCount();

        // Assert
        Assert.Equal(2, uniqueCount);
    }

    [Fact]
    public void ErrorAggregator_GetErrorSummaryByWorker_ReturnsCorrectSummary()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        aggregator.AddError(new WorkerError(1, new InvalidOperationException("Error 1")));
        aggregator.AddError(new WorkerError(1, new InvalidOperationException("Error 2")));
        aggregator.AddError(new WorkerError(2, new InvalidOperationException("Error 3")));

        // Act
        var summary = aggregator.GetErrorSummaryByWorker();

        // Assert
        Assert.Equal(2, summary.Count);
        Assert.Equal(2, summary[1]);
        Assert.Equal(1, summary[2]);
    }

    [Fact]
    public void ErrorAggregator_GetErrorSummaryByExceptionType_ReturnsCorrectSummary()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        aggregator.AddError(new WorkerError(1, new InvalidOperationException("Error 1")));
        aggregator.AddError(new WorkerError(2, new InvalidOperationException("Error 2")));
        aggregator.AddError(new WorkerError(3, new ArgumentException("Error 3")));

        // Act
        var summary = aggregator.GetErrorSummaryByExceptionType();

        // Assert
        Assert.Equal(2, summary.Count);
        Assert.Equal(2, summary["InvalidOperationException"]);
        Assert.Equal(1, summary["ArgumentException"]);
    }

    [Fact]
    public void ErrorAggregator_GetErrorSummary_ReturnsFormattedSummary()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        aggregator.AddError(new WorkerError(1, new InvalidOperationException("Error 1")));

        // Act
        var summary = aggregator.GetErrorSummary();

        // Assert
        Assert.Contains("Total Errors: 1", summary);
        Assert.Contains("Unique Workers with Errors: 1");
    }

    [Fact]
    public void ErrorAggregator_ThreadSafety_MultipleThreadsCanAddErrors()
    {
        // Arrange
        var aggregator = new ErrorAggregator();
        var tasks = new Task[10];
        var errorsPerTask = 100;

        // Act
        for (int i = 0; i < tasks.Length; i++)
        {
            int taskId = i;
            tasks[i] = Task.Run(() =>
            {
                for (int j = 0; j < errorsPerTask; j++)
                {
                    aggregator.AddError(new WorkerError(taskId, new InvalidOperationException($"Error {j}")));
                }
            });
        }

        Task.WaitAll(tasks);

        // Assert
        Assert.Equal(10 * errorsPerTask, aggregator.TotalErrors);
    }

    #endregion

    #region ConsoleErrorLogger Tests

    [Fact]
    public void ConsoleErrorLogger_LogError_LogsErrorToConsole()
    {
        // Arrange
        var logger = new ConsoleErrorLogger(useColors: false);
        var error = new WorkerError(1, new InvalidOperationException("Test error"), "TestContext");

        // Act & Assert - should not throw
        logger.LogError(error);
    }

    [Fact]
    public void ConsoleErrorLogger_LogWarning_LogsWarningToConsole()
    {
        // Arrange
        var logger = new ConsoleErrorLogger(useColors: false);

        // Act & Assert - should not throw
        logger.LogWarning("Test warning");
    }

    [Fact]
    public void ConsoleErrorLogger_LogInfo_LogsInfoToConsole()
    {
        // Arrange
        var logger = new ConsoleErrorLogger(useColors: false);

        // Act & Assert - should not throw
        logger.LogInfo("Test info");
    }

    [Fact]
    public void ConsoleErrorLogger_LogDebug_LogsDebugToConsole()
    {
        // Arrange
        var logger = new ConsoleErrorLogger(useColors: false);

        // Act & Assert - should not throw
        logger.LogDebug("Test debug");
    }

    #endregion

    #region WorkerRecoveryService Tests

    [Fact]
    public void WorkerRecoveryService_Constructor_ValidParameters_CreatesInstance()
    {
        // Act
        var service = new WorkerRecoveryService(ErrorPolicy.Restart, maxRetries: 3);

        // Assert
        Assert.Equal(ErrorPolicy.Restart, service.ErrorPolicy);
        Assert.Equal(3, service.MaxRetries);
    }

    [Fact]
    public void WorkerRecoveryService_Constructor_NegativeMaxRetries_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WorkerRecoveryService(ErrorPolicy.Restart, maxRetries: -1));
    }

    [Fact]
    public void WorkerRecoveryService_TryRestartWorker_PolicyNotRestart_ReturnsFalse()
    {
        // Arrange
        var service = new WorkerRecoveryService(ErrorPolicy.FailFast);
        var error = new WorkerError(1, new InvalidOperationException("Test error"));

        // Act
        var result = service.TryRestartWorkerAsync(1, error).Result;

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void WorkerRecoveryService_TryRestartWorker_WithinRetryLimit_ReturnsTrue()
    {
        // Arrange
        var service = new WorkerRecoveryService(ErrorPolicy.Restart, maxRetries: 3);
        var error = new WorkerError(1, new InvalidOperationException("Test error"));

        // Act
        var result = service.TryRestartWorkerAsync(1, error).Result;

        // Assert
        Assert.True(result);
        Assert.Equal(1, service.GetRetryCount(1));
    }

    [Fact]
    public void WorkerRecoveryService_TryRestartWorker_ExceedsRetryLimit_ReturnsFalse()
    {
        // Arrange
        var service = new WorkerRecoveryService(ErrorPolicy.Restart, maxRetries: 2);
        var error = new WorkerError(1, new InvalidOperationException("Test error"));

        // Act
        var result1 = service.TryRestartWorkerAsync(1, error).Result;
        var result2 = service.TryRestartWorkerAsync(1, error).Result;
        var result3 = service.TryRestartWorkerAsync(1, error).Result;

        // Assert
        Assert.True(result1);
        Assert.True(result2);
        Assert.False(result3);
        Assert.True(service.IsWorkerFailed(1));
    }

    [Fact]
    public void WorkerRecoveryService_MarkWorkerFailed_PreventsFutureRestarts()
    {
        // Arrange
        var service = new WorkerRecoveryService(ErrorPolicy.Restart, maxRetries: 3);
        service.MarkWorkerFailed(1);
        var error = new WorkerError(1, new InvalidOperationException("Test error"));

        // Act
        var result = service.TryRestartWorkerAsync(1, error).Result;

        // Assert
        Assert.False(result);
        Assert.True(service.IsWorkerFailed(1));
    }

    [Fact]
    public void WorkerRecoveryService_GetRemainingRetries_ReturnsCorrectValue()
    {
        // Arrange
        var service = new WorkerRecoveryService(ErrorPolicy.Restart, maxRetries: 5);
        var error = new WorkerError(1, new InvalidOperationException("Test error"));

        // Act
        service.TryRestartWorkerAsync(1, error).Result;
        service.TryRestartWorkerAsync(1, error).Result;

        // Assert
        Assert.Equal(3, service.GetRemainingRetries(1));
    }

    [Fact]
    public void WorkerRecoveryService_ResetWorker_ClearsRetryCount()
    {
        // Arrange
        var service = new WorkerRecoveryService(ErrorPolicy.Restart, maxRetries: 3);
        var error = new WorkerError(1, new InvalidOperationException("Test error"));

        service.TryRestartWorkerAsync(1, error).Result;

        // Act
        service.ResetWorker(1);
        var remaining = service.GetRemainingRetries(1);

        // Assert
        Assert.Equal(3, remaining);
    }

    [Fact]
    public void WorkerRecoveryService_CanRestart_ReturnsCorrectValue()
    {
        // Arrange
        var service = new WorkerRecoveryService(ErrorPolicy.Restart, maxRetries: 2);

        // Act & Assert
        Assert.True(service.CanRestart(1));

        var error = new WorkerError(1, new InvalidOperationException("Test error"));
        service.TryRestartWorkerAsync(1, error).Result;
        service.TryRestartWorkerAsync(1, error).Result;

        Assert.False(service.CanRestart(1));
    }

    #endregion

    #region WorkerTimeoutTracker Tests

    [Fact]
    public void WorkerTimeoutTracker_Constructor_ValidParameters_CreatesInstance()
    {
        // Act
        var tracker = new WorkerTimeoutTracker(TimeSpan.FromSeconds(30));

        // Assert
        Assert.Equal(TimeSpan.FromSeconds(30), tracker.Timeout);
    }

    [Fact]
    public void WorkerTimeoutTracker_Constructor_InvalidTimeout_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WorkerTimeoutTracker(TimeSpan.Zero));
    }

    [Fact]
    public void WorkerTimeoutTracker_StartOperation_StartsTracking()
    {
        // Arrange
        var tracker = new WorkerTimeoutTracker(TimeSpan.FromSeconds(30));

        // Act
        tracker.StartOperation(1);

        // Assert
        Assert.True(tracker.IsTrackingOperation(1));
        Assert.Equal(1, tracker.GetActiveWorkerCount());
    }

    [Fact]
    public void WorkerTimeoutTracker_EndOperation_StopsTracking()
    {
        // Arrange
        var tracker = new WorkerTimeoutTracker(TimeSpan.FromSeconds(30));
        tracker.StartOperation(1);

        // Act
        tracker.EndOperation(1);

        // Assert
        Assert.False(tracker.IsTrackingOperation(1));
    }

    [Fact]
    public void WorkerTimeoutTracker_GetElapsedTime_ReturnsCorrectValue()
    {
        // Arrange
        var tracker = new WorkerTimeoutTracker(TimeSpan.FromSeconds(30));
        tracker.StartOperation(1);

        // Act
        Thread.Sleep(100);
        var elapsed = tracker.GetElapsedTime(1);

        // Assert
        Assert.NotNull(elapsed);
        Assert.True(elapsed.Value >= TimeSpan.FromMilliseconds(100));
    }

    [Fact]
    public void WorkerTimeoutTracker_ClearAll_RemovesAllTrackedOperations()
    {
        // Arrange
        var tracker = new WorkerTimeoutTracker(TimeSpan.FromSeconds(30));
        tracker.StartOperation(1);
        tracker.StartOperation(2);

        // Act
        tracker.ClearAll();

        // Assert
        Assert.Equal(0, tracker.GetActiveWorkerCount());
    }

    [Fact]
    public void WorkerTimeoutTracker_Dispose_StopsMonitoring()
    {
        // Arrange
        var tracker = new WorkerTimeoutTracker(TimeSpan.FromSeconds(30));
        tracker.StartMonitoring();

        // Act
        tracker.Dispose();
        tracker.StartOperation(1);

        // Assert - should not throw
        Assert.True(true);
    }

    #endregion

    #region WorkerCrashDetector Tests

    [Fact]
    public void WorkerCrashDetector_Constructor_ValidParameters_CreatesInstance()
    {
        // Act
        var detector = new WorkerCrashDetector(TimeSpan.FromSeconds(30));

        // Assert
        Assert.Equal(TimeSpan.FromSeconds(30), detector.HeartbeatTimeout);
    }

    [Fact]
    public void WorkerCrashDetector_RegisterWorker_RegistersWorkerForMonitoring()
    {
        // Arrange
        var detector = new WorkerCrashDetector(TimeSpan.FromSeconds(30));
        var task = Task.Run(() => Task.Delay(1000));

        // Act
        detector.RegisterWorker(1, task);

        // Assert
        Assert.Equal(1, detector.GetRegisteredWorkerCount());
    }

    [Fact]
    public void WorkerCrashDetector_RegisterWorker_NullTask_ThrowsException()
    {
        // Arrange
        var detector = new WorkerCrashDetector(TimeSpan.FromSeconds(30));

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => detector.RegisterWorker(1, null!));
    }

    [Fact]
    public void WorkerCrashDetector_UpdateHeartbeat_UpdatesTimestamp()
    {
        // Arrange
        var detector = new WorkerCrashDetector(TimeSpan.FromSeconds(30));
        var task = Task.Run(() => Task.Delay(1000));
        detector.RegisterWorker(1, task);

        // Act
        detector.UpdateHeartbeat(1);
        var timeSinceHeartbeat = detector.GetTimeSinceLastHeartbeat(1);

        // Assert
        Assert.NotNull(timeSinceHeartbeat);
        Assert.True(timeSinceHeartbeat.Value < TimeSpan.FromMilliseconds(100));
    }

    [Fact]
    public void WorkerCrashDetector_UnregisterWorker_RemovesWorkerFromMonitoring()
    {
        // Arrange
        var detector = new WorkerCrashDetector(TimeSpan.FromSeconds(30));
        var task = Task.Run(() => Task.Delay(1000));
        detector.RegisterWorker(1, task);

        // Act
        detector.UnregisterWorker(1);

        // Assert
        Assert.Equal(0, detector.GetRegisteredWorkerCount());
    }

    [Fact]
    public void WorkerCrashDetector_GetTimeSinceLastHeartbeat_NotRegistered_ReturnsNull()
    {
        // Arrange
        var detector = new WorkerCrashDetector(TimeSpan.FromSeconds(30));

        // Act
        var result = detector.GetTimeSinceLastHeartbeat(1);

        // Assert
        Assert.Null(result);
    }

    #endregion

    #region DataLoaderConfig Error Handling Tests

    [Fact]
    public void DataLoaderConfig_Constructor_WithErrorHandlingParameters_CreatesInstance()
    {
        // Act
        var config = new DataLoaderConfig(
            numWorkers: 4,
            batchSize: 32,
            prefetchCount: 2,
            queueSize: 10,
            shuffle: true,
            seed: 42,
            pinMemory: true,
            errorPolicy: ErrorPolicy.Restart,
            maxWorkerRetries: 3,
            workerTimeout: TimeSpan.FromSeconds(30),
            logErrors: true);

        // Assert
        Assert.Equal(ErrorPolicy.Restart, config.ErrorPolicy);
        Assert.Equal(3, config.MaxWorkerRetries);
        Assert.Equal(TimeSpan.FromSeconds(30), config.WorkerTimeout);
        Assert.True(config.LogErrors);
    }

    [Fact]
    public void DataLoaderConfig_WithErrorPolicy_CreatesCorrectConfig()
    {
        // Arrange
        var baseConfig = new DataLoaderConfig();

        // Act
        var newConfig = baseConfig.WithErrorPolicy(ErrorPolicy.FailFast);

        // Assert
        Assert.Equal(ErrorPolicy.FailFast, newConfig.ErrorPolicy);
        Assert.Equal(baseConfig.NumWorkers, newConfig.NumWorkers);
    }

    [Fact]
    public void DataLoaderConfig_WithMaxWorkerRetries_CreatesCorrectConfig()
    {
        // Arrange
        var baseConfig = new DataLoaderConfig();

        // Act
        var newConfig = baseConfig.WithMaxWorkerRetries(5);

        // Assert
        Assert.Equal(5, newConfig.MaxWorkerRetries);
    }

    [Fact]
    public void DataLoaderConfig_WithWorkerTimeout_CreatesCorrectConfig()
    {
        // Arrange
        var baseConfig = new DataLoaderConfig();

        // Act
        var newConfig = baseConfig.WithWorkerTimeout(TimeSpan.FromSeconds(60));

        // Assert
        Assert.Equal(TimeSpan.FromSeconds(60), newConfig.WorkerTimeout);
    }

    [Fact]
    public void DataLoaderConfig_WithLogErrors_CreatesCorrectConfig()
    {
        // Arrange
        var baseConfig = new DataLoaderConfig();

        // Act
        var newConfig = baseConfig.WithLogErrors(false);

        // Assert
        Assert.False(newConfig.LogErrors);
    }

    [Fact]
    public void DataLoaderConfig_Constructor_NegativeMaxRetries_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DataLoaderConfig(maxWorkerRetries: -1));
    }

    [Fact]
    public void DataLoaderConfig_Constructor_InvalidWorkerTimeout_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DataLoaderConfig(workerTimeout: TimeSpan.Zero));
    }

    [Fact]
    public void DataLoaderConfig_ToString_IncludesErrorHandlingProperties()
    {
        // Arrange
        var config = new DataLoaderConfig(
            errorPolicy: ErrorPolicy.Restart,
            maxWorkerRetries: 5,
            workerTimeout: TimeSpan.FromSeconds(45));

        // Act
        var result = config.ToString();

        // Assert
        Assert.Contains("ErrorPolicy: Restart", result);
        Assert.Contains("MaxWorkerRetries: 5", result);
        Assert.Contains("WorkerTimeout: 45", result);
    }

    [Fact]
    public void DataLoaderConfig_ForCPUBound_IncludesErrorHandlingDefaults()
    {
        // Act
        var config = DataLoaderConfigPresets.ForCPUBound();

        // Assert
        Assert.Equal(ErrorPolicy.Continue, config.ErrorPolicy);
        Assert.Equal(3, config.MaxWorkerRetries);
        Assert.Equal(TimeSpan.FromSeconds(30), config.WorkerTimeout);
    }

    [Fact]
    public void DataLoaderConfig_ForGPUBound_IncludesErrorHandlingDefaults()
    {
        // Act
        var config = DataLoaderConfigPresets.ForGPUBound();

        // Assert
        Assert.Equal(ErrorPolicy.Restart, config.ErrorPolicy);
        Assert.Equal(3, config.MaxWorkerRetries);
        Assert.Equal(TimeSpan.FromSeconds(60), config.WorkerTimeout);
    }

    #endregion
}
