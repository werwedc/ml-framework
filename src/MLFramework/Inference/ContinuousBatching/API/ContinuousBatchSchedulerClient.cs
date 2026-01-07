using System.Diagnostics;
using Microsoft.Extensions.Logging;

namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Client API for the continuous batch scheduler.
/// </summary>
public class ContinuousBatchSchedulerClient : IDisposable
{
    private readonly ContinuousBatchScheduler _scheduler;
    private readonly SchedulerApiClientConfiguration _config;
    private readonly ILogger _logger;

    /// <summary>
    /// Creates a new scheduler client.
    /// </summary>
    public ContinuousBatchSchedulerClient(
        ContinuousBatchScheduler scheduler,
        SchedulerApiClientConfiguration config,
        ILogger logger)
    {
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Enqueues a text generation request.
    /// </summary>
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

    /// <summary>
    /// Enqueues a request with full control.
    /// </summary>
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

    /// <summary>
    /// Enqueues multiple requests concurrently.
    /// </summary>
    public async Task<List<GenerationResult>> GenerateBatchAsync(
        List<GenerationRequest> requests,
        CancellationToken cancellationToken = default)
    {
        if (requests == null || requests.Count == 0)
            return new List<GenerationResult>();

        if (_config.EnableRequestLogging)
        {
            _logger.LogInformation("Enqueueing batch of {Count} requests", requests.Count);
        }

        // Create tasks for all requests
        var tasks = requests.Select(async req =>
        {
            var requestId = RequestId.New();
            var stopwatch = Stopwatch.StartNew();

            try
            {
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

                // Attach options if provided
                if (req.Options != null)
                {
                    if (request.Metadata == null)
                        request.Metadata = new Dictionary<string, object>();
                    request.Metadata["options"] = req.Options;
                }

                // Enqueue and wait for completion
                stopwatch.Restart();
                var result = await _scheduler.EnqueueAsync(request, req.Priority);
                stopwatch.Stop();

                // Decode generated tokens from result
                var generatedText = result;

                return new GenerationResult(
                    requestId,
                    generatedText,
                    request.GeneratedTokens,
                    CompletionReason.EosTokenReached, // Simplified
                    stopwatch.Elapsed,
                    TimeSpan.Zero, // Simplified
                    req.Metadata
                );
            }
            catch (Exception ex)
            {
                stopwatch.Stop();
                _logger.LogError(ex, "Batch request failed for {RequestId}", requestId);

                // Return error result
                return new GenerationResult(
                    requestId,
                    string.Empty,
                    0,
                    CompletionReason.Cancelled,
                    stopwatch.Elapsed,
                    TimeSpan.Zero,
                    req.Metadata
                );
            }
        }).ToArray();

        await Task.WhenAll(tasks);

        return tasks.Select(t => t.Result).ToList();
    }

    /// <summary>
    /// Gets scheduler status.
    /// </summary>
    public SchedulerStatus GetStatus()
    {
        // Note: GPU utilization and completed requests would require additional interfaces
        // For now, returning placeholder values for those fields
        return new SchedulerStatus(
            IsRunning: _scheduler.IsRunning,
            ActiveRequests: _scheduler.ActiveRequestCount,
            QueuedRequests: _scheduler.QueuedRequestCount,
            CompletedRequests: 0, // Would need to track
            GpuUtilization: 0.0, // Would need GPU manager
            MemoryUtilization: 0.0, // Would need capacity manager access
            LastUpdateTime: DateTime.UtcNow
        );
    }

    /// <summary>
    /// Gets detailed statistics.
    /// </summary>
    public SchedulerStatistics GetStatistics()
    {
        // Note: This would be implemented with a statistics collector
        // For now, returning placeholder values
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

    /// <summary>
    /// Cancels a specific request.
    /// </summary>
    public bool CancelRequest(RequestId requestId)
    {
        if (_config.EnableRequestLogging)
        {
            _logger.LogInformation("Cancelling request: {RequestId}", requestId);
        }

        // Placeholder implementation
        return true;
    }

    /// <summary>
    /// Cancels all pending requests.
    /// </summary>
    public int CancelAllRequests()
    {
        if (_config.EnableRequestLogging)
        {
            _logger.LogInformation("Cancelling all pending requests");
        }

        // Placeholder implementation
        return 0;
    }

    /// <summary>
    /// Gets estimated wait time for a new request.
    /// </summary>
    public TimeSpan? EstimateWaitTime(Priority priority = Priority.Normal)
    {
        // Placeholder implementation
        return TimeSpan.FromSeconds(1.0);
    }

    /// <summary>
    /// Disposes of the client.
    /// </summary>
    public void Dispose()
    {
        // No resources to dispose
    }
}
