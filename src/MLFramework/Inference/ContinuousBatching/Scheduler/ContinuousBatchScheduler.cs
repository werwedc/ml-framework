using System.Diagnostics;

namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Core continuous batch scheduler orchestrator.
/// </summary>
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

    /// <summary>
    /// Gets whether the scheduler is currently running.
    /// </summary>
    public bool IsRunning { get; private set; }

    /// <summary>
    /// Gets the number of active requests.
    /// </summary>
    public int ActiveRequestCount => _batchManager.ActiveRequestCount;

    /// <summary>
    /// Creates a new continuous batch scheduler.
    /// </summary>
    public ContinuousBatchScheduler(
        RequestQueue requestQueue,
        BatchManager batchManager,
        CompletionDetector completionDetector,
        CapacityManager capacityManager,
        IModelExecutor modelExecutor,
        ISchedulerMetrics metrics,
        SchedulerConfiguration config)
    {
        _requestQueue = requestQueue ?? throw new ArgumentNullException(nameof(requestQueue));
        _batchManager = batchManager ?? throw new ArgumentNullException(nameof(batchManager));
        _completionDetector = completionDetector ?? throw new ArgumentNullException(nameof(completionDetector));
        _capacityManager = capacityManager ?? throw new ArgumentNullException(nameof(capacityManager));
        _modelExecutor = modelExecutor ?? throw new ArgumentNullException(nameof(modelExecutor));
        _metrics = metrics ?? throw new ArgumentNullException(nameof(metrics));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _shutdownTokenSource = new CancellationTokenSource();
        _iterationCount = 0;
        IsRunning = false;
    }

    /// <summary>
    /// Starts the scheduler loop.
    /// </summary>
    public void Start()
    {
        if (IsRunning)
            throw new InvalidOperationException("Scheduler is already running");

        IsRunning = true;
        _schedulerTask = Task.Run(async () => await SchedulerLoopAsync());
    }

    /// <summary>
    /// Stops the scheduler gracefully.
    /// </summary>
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
            }
        }
        else
        {
            await stopTask;
        }

        IsRunning = false;
    }

    /// <summary>
    /// Enqueues a new request.
    /// </summary>
    public Task<string> EnqueueAsync(Request request, Priority priority = Priority.Normal)
    {
        _requestQueue.Enqueue(request, priority);
        return request.CompletionSource.Task;
    }

    /// <summary>
    /// Executes a single iteration (for testing).
    /// </summary>
    public async Task<IterationResult> ExecuteIterationAsync(
        CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();

        // Prepare batch for this iteration
        var batch = _batchManager.PrepareNextIteration();

        if (batch.Size == 0)
        {
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

        return ProcessBatch(batch, output, stopwatch);
    }

    /// <summary>
    /// Main scheduler loop.
    /// </summary>
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

                // Increment iteration count after successful execution
                _iterationCount++;

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
                await Task.Delay(100, _shutdownTokenSource.Token);
            }
        }

        IsRunning = false;
    }

    /// <summary>
    /// Processes batch output and updates request state.
    /// </summary>
    private IterationResult ProcessBatch(
        Batch batch,
        ModelOutput output,
        Stopwatch stopwatch)
    {
        int totalTokensGenerated = 0;
        int completedRequests = 0;
        List<RequestId> completedIds = new List<RequestId>();

        foreach (var request in batch.Requests)
        {
            if (request.GeneratedTokenIds.Count > 0)
            {
                // Token was already generated in mock executor
                int tokenId = request.GeneratedTokenIds[^1];
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

        stopwatch.Stop();

        return new IterationResult(
            _iterationCount,
            batch.Size,
            totalTokensGenerated,
            completedRequests,
            stopwatch.Elapsed,
            batch.EstimatedMemoryBytes
        );
    }

    /// <summary>
    /// Decodes token IDs to text (placeholder).
    /// </summary>
    private string DecodeTokens(List<int> tokenIds)
    {
        // Placeholder implementation
        return $"[{string.Join(", ", tokenIds)}]";
    }

    /// <summary>
    /// Disposes of the scheduler and releases resources.
    /// </summary>
    public void Dispose()
    {
        _shutdownTokenSource.Cancel();
        _shutdownTokenSource.Dispose();
        _schedulerTask?.Dispose();
    }
}
