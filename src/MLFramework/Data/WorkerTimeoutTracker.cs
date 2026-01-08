namespace MLFramework.Data;

/// <summary>
/// Tracks worker operation durations and detects timeouts.
/// Raises events when worker operations exceed the configured timeout threshold.
/// </summary>
public sealed class WorkerTimeoutTracker : IDisposable
{
    private readonly Dictionary<int, DateTime> _operationStartTimes;
    private readonly TimeSpan _timeout;
    private readonly TimeSpan _checkInterval;
    private readonly IErrorLogger? _logger;
    private readonly object _lock = new object();
    private readonly CancellationTokenSource _monitoringCts;
    private Task? _monitoringTask;
    private volatile bool _isDisposed;

    /// <summary>
    /// Event raised when a worker timeout is detected.
    /// The first parameter is the worker ID, the second is the elapsed time.
    /// </summary>
    public event Action<int, TimeSpan>? OnWorkerTimeout;

    /// <summary>
    /// Gets the timeout threshold for worker operations.
    /// </summary>
    public TimeSpan Timeout => _timeout;

    /// <summary>
    /// Initializes a new instance of the WorkerTimeoutTracker class.
    /// </summary>
    /// <param name="timeout">The timeout threshold for worker operations.</param>
    /// <param name="checkInterval">How often to check for timeouts (default: 1 second).</param>
    /// <param name="logger">Optional logger for recording timeout events.</param>
    public WorkerTimeoutTracker(
        TimeSpan timeout,
        TimeSpan? checkInterval = null,
        IErrorLogger? logger = null)
    {
        if (timeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(timeout), timeout, "Timeout must be > TimeSpan.Zero.");

        _timeout = timeout;
        _checkInterval = checkInterval ?? TimeSpan.FromSeconds(1);
        _logger = logger;
        _operationStartTimes = new Dictionary<int, DateTime>();
        _monitoringCts = new CancellationTokenSource();
        _isDisposed = false;
    }

    /// <summary>
    /// Starts tracking an operation for a specific worker.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    public void StartOperation(int workerId)
    {
        lock (_lock)
        {
            _operationStartTimes[workerId] = DateTime.UtcNow;

            _logger?.LogDebug($"Started operation tracking for worker {workerId}");
        }
    }

    /// <summary>
    /// Ends tracking an operation for a specific worker.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    public void EndOperation(int workerId)
    {
        lock (_lock)
        {
            if (_operationStartTimes.Remove(workerId))
            {
                _logger?.LogDebug($"Ended operation tracking for worker {workerId}");
            }
        }
    }

    /// <summary>
    /// Gets the elapsed time for a currently running operation.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    /// <returns>The elapsed time, or null if no operation is being tracked.</returns>
    public TimeSpan? GetElapsedTime(int workerId)
    {
        lock (_lock)
        {
            if (_operationStartTimes.TryGetValue(workerId, out var startTime))
            {
                return DateTime.UtcNow - startTime;
            }
            return null;
        }
    }

    /// <summary>
    /// Gets the number of workers currently being tracked.
    /// </summary>
    /// <returns>The count of workers with active operations.</returns>
    public int GetActiveWorkerCount()
    {
        lock (_lock)
        {
            return _operationStartTimes.Count;
        }
    }

    /// <summary>
    /// Checks if a worker currently has an operation being tracked.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    /// <returns>True if an operation is being tracked, false otherwise.</returns>
    public bool IsTrackingOperation(int workerId)
    {
        lock (_lock)
        {
            return _operationStartTimes.ContainsKey(workerId);
        }
    }

    /// <summary>
    /// Starts the background monitoring task.
    /// </summary>
    public void StartMonitoring()
    {
        lock (_lock)
        {
            if (_monitoringTask != null && !_monitoringTask.IsCompleted)
                return; // Already monitoring

            _monitoringTask = Task.Run(() => MonitorTimeoutsAsync(_monitoringCts.Token));
            _logger?.LogInfo("Worker timeout monitoring started");
        }
    }

    /// <summary>
    /// Stops the background monitoring task.
    /// </summary>
    public void StopMonitoring()
    {
        lock (_lock)
        {
            _monitoringCts.Cancel();
            _monitoringTask = null;
            _logger?.LogInfo("Worker timeout monitoring stopped");
        }
    }

    /// <summary>
    /// Monitoring loop that periodically checks for timeouts.
    /// </summary>
    private async Task MonitorTimeoutsAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                CheckTimeouts();
                await Task.Delay(_checkInterval, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger?.LogError(new WorkerError(-1, ex, "TimeoutTrackerMonitor"));
            }
        }
    }

    /// <summary>
    /// Checks all tracked workers for timeouts.
    /// </summary>
    private void CheckTimeouts()
    {
        List<(int workerId, TimeSpan elapsed)> timeoutWorkers = new List<(int, TimeSpan)>();

        lock (_lock)
        {
            var workerIds = _operationStartTimes.Keys.ToList();

            foreach (var workerId in workerIds)
            {
                if (!_operationStartTimes.TryGetValue(workerId, out var startTime))
                    continue;

                var elapsed = DateTime.UtcNow - startTime;
                if (elapsed > _timeout)
                {
                    timeoutWorkers.Add((workerId, elapsed));

                    // Remove the timed-out operation from tracking
                    _operationStartTimes.Remove(workerId);
                }
            }
        }

        // Raise timeout events for timed-out workers
        foreach (var (workerId, elapsed) in timeoutWorkers)
        {
            _logger?.LogWarning($"Worker {workerId} timeout detected after {elapsed.TotalSeconds:F2}s (threshold: {_timeout.TotalSeconds:F2}s)");
            OnWorkerTimeout?.Invoke(workerId, elapsed);
        }
    }

    /// <summary>
    /// Clears all tracked operations.
    /// </summary>
    public void ClearAll()
    {
        lock (_lock)
        {
            _operationStartTimes.Clear();

            _logger?.LogInfo("Cleared all operation tracking");
        }
    }

    /// <summary>
    /// Gets a summary of all currently tracked operations.
    /// </summary>
    /// <returns>A string containing tracking statistics.</returns>
    public string GetTrackingSummary()
    {
        lock (_lock)
        {
            var summary = new System.Text.StringBuilder();
            summary.AppendLine($"Timeout Threshold: {_timeout.TotalSeconds:F2}s");
            summary.AppendLine($"Active Workers: {_operationStartTimes.Count}");

            if (_operationStartTimes.Count > 0)
            {
                summary.AppendLine("\nOperation Times:");
                foreach (var kvp in _operationStartTimes)
                {
                    var elapsed = DateTime.UtcNow - kvp.Value;
                    summary.AppendLine($"  Worker {kvp.Key}: {elapsed.TotalSeconds:F2}s");
                }
            }

            return summary.ToString();
        }
    }

    /// <summary>
    /// Disposes of the timeout tracker and stops monitoring.
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
            return;

        _isDisposed = true;
        StopMonitoring();
        _monitoringCts.Dispose();

        lock (_lock)
        {
            _operationStartTimes.Clear();
        }
    }
}
