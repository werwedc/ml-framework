namespace MLFramework.Data;

/// <summary>
/// Manages worker recovery operations including restart attempts and failure tracking.
/// Provides mechanisms for automatic recovery from worker failures.
/// </summary>
public sealed class WorkerRecoveryService
{
    private readonly Dictionary<int, int> _retryCounts;
    private readonly HashSet<int> _failedWorkers;
    private readonly ErrorPolicy _errorPolicy;
    private readonly int _maxRetries;
    private readonly IErrorLogger? _logger;
    private readonly object _lock = new object();

    /// <summary>
    /// Event raised when a worker is successfully restarted.
    /// The first parameter is the worker ID, the second is the retry count.
    /// </summary>
    public event Action<int, int>? OnWorkerRestarted;

    /// <summary>
    /// Event raised when a worker is marked as permanently failed.
    /// </summary>
    public event Action<int>? OnWorkerFailed;

    /// <summary>
    /// Gets the error policy being used by the recovery service.
    /// </summary>
    public ErrorPolicy ErrorPolicy => _errorPolicy;

    /// <summary>
    /// Gets the maximum number of retry attempts per worker.
    /// </summary>
    public int MaxRetries => _maxRetries;

    /// <summary>
    /// Initializes a new instance of the WorkerRecoveryService class.
    /// </summary>
    /// <param name="errorPolicy">The error policy to determine recovery behavior.</param>
    /// <param name="maxRetries">Maximum number of retry attempts per worker (default: 3).</param>
    /// <param name="logger">Optional logger for recording recovery events.</param>
    public WorkerRecoveryService(
        ErrorPolicy errorPolicy,
        int maxRetries = 3,
        IErrorLogger? logger = null)
    {
        if (maxRetries < 0)
            throw new ArgumentOutOfRangeException(nameof(maxRetries), maxRetries, "MaxRetries must be >= 0.");

        _errorPolicy = errorPolicy;
        _maxRetries = maxRetries;
        _logger = logger;
        _retryCounts = new Dictionary<int, int>();
        _failedWorkers = new HashSet<int>();
    }

    /// <summary>
    /// Attempts to restart a failed worker.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    /// <param name="error">The error that caused the worker to fail.</param>
    /// <param name="restartCallback">Optional callback function that will be called to restart the worker.</param>
    /// <returns>True if the restart was attempted/successful, false if the worker should not be restarted.</returns>
    public async Task<bool> TryRestartWorkerAsync(
        int workerId,
        WorkerError error,
        Func<int, Task>? restartCallback = null)
    {
        if (error == null)
            throw new ArgumentNullException(nameof(error));

        lock (_lock)
        {
            // Check if worker is already marked as failed
            if (_failedWorkers.Contains(workerId))
            {
                _logger?.LogWarning($"Worker {workerId} is already marked as failed, skipping restart attempt");
                return false;
            }

            // Check if restart is allowed based on error policy
            if (_errorPolicy != ErrorPolicy.Restart)
            {
                _logger?.LogInfo($"Restart not allowed for worker {workerId} with error policy: {_errorPolicy}");
                return false;
            }

            // Get current retry count
            int retryCount = _retryCounts.TryGetValue(workerId, out var count) ? count : 0;

            // Check if retry limit has been exceeded
            if (retryCount >= _maxRetries)
            {
                _logger?.LogWarning($"Worker {workerId} has exceeded max retries ({_maxRetries}), marking as failed");
                MarkWorkerFailed(workerId);
                return false;
            }

            // Increment retry count
            _retryCounts[workerId] = retryCount + 1;
        }

        _logger?.LogInfo($"Attempting to restart worker {workerId} (retry {_retryCounts[workerId]}/{_maxRetries})");

        try
        {
            // Call the restart callback if provided
            if (restartCallback != null)
            {
                await restartCallback(workerId);
            }

            // Calculate backoff delay with exponential increase
            int retryCount = _retryCounts[workerId];
            int delayMs = 100 * retryCount; // 100ms, 200ms, 300ms, etc.

            if (delayMs > 0)
            {
                await Task.Delay(delayMs);
            }

            _logger?.LogInfo($"Worker {workerId} restart completed successfully");
            OnWorkerRestarted?.Invoke(workerId, retryCount);
            return true;
        }
        catch (Exception ex)
        {
            _logger?.LogError(new WorkerError(workerId, ex, "WorkerRestart"));

            // If restart fails, mark worker as failed if we've used all retries
            lock (_lock)
            {
                int retryCount = _retryCounts[workerId];
                if (retryCount >= _maxRetries)
                {
                    MarkWorkerFailed(workerId);
                }
            }

            return false;
        }
    }

    /// <summary>
    /// Marks a worker as permanently failed.
    /// No further restart attempts will be made for this worker.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    public void MarkWorkerFailed(int workerId)
    {
        lock (_lock)
        {
            if (_failedWorkers.Add(workerId))
            {
                _logger?.LogWarning($"Worker {workerId} marked as permanently failed");
                OnWorkerFailed?.Invoke(workerId);
            }
        }
    }

    /// <summary>
    /// Gets the current retry count for a worker.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    /// <returns>The number of retry attempts made for this worker.</returns>
    public int GetRetryCount(int workerId)
    {
        lock (_lock)
        {
            return _retryCounts.TryGetValue(workerId, out var count) ? count : 0;
        }
    }

    /// <summary>
    /// Gets the remaining retry attempts for a worker.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    /// <returns>The number of remaining retry attempts, or -1 if the worker is marked as failed.</returns>
    public int GetRemainingRetries(int workerId)
    {
        lock (_lock)
        {
            if (_failedWorkers.Contains(workerId))
                return -1;

            int retryCount = _retryCounts.TryGetValue(workerId, out var count) ? count : 0;
            return _maxRetries - retryCount;
        }
    }

    /// <summary>
    /// Checks if a worker is marked as permanently failed.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    /// <returns>True if the worker is marked as failed, false otherwise.</returns>
    public bool IsWorkerFailed(int workerId)
    {
        lock (_lock)
        {
            return _failedWorkers.Contains(workerId);
        }
    }

    /// <summary>
    /// Checks if a worker can be restarted based on current state.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    /// <returns>True if the worker can be restarted, false otherwise.</returns>
    public bool CanRestart(int workerId)
    {
        lock (_lock)
        {
            if (_failedWorkers.Contains(workerId))
                return false;

            if (_errorPolicy != ErrorPolicy.Restart)
                return false;

            int retryCount = _retryCounts.TryGetValue(workerId, out var count) ? count : 0;
            return retryCount < _maxRetries;
        }
    }

    /// <summary>
    /// Gets the number of workers marked as failed.
    /// </summary>
    public int GetFailedWorkerCount()
    {
        lock (_lock)
        {
            return _failedWorkers.Count;
        }
    }

    /// <summary>
    /// Gets all worker IDs marked as failed.
    /// </summary>
    /// <returns>A read-only list of failed worker IDs.</returns>
    public IReadOnlyList<int> GetFailedWorkers()
    {
        lock (_lock)
        {
            return _failedWorkers.ToList().AsReadOnly();
        }
    }

    /// <summary>
    /// Resets the recovery state for a specific worker.
    /// Allows the worker to be restarted again as if it never failed.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    public void ResetWorker(int workerId)
    {
        lock (_lock)
        {
            _retryCounts.Remove(workerId);
            _failedWorkers.Remove(workerId);

            _logger?.LogInfo($"Reset recovery state for worker {workerId}");
        }
    }

    /// <summary>
    /// Resets the recovery state for all workers.
    /// </summary>
    public void ResetAll()
    {
        lock (_lock)
        {
            _retryCounts.Clear();
            _failedWorkers.Clear();

            _logger?.LogInfo("Reset recovery state for all workers");
        }
    }

    /// <summary>
    /// Gets a summary of the recovery service state.
    /// </summary>
    /// <returns>A string containing recovery statistics.</returns>
    public string GetRecoverySummary()
    {
        lock (_lock)
        {
            var summary = new System.Text.StringBuilder();
            summary.AppendLine($"Error Policy: {_errorPolicy}");
            summary.AppendLine($"Max Retries: {_maxRetries}");
            summary.AppendLine($"Failed Workers: {_failedWorkers.Count}");

            if (_retryCounts.Count > 0)
            {
                summary.AppendLine("\nRetry Counts:");
                foreach (var kvp in _retryCounts)
                {
                    summary.AppendLine($"  Worker {kvp.Key}: {kvp.Value} attempt(s)");
                }
            }

            return summary.ToString();
        }
    }
}
