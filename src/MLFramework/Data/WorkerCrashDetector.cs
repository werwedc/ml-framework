namespace MLFramework.Data;

/// <summary>
/// Detects worker crashes by monitoring task status and heartbeats.
/// Raises events when workers crash or become unresponsive.
/// </summary>
public sealed class WorkerCrashDetector : IDisposable
{
    private readonly Dictionary<int, Task> _workerTasks;
    private readonly Dictionary<int, DateTime> _lastHeartbeat;
    private readonly TimeSpan _heartbeatTimeout;
    private readonly TimeSpan _checkInterval;
    private readonly ErrorPolicy _errorPolicy;
    private readonly IErrorLogger? _logger;
    private readonly object _lock = new object();
    private readonly CancellationTokenSource _monitoringCts;
    private Task? _monitoringTask;
    private volatile bool _isDisposed;

    /// <summary>
    /// Event raised when a worker crash is detected.
    /// </summary>
    public event Action<WorkerError>? OnWorkerCrashed;

    /// <summary>
    /// Gets the heartbeat timeout threshold.
    /// </summary>
    public TimeSpan HeartbeatTimeout => _heartbeatTimeout;

    /// <summary>
    /// Initializes a new instance of the WorkerCrashDetector class.
    /// </summary>
    /// <param name="heartbeatTimeout">The timeout for detecting stalled workers (no heartbeat).</param>
    /// <param name="checkInterval">How often to check worker status (default: 1 second).</param>
    /// <param name="errorPolicy">The error policy to determine crash behavior.</param>
    /// <param name="logger">Optional logger for recording detection events.</param>
    public WorkerCrashDetector(
        TimeSpan heartbeatTimeout,
        TimeSpan? checkInterval = null,
        ErrorPolicy errorPolicy = ErrorPolicy.Continue,
        IErrorLogger? logger = null)
    {
        _heartbeatTimeout = heartbeatTimeout;
        _checkInterval = checkInterval ?? TimeSpan.FromSeconds(1);
        _errorPolicy = errorPolicy;
        _logger = logger;
        _workerTasks = new Dictionary<int, Task>();
        _lastHeartbeat = new Dictionary<int, DateTime>();
        _monitoringCts = new CancellationTokenSource();
        _isDisposed = false;
    }

    /// <summary>
    /// Registers a worker for crash detection.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    /// <param name="workerTask">The task representing the worker's execution.</param>
    public void RegisterWorker(int workerId, Task workerTask)
    {
        if (workerTask == null)
            throw new ArgumentNullException(nameof(workerTask));

        lock (_lock)
        {
            _workerTasks[workerId] = workerTask;
            _lastHeartbeat[workerId] = DateTime.UtcNow;

            _logger?.LogDebug($"Registered worker {workerId} for crash detection");
        }
    }

    /// <summary>
    /// Updates the heartbeat timestamp for a worker.
    /// Should be called periodically by the worker to indicate it's still alive.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    public void UpdateHeartbeat(int workerId)
    {
        lock (_lock)
        {
            if (_lastHeartbeat.ContainsKey(workerId))
            {
                _lastHeartbeat[workerId] = DateTime.UtcNow;
            }
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

            _monitoringTask = Task.Run(() => MonitorAsync(_monitoringCts.Token));
            _logger?.LogInfo("Worker crash detection monitoring started");
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
            _logger?.LogInfo("Worker crash detection monitoring stopped");
        }
    }

    /// <summary>
    /// Unregisters a worker from crash detection.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker.</param>
    public void UnregisterWorker(int workerId)
    {
        lock (_lock)
        {
            _workerTasks.Remove(workerId);
            _lastHeartbeat.Remove(workerId);

            _logger?.LogDebug($"Unregistered worker {workerId} from crash detection");
        }
    }

    /// <summary>
    /// Monitoring loop that periodically checks worker status.
    /// </summary>
    private async Task MonitorAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                CheckWorkers();
                await Task.Delay(_checkInterval, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger?.LogError(new WorkerError(-1, ex, "CrashDetectorMonitor"));
            }
        }
    }

    /// <summary>
    /// Checks all registered workers for crashes or timeouts.
    /// </summary>
    private void CheckWorkers()
    {
        List<int> crashedWorkers = new List<int>();
        List<int> stalledWorkers = new List<int>();

        lock (_lock)
        {
            var workerIds = _workerTasks.Keys.ToList();

            foreach (var workerId in workerIds)
            {
                if (!_workerTasks.TryGetValue(workerId, out var workerTask))
                    continue;

                // Check for crashed workers (faulted or cancelled unexpectedly)
                if (workerTask.IsFaulted)
                {
                    crashedWorkers.Add(workerId);
                }
                else if (workerTask.IsCanceled && !cancellationToken.IsCancellationRequested)
                {
                    // Unexpected cancellation
                    crashedWorkers.Add(workerId);
                }

                // Check for stalled workers (no heartbeat within timeout)
                if (_lastHeartbeat.TryGetValue(workerId, out var lastHeartbeat))
                {
                    var timeSinceHeartbeat = DateTime.UtcNow - lastHeartbeat;
                    if (timeSinceHeartbeat > _heartbeatTimeout)
                    {
                        stalledWorkers.Add(workerId);
                    }
                }
            }
        }

        // Handle crashed workers
        foreach (var workerId in crashedWorkers)
        {
            Exception? exception = null;
            lock (_lock)
            {
                if (_workerTasks.TryGetValue(workerId, out var task) && task.Exception != null)
                {
                    exception = task.Exception.InnerException ?? task.Exception;
                }
            }

            var error = exception != null
                ? new WorkerError(workerId, exception, "CrashDetector")
                : new WorkerError(workerId, new InvalidOperationException("Worker crashed without exception"), "CrashDetector");

            _logger?.LogError(error);
            OnWorkerCrashed?.Invoke(error);
        }

        // Handle stalled workers
        foreach (var workerId in stalledWorkers)
        {
            var error = new WorkerError(
                workerId,
                new TimeoutException($"Worker stalled - no heartbeat for {_heartbeatTimeout.TotalSeconds} seconds"),
                "CrashDetector");

            _logger?.LogError(error);
            OnWorkerCrashed?.Invoke(error);
        }
    }

    /// <summary>
    /// Gets the number of registered workers.
    /// </summary>
    public int GetRegisteredWorkerCount()
    {
        lock (_lock)
        {
            return _workerTasks.Count;
        }
    }

    /// <summary>
    /// Gets the time since the last heartbeat for a specific worker.
    /// </summary>
    /// <param name="workerId">The worker ID to check.</param>
    /// <returns>The time since the last heartbeat, or null if worker is not registered.</returns>
    public TimeSpan? GetTimeSinceLastHeartbeat(int workerId)
    {
        lock (_lock)
        {
            if (_lastHeartbeat.TryGetValue(workerId, out var lastHeartbeat))
            {
                return DateTime.UtcNow - lastHeartbeat;
            }
            return null;
        }
    }

    /// <summary>
    /// Disposes of the crash detector and stops monitoring.
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
            _workerTasks.Clear();
            _lastHeartbeat.Clear();
        }
    }
}
