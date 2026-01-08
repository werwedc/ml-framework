using System.Threading.Tasks;

namespace MLFramework.Data;

/// <summary>
/// Manages a pool of worker tasks that produce data in parallel.
/// Workers communicate with the main process via a shared queue.
/// </summary>
/// <typeparam name="T">The type of data items produced by workers.</typeparam>
public sealed class WorkerPool<T> : IDisposable
{
    private readonly DataWorker<T> _workerFunc;
    private readonly SharedQueue<T> _outputQueue;
    private readonly CancellationTokenSource _cancellationTokenSource;
    private readonly int _numWorkers;
    private readonly ErrorAggregator _errorAggregator;
    private readonly WorkerCrashDetector _crashDetector;
    private readonly WorkerRecoveryService _recoveryService;
    private readonly WorkerTimeoutTracker _timeoutTracker;
    private readonly IErrorLogger? _logger;
    private readonly ErrorPolicy _errorPolicy;
    private readonly int _maxRetries;
    private readonly TimeSpan _workerTimeout;
    private readonly bool _enableErrorHandling;
    private Task[]? _workerTasks;
    private volatile bool _isRunning;
    private int _activeWorkers;
    private volatile int _failedWorkers;

    /// <summary>
    /// Gets whether the worker pool is currently running.
    /// </summary>
    public bool IsRunning => _isRunning;

    /// <summary>
    /// Gets the total number of workers in the pool.
    /// </summary>
    public int NumWorkers => _numWorkers;

    /// <summary>
    /// Gets the number of currently active workers.
    /// </summary>
    public int ActiveWorkers => _activeWorkers;

    /// <summary>
    /// Gets the number of failed workers.
    /// </summary>
    public int FailedWorkers => _failedWorkers;

    /// <summary>
    /// Gets the error aggregator that collects all worker errors.
    /// </summary>
    public ErrorAggregator ErrorAggregator => _errorAggregator;

    /// <summary>
    /// Event raised when a worker starts.
    /// </summary>
    public event Action<int>? WorkerStarted;

    /// <summary>
    /// Event raised when a worker completes.
    /// </summary>
    /// <remarks>
    /// The second parameter is true if completed successfully, false if failed.
    /// </remarks>
    public event Action<int, bool>? WorkerCompleted;

    /// <summary>
    /// Event raised when a worker encounters an error.
    /// </summary>
    public event Action<WorkerError>? OnWorkerError;

    /// <summary>
    /// Event raised when a worker is restarted.
    /// </summary>
    public event Action<int>? OnWorkerRestarted;

    /// <summary>
    /// Event raised when a worker is marked as failed.
    /// </summary>
    public event Action<int>? OnWorkerFailed;

    /// <summary>
    /// Initializes a new instance of the WorkerPool class.
    /// </summary>
    /// <param name="workerFunc">Function that defines what each worker does.</param>
    /// <param name="outputQueue">Queue where workers deposit completed items.</param>
    /// <param name="numWorkers">Number of parallel workers.</param>
    /// <param name="cancellationToken">Optional cancellation token for graceful shutdown.</param>
    /// <param name="errorPolicy">Error handling policy for worker failures.</param>
    /// <param name="maxRetries">Maximum number of retry attempts per worker.</param>
    /// <param name="workerTimeout">Timeout for worker operations before considering them stalled.</param>
    /// <param name="enableErrorHandling">Whether to enable advanced error handling features.</param>
    /// <param name="logger">Optional logger for recording error events.</param>
    /// <exception cref="ArgumentNullException">Thrown when workerFunc or outputQueue is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when numWorkers is not positive.</exception>
    public WorkerPool(
        DataWorker<T> workerFunc,
        SharedQueue<T> outputQueue,
        int numWorkers,
        CancellationToken? cancellationToken = null,
        ErrorPolicy errorPolicy = ErrorPolicy.Continue,
        int maxRetries = 3,
        TimeSpan? workerTimeout = null,
        bool enableErrorHandling = true,
        IErrorLogger? logger = null)
    {
        _workerFunc = workerFunc ?? throw new ArgumentNullException(nameof(workerFunc));
        _outputQueue = outputQueue ?? throw new ArgumentNullException(nameof(outputQueue));

        if (numWorkers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numWorkers), numWorkers, "NumWorkers must be > 0.");

        if (maxRetries < 0)
            throw new ArgumentOutOfRangeException(nameof(maxRetries), maxRetries, "MaxRetries must be >= 0.");

        _numWorkers = numWorkers;
        _cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken ?? CancellationToken.None);
        _errorPolicy = errorPolicy;
        _maxRetries = maxRetries;
        _workerTimeout = workerTimeout ?? TimeSpan.FromSeconds(30);
        _enableErrorHandling = enableErrorHandling;
        _logger = logger;

        _isRunning = false;
        _activeWorkers = 0;
        _failedWorkers = 0;

        // Initialize error handling components
        _errorAggregator = new ErrorAggregator();

        if (enableErrorHandling)
        {
            _crashDetector = new WorkerCrashDetector(
                _workerTimeout,
                TimeSpan.FromSeconds(1),
                errorPolicy,
                logger);

            _recoveryService = new WorkerRecoveryService(
                errorPolicy,
                maxRetries,
                logger);

            _timeoutTracker = new WorkerTimeoutTracker(
                _workerTimeout,
                TimeSpan.FromSeconds(1),
                logger);

            // Wire up error handling events
            _crashDetector.OnWorkerCrashed += HandleWorkerCrash;
            _timeoutTracker.OnWorkerTimeout += HandleWorkerTimeout;
            _recoveryService.OnWorkerRestarted += HandleWorkerRestart;
            _recoveryService.OnWorkerFailed += HandleWorkerFailure;
        }
        else
        {
            _crashDetector = null!;
            _recoveryService = null!;
            _timeoutTracker = null!;
        }
    }

    /// <summary>
    /// Starts the worker pool and launches all worker tasks.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the pool is already running.</exception>
    public void Start()
    {
        if (_isRunning)
            throw new InvalidOperationException("Worker pool is already running.");

        _isRunning = true;
        _workerTasks = new Task[_numWorkers];

        for (int i = 0; i < _numWorkers; i++)
        {
            int workerId = i;
            _workerTasks[i] = Task.Run(async () => await WorkerLoopAsync(workerId), _cancellationTokenSource.Token);
        }

        // Start error handling monitoring if enabled
        if (_enableErrorHandling)
        {
            _crashDetector?.StartMonitoring();
            _timeoutTracker?.StartMonitoring();
        }
    }

    /// <summary>
    /// Worker loop that continuously produces data until stopped.
    /// </summary>
    /// <param name="workerId">The unique identifier for this worker.</param>
    private async Task WorkerLoopAsync(int workerId)
    {
        bool completedSuccessfully = false;
        Interlocked.Increment(ref _activeWorkers);
        int retryCount = 0;

        try
        {
            WorkerStarted?.Invoke(workerId);

            // Register with crash detector if error handling is enabled
            if (_enableErrorHandling)
            {
                _crashDetector.RegisterWorker(workerId, Task.CurrentId == null ? Task.CompletedTask : Task.CurrentTask);
            }

            while (!_cancellationTokenSource.IsCancellationRequested)
            {
                try
                {
                    // Update heartbeat if error handling is enabled
                    if (_enableErrorHandling)
                    {
                        _crashDetector.UpdateHeartbeat(workerId);
                        _timeoutTracker.StartOperation(workerId);
                    }

                    // Perform work
                    T result = _workerFunc(workerId, _cancellationTokenSource.Token);

                    // Enqueue result
                    _outputQueue.Enqueue(result);

                    // Clear retry count on success
                    retryCount = 0;

                    // End operation tracking
                    if (_enableErrorHandling)
                    {
                        _timeoutTracker.EndOperation(workerId);
                    }
                }
                catch (OperationCanceledException)
                {
                    // Expected during shutdown
                    break;
                }
                catch (Exception ex)
                {
                    // Handle error based on policy
                    var error = new WorkerError(workerId, ex, "WorkerLoop");
                    _errorAggregator.AddError(error);
                    OnWorkerError?.Invoke(error);

                    _logger?.LogError(error);

                    if (_enableErrorHandling && _errorPolicy == ErrorPolicy.Restart && retryCount < _maxRetries)
                    {
                        retryCount++;
                        var delayMs = 100 * retryCount;

                        _logger?.LogInfo($"Retrying worker {workerId} in {delayMs}ms (attempt {retryCount}/{_maxRetries})");

                        // Delay before retry with exponential backoff
                        await Task.Delay(delayMs, _cancellationTokenSource.Token);
                        continue;
                    }
                    else if (_errorPolicy == ErrorPolicy.Continue)
                    {
                        // Skip this worker
                        _logger?.LogWarning($"Worker {workerId} encountered error and will be skipped due to Continue policy");
                        break;
                    }
                    else if (_errorPolicy == ErrorPolicy.Ignore)
                    {
                        // Silently ignore and continue
                        continue;
                    }
                    else // FailFast or no more retries
                    {
                        _logger?.LogError(error);
                        Interlocked.Increment(ref _failedWorkers);
                        throw;
                    }
                }
            }

            completedSuccessfully = true;
        }
        finally
        {
            Interlocked.Decrement(ref _activeWorkers);

            // Unregister from crash detector
            if (_enableErrorHandling)
            {
                _crashDetector.UnregisterWorker(workerId);
                _timeoutTracker.EndOperation(workerId);
            }

            WorkerCompleted?.Invoke(workerId, completedSuccessfully);
        }
    }

    /// <summary>
    /// Handles worker crash events from the crash detector.
    /// </summary>
    private void HandleWorkerCrash(WorkerError error)
    {
        _errorAggregator.AddError(error);
        OnWorkerError?.Invoke(error);

        if (_errorPolicy == ErrorPolicy.Restart)
        {
            _logger?.LogInfo($"Attempting to recover crashed worker {error.WorkerId}");
            _recoveryService.TryRestartWorkerAsync(error.WorkerId, error, RestartWorkerAsync).Wait();
        }
        else if (_errorPolicy == ErrorPolicy.FailFast)
        {
            _logger?.LogError(error);
            Interlocked.Increment(ref _failedWorkers);
            _cancellationTokenSource.Cancel();
        }
    }

    /// <summary>
    /// Handles worker timeout events from the timeout tracker.
    /// </summary>
    private void HandleWorkerTimeout(int workerId, TimeSpan elapsed)
    {
        var error = new WorkerError(
            workerId,
            new TimeoutException($"Worker timed out after {elapsed.TotalSeconds:F2}s"),
            "TimeoutTracker");

        _errorAggregator.AddError(error);
        OnWorkerError?.Invoke(error);

        if (_errorPolicy == ErrorPolicy.Restart)
        {
            _logger?.LogInfo($"Attempting to recover timed-out worker {workerId}");
            _recoveryService.TryRestartWorkerAsync(workerId, error, RestartWorkerAsync).Wait();
        }
    }

    /// <summary>
    /// Handles worker restart events from the recovery service.
    /// </summary>
    private void HandleWorkerRestart(int workerId, int retryCount)
    {
        _logger?.LogInfo($"Worker {workerId} restarted successfully (retry {retryCount})");
        Interlocked.Decrement(ref _failedWorkers);
        OnWorkerRestarted?.Invoke(workerId);
    }

    /// <summary>
    /// Handles worker failure events from the recovery service.
    /// </summary>
    private void HandleWorkerFailure(int workerId)
    {
        _logger?.LogWarning($"Worker {workerId} marked as permanently failed");
        Interlocked.Increment(ref _failedWorkers);
        OnWorkerFailed?.Invoke(workerId);
    }

    /// <summary>
    /// Attempts to restart a worker by creating a new task.
    /// </summary>
    private async Task RestartWorkerAsync(int workerId)
    {
        var newTask = Task.Run(async () => await WorkerLoopAsync(workerId), _cancellationTokenSource.Token);

        // Update the worker task in the array
        if (_workerTasks != null && workerId < _workerTasks.Length)
        {
            _workerTasks[workerId] = newTask;
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Stops all workers and waits for them to complete gracefully.
    /// </summary>
    /// <param name="timeout">Maximum time to wait for workers to stop.</param>
    /// <exception cref="TimeoutException">Thrown when workers don't stop within the timeout.</exception>
    public async Task StopAsync(TimeSpan timeout)
    {
        if (!_isRunning)
            return;

        // Stop error handling monitoring
        if (_enableErrorHandling)
        {
            _crashDetector?.StopMonitoring();
            _timeoutTracker?.StopMonitoring();
        }

        // Signal cancellation
        _cancellationTokenSource.Cancel();

        // Wait for all workers to complete
        if (_workerTasks != null)
        {
            try
            {
                await Task.WhenAll(_workerTasks).WaitAsync(timeout);
            }
            catch (TimeoutException)
            {
                throw new TimeoutException($"Worker pool did not stop within {timeout.TotalSeconds} seconds.");
            }
        }

        // Mark output queue as complete
        _outputQueue.CompleteAdding();
        _isRunning = false;
    }

    /// <summary>
    /// Waits for all worker tasks to complete.
    /// </summary>
    /// <exception cref="AggregateException">Thrown when any worker task fails.</exception>
    public async Task WaitAsync()
    {
        if (_workerTasks != null)
        {
            await Task.WhenAll(_workerTasks);
        }
    }

    /// <summary>
    /// Disposes of all resources used by the worker pool.
    /// </summary>
    public void Dispose()
    {
        if (_isRunning)
        {
            try
            {
                StopAsync(TimeSpan.FromSeconds(5)).GetAwaiter().GetResult();
            }
            catch
            {
                // Ignore cleanup errors during disposal
            }
        }

        _cancellationTokenSource.Dispose();

        // Dispose error handling components
        if (_enableErrorHandling)
        {
            _crashDetector?.Dispose();
            _timeoutTracker?.Dispose();
        }
    }
}
