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
    private Task[]? _workerTasks;
    private volatile bool _isRunning;
    private int _activeWorkers;

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
    /// Initializes a new instance of the WorkerPool class.
    /// </summary>
    /// <param name="workerFunc">Function that defines what each worker does.</param>
    /// <param name="outputQueue">Queue where workers deposit completed items.</param>
    /// <param name="numWorkers">Number of parallel workers.</param>
    /// <param name="cancellationToken">Optional cancellation token for graceful shutdown.</param>
    /// <exception cref="ArgumentNullException">Thrown when workerFunc or outputQueue is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when numWorkers is not positive.</exception>
    public WorkerPool(
        DataWorker<T> workerFunc,
        SharedQueue<T> outputQueue,
        int numWorkers,
        CancellationToken? cancellationToken = null)
    {
        _workerFunc = workerFunc ?? throw new ArgumentNullException(nameof(workerFunc));
        _outputQueue = outputQueue ?? throw new ArgumentNullException(nameof(outputQueue));

        if (numWorkers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numWorkers), numWorkers, "NumWorkers must be > 0.");

        _numWorkers = numWorkers;
        _cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken ?? CancellationToken.None);
        _isRunning = false;
        _activeWorkers = 0;
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
    }

    /// <summary>
    /// Worker loop that continuously produces data until stopped.
    /// </summary>
    /// <param name="workerId">The unique identifier for this worker.</param>
    private async Task WorkerLoopAsync(int workerId)
    {
        bool completedSuccessfully = false;
        Interlocked.Increment(ref _activeWorkers);

        try
        {
            WorkerStarted?.Invoke(workerId);

            while (!_cancellationTokenSource.IsCancellationRequested)
            {
                try
                {
                    // Perform work
                    T result = _workerFunc(workerId, _cancellationTokenSource.Token);

                    // Enqueue result
                    _outputQueue.Enqueue(result);
                }
                catch (OperationCanceledException)
                {
                    // Expected during shutdown
                    break;
                }
                catch (Exception ex)
                {
                    // Log error and break
                    System.Console.WriteLine($"Worker {workerId} error: {ex.Message}");
                    break;
                }
            }

            completedSuccessfully = true;
        }
        finally
        {
            Interlocked.Decrement(ref _activeWorkers);
            WorkerCompleted?.Invoke(workerId, completedSuccessfully);
        }
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
    }
}
