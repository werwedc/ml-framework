using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Data.Worker
{
    /// <summary>
    /// Worker pool that manages multiple worker tasks for parallel data loading.
    /// Each worker processes data independently using C# Task-based parallelism.
    /// Communication is handled via BlockingCollection queues.
    /// </summary>
    public class WorkerPool : IWorkerPool
    {
        private readonly int _numWorkers;
        private readonly CancellationTokenSource _cancellationToken;
        private readonly Task[] _workers;
        private readonly BlockingCollection<object> _taskQueue;
        private readonly BlockingCollection<object> _resultQueue;
        private volatile bool _isRunning;

        /// <summary>
        /// Initializes a new instance of the WorkerPool class.
        /// </summary>
        /// <param name="numWorkers">The number of worker tasks to create. Must be positive.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when numWorkers is less than or equal to zero.</exception>
        public WorkerPool(int numWorkers = 4)
        {
            if (numWorkers <= 0)
                throw new ArgumentOutOfRangeException(nameof(numWorkers), "Number of workers must be positive.");

            _numWorkers = numWorkers;
            _cancellationToken = new CancellationTokenSource();
            _taskQueue = new BlockingCollection<object>();
            _resultQueue = new BlockingCollection<object>();
            _workers = new Task[numWorkers];
            _isRunning = false;
        }

        /// <summary>
        /// Gets whether the worker pool is currently running.
        /// </summary>
        public bool IsRunning => _isRunning;

        /// <summary>
        /// Gets the number of workers in the pool.
        /// </summary>
        public int NumWorkers => _numWorkers;

        /// <summary>
        /// Starts the worker pool and initializes all worker tasks.
        /// </summary>
        public void Start()
        {
            if (_isRunning)
                return;

            _isRunning = true;

            for (int i = 0; i < _numWorkers; i++)
            {
                _workers[i] = Task.Run(() => WorkerLoop(_cancellationToken.Token));
            }
        }

        /// <summary>
        /// Worker loop that continuously processes tasks from the queue.
        /// </summary>
        /// <param name="token">Cancellation token to signal worker shutdown.</param>
        private void WorkerLoop(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                try
                {
                    // Get task (with timeout to allow cancellation)
                    var taskObj = _taskQueue.Take(token);

                    if (taskObj is Func<object> task)
                    {
                        var result = task();
                        _resultQueue.Add(result, token);
                    }
                }
                catch (OperationCanceledException)
                {
                    // Graceful shutdown
                    break;
                }
                catch (Exception ex)
                {
                    // Log error and continue
                    Console.WriteLine($"Worker error: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Submits a task to the worker pool for execution.
        /// </summary>
        /// <typeparam name="T">The type of result returned by the task.</typeparam>
        /// <param name="task">The task to execute.</param>
        /// <exception cref="InvalidOperationException">Thrown when the pool is not running.</exception>
        /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
        public void SubmitTask<T>(Func<T> task)
        {
            if (!_isRunning)
                throw new InvalidOperationException("Worker pool is not running");

            if (task == null)
                throw new ArgumentNullException(nameof(task));

            // Wrap the task to match Func<object>
            _taskQueue.Add(new Func<object>(() => task()!));
        }

        /// <summary>
        /// Gets a result from the worker pool. Blocks until a result is available.
        /// </summary>
        /// <typeparam name="T">The type of result expected.</typeparam>
        /// <returns>The result from a completed task.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the pool is not running.</exception>
        public T GetResult<T>()
        {
            if (!_isRunning)
                throw new InvalidOperationException("Worker pool is not running");

            var result = _resultQueue.Take();
            return (T)result;
        }

        /// <summary>
        /// Tries to get a result from the worker pool without blocking.
        /// </summary>
        /// <typeparam name="T">The type of result expected.</typeparam>
        /// <param name="result">The result if available.</param>
        /// <returns>True if a result was available, false otherwise.</returns>
        public bool TryGetResult<T>(out T result)
        {
            if (!_isRunning)
            {
                result = default(T);
                return false;
            }

            if (_resultQueue.TryTake(out var resultObj))
            {
                result = (T)resultObj;
                return true;
            }

            result = default(T);
            return false;
        }

        /// <summary>
        /// Stops the worker pool and gracefully cancels all workers.
        /// </summary>
        public void Stop()
        {
            if (!_isRunning)
                return;

            _cancellationToken.Cancel();

            // Wait for all workers to finish with a timeout
            Task.WaitAll(_workers, TimeSpan.FromSeconds(30));

            _isRunning = false;
        }

        /// <summary>
        /// Disposes of all resources used by the worker pool.
        /// </summary>
        public void Dispose()
        {
            Stop();
            _cancellationToken?.Dispose();
            _taskQueue?.Dispose();
            _resultQueue?.Dispose();
        }
    }
}
