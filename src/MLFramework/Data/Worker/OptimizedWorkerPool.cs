using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Data.Worker
{
    /// <summary>
    /// Optimized worker pool that supports lazy initialization, warmup batches, and custom worker initialization.
    /// Reduces startup overhead by delaying worker creation until first task submission.
    /// </summary>
    public class OptimizedWorkerPool : WorkerPool
    {
        private readonly bool _lazyInitialization;
        private readonly int _warmupBatches;
        private readonly Action<WorkerContext>? _workerInitializer;
        private readonly Stopwatch _initializationTimer;
        private volatile bool _workersCreated;
        private readonly ConcurrentDictionary<int, WorkerContext> _workerContexts;
        private readonly object _creationLock;

        /// <summary>
        /// Initializes a new instance of the OptimizedWorkerPool class.
        /// </summary>
        /// <param name="numWorkers">The number of worker tasks to create. Must be positive.</param>
        /// <param name="lazyInitialization">If true, workers are created on first task submission. If false, workers are created immediately on Start().</param>
        /// <param name="warmupBatches">The number of warmup batches to process during initialization. 0 means no warmup.</param>
        /// <param name="workerInitializer">An optional callback to initialize each worker's context.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when numWorkers is less than or equal to zero or warmupBatches is negative.</exception>
        public OptimizedWorkerPool(
            int numWorkers = 4,
            bool lazyInitialization = true,
            int warmupBatches = 0,
            Action<WorkerContext>? workerInitializer = null)
            : base(numWorkers)
        {
            if (warmupBatches < 0)
                throw new ArgumentOutOfRangeException(nameof(warmupBatches), "Warmup batch count cannot be negative.");

            _lazyInitialization = lazyInitialization;
            _warmupBatches = warmupBatches;
            _workerInitializer = workerInitializer;
            _initializationTimer = new Stopwatch();
            _workersCreated = false;
            _workerContexts = new ConcurrentDictionary<int, WorkerContext>();
            _creationLock = new object();
        }

        /// <summary>
        /// Gets the time elapsed during worker pool initialization.
        /// </summary>
        public TimeSpan InitializationTime => _initializationTimer.Elapsed;

        /// <summary>
        /// Gets whether the workers have been created.
        /// </summary>
        public bool WorkersInitialized => _workersCreated;

        /// <summary>
        /// Starts the worker pool.
        /// If lazy initialization is enabled, workers are not created until the first task is submitted.
        /// If lazy initialization is disabled, workers are created immediately.
        /// </summary>
        public new void Start()
        {
            if (_workersCreated)
                return;

            _initializationTimer.Start();

            if (!_lazyInitialization)
            {
                // Eager initialization
                CreateWorkers();
            }

            _initializationTimer.Stop();
        }

        /// <summary>
        /// Creates all workers and initializes their contexts.
        /// This method is thread-safe and will only create workers once.
        /// </summary>
        private void CreateWorkers()
        {
            if (_workersCreated)
                return;

            lock (_creationLock)
            {
                // Double-check pattern for thread safety
                if (_workersCreated)
                    return;

                _initializationTimer.Start();

                // Create worker contexts for each worker
                for (int i = 0; i < NumWorkers; i++)
                {
                    int workerId = i;

                    // Create worker context
                    var context = new WorkerContext
                    {
                        WorkerId = workerId,
                        WorkerThread = Thread.CurrentThread
                    };

                    // Run user initializer if provided
                    _workerInitializer?.Invoke(context);

                    // Store the context
                    _workerContexts[workerId] = context;
                }

                // Start the base worker pool
                base.Start();

                _workersCreated = true;

                // Process warmup batches if configured
                if (_warmupBatches > 0)
                {
                    ProcessWarmupBatches();
                }

                _initializationTimer.Stop();
            }
        }

        /// <summary>
        /// Submits a task to the worker pool for execution.
        /// If lazy initialization is enabled and workers haven't been created yet,
        /// this will trigger worker creation.
        /// </summary>
        /// <typeparam name="T">The type of result returned by the task.</typeparam>
        /// <param name="task">The task to execute.</param>
        /// <exception cref="InvalidOperationException">Thrown when the pool is not running.</exception>
        /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
        public new void SubmitTask<T>(Func<T> task)
        {
            if (!_workersCreated && _lazyInitialization)
            {
                // Lazy initialization on first task
                CreateWorkers();
            }

            base.SubmitTask(task);
        }

        /// <summary>
        /// Processes warmup batches to populate caches and reduce initial latency.
        /// Warmup tasks are submitted and their results are discarded.
        /// </summary>
        private void ProcessWarmupBatches()
        {
            // Placeholder: Process warmup batches to populate caches
            // In a real implementation, this would interface with the actual data loading pipeline

            for (int i = 0; i < _warmupBatches; i++)
            {
                // Submit dummy tasks that load and cache data
                // This is a placeholder - actual implementation would depend on the data loading pipeline
                try
                {
                    base.SubmitTask(() =>
                    {
                        // Warmup task - in real usage, this would load actual data
                        return i;
                    });
                }
                catch (Exception ex)
                {
                    // Log warmup failure but continue
                    Console.WriteLine($"Warmup batch {i} failed: {ex.Message}");
                }
            }

            // Discard warmup results
            int warmupResultsProcessed = 0;
            while (warmupResultsProcessed < _warmupBatches)
            {
                if (base.TryGetResult<object>(out _))
                {
                    warmupResultsProcessed++;
                }
                else
                {
                    // No more results available
                    break;
                }
            }
        }

        /// <summary>
        /// Gets the worker context for a specific worker.
        /// </summary>
        /// <param name="workerId">The ID of the worker.</param>
        /// <returns>The worker context, or null if the worker hasn't been created.</returns>
        public WorkerContext? GetWorkerContext(int workerId)
        {
            if (workerId < 0 || workerId >= NumWorkers)
                throw new ArgumentOutOfRangeException(nameof(workerId), "Worker ID is out of range.");

            _workerContexts.TryGetValue(workerId, out var context);
            return context;
        }

        /// <summary>
        /// Gets the worker context for the current worker (if called from within a worker).
        /// This method uses a best-effort approach to identify the current worker.
        /// </summary>
        /// <returns>The current worker's context, or null if not found.</returns>
        public WorkerContext? GetCurrentWorkerContext()
        {
            // In a real implementation, you might use AsyncLocal or thread-local storage
            // to track the current worker ID. For now, we return the first context
            // as a placeholder.
            foreach (var context in _workerContexts.Values)
            {
                if (context.WorkerThread == Thread.CurrentThread)
                {
                    return context;
                }
            }

            return null;
        }
    }
}
