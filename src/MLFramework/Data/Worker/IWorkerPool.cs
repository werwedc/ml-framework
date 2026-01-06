using System;

namespace MLFramework.Data.Worker
{
    /// <summary>
    /// Interface for managing a pool of worker processes for parallel data loading.
    /// </summary>
    public interface IWorkerPool : IDisposable
    {
        /// <summary>
        /// Starts the worker pool.
        /// </summary>
        void Start();

        /// <summary>
        /// Stops the worker pool gracefully.
        /// </summary>
        void Stop();

        /// <summary>
        /// Gets whether the worker pool is currently running.
        /// </summary>
        bool IsRunning { get; }

        /// <summary>
        /// Gets the number of workers in the pool.
        /// </summary>
        int NumWorkers { get; }

        /// <summary>
        /// Submits a task to the worker pool for execution.
        /// </summary>
        /// <typeparam name="T">The type of result returned by the task.</typeparam>
        /// <param name="task">The task to execute.</param>
        void SubmitTask<T>(Func<T> task);

        /// <summary>
        /// Gets a result from the worker pool. Blocks until a result is available.
        /// </summary>
        /// <typeparam name="T">The type of result expected.</typeparam>
        /// <returns>The result from a completed task.</returns>
        T GetResult<T>();

        /// <summary>
        /// Tries to get a result from the worker pool without blocking.
        /// </summary>
        /// <typeparam name="T">The type of result expected.</typeparam>
        /// <param name="result">The result if available.</param>
        /// <returns>True if a result was available, false otherwise.</returns>
        bool TryGetResult<T>(out T result);
    }
}
