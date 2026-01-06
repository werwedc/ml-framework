using System;
using System.Collections.Generic;
using System.Threading;

namespace MLFramework.Data.Worker
{
    /// <summary>
    /// Context information for a worker in the pool.
    /// Provides access to worker-specific state and resources.
    /// </summary>
    public class WorkerContext
    {
        /// <summary>
        /// Gets or sets the unique identifier for this worker.
        /// </summary>
        public int WorkerId { get; set; }

        /// <summary>
        /// Gets or sets the thread associated with this worker.
        /// </summary>
        public Thread? WorkerThread { get; set; }

        /// <summary>
        /// Gets or sets the shared state dictionary for this worker.
        /// Can be used to store resources like file handles, database connections, or caches.
        /// </summary>
        public Dictionary<string, object> SharedState { get; set; }

        /// <summary>
        /// Initializes a new instance of the WorkerContext class.
        /// </summary>
        public WorkerContext()
        {
            SharedState = new Dictionary<string, object>();
        }

        /// <summary>
        /// Initializes a new instance of the WorkerContext class with specified parameters.
        /// </summary>
        /// <param name="workerId">The unique identifier for this worker.</param>
        /// <param name="workerThread">The thread associated with this worker.</param>
        public WorkerContext(int workerId, Thread workerThread) : this()
        {
            WorkerId = workerId;
            WorkerThread = workerThread;
        }

        /// <summary>
        /// Retrieves a value from the shared state.
        /// </summary>
        /// <typeparam name="T">The type of the value to retrieve.</typeparam>
        /// <param name="key">The key of the value to retrieve.</param>
        /// <returns>The value associated with the key, or default if not found.</returns>
        public T? GetState<T>(string key)
        {
            if (SharedState.TryGetValue(key, out var value) && value is T typedValue)
            {
                return typedValue;
            }
            return default;
        }

        /// <summary>
        /// Sets a value in the shared state.
        /// </summary>
        /// <typeparam name="T">The type of the value to set.</typeparam>
        /// <param name="key">The key of the value to set.</param>
        /// <param name="value">The value to set.</param>
        public void SetState<T>(string key, T value)
        {
            SharedState[key] = value!;
        }

        /// <summary>
        /// Checks if a key exists in the shared state.
        /// </summary>
        /// <param name="key">The key to check.</param>
        /// <returns>True if the key exists, false otherwise.</returns>
        public bool HasState(string key)
        {
            return SharedState.ContainsKey(key);
        }

        /// <summary>
        /// Removes a value from the shared state.
        /// </summary>
        /// <param name="key">The key of the value to remove.</param>
        /// <returns>True if the value was removed, false if the key didn't exist.</returns>
        public bool RemoveState(string key)
        {
            return SharedState.Remove(key);
        }
    }
}
