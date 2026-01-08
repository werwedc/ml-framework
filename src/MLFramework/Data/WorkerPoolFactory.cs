namespace MLFramework.Data;

/// <summary>
/// Static factory class providing convenient methods for creating worker pools.
/// </summary>
public static class WorkerPoolFactory
{
    /// <summary>
    /// Creates a worker pool for dataset loading with static partitioning.
    /// </summary>
    /// <typeparam name="T">The type of data in the dataset.</typeparam>
    /// <param name="dataset">The dataset to load data from.</param>
    /// <param name="outputQueue">Queue where workers deposit completed items.</param>
    /// <param name="config">Configuration for the worker pool.</param>
    /// <param name="cancellationToken">Optional cancellation token for graceful shutdown.</param>
    /// <returns>A configured worker pool ready to start.</returns>
    /// <exception cref="ArgumentNullException">Thrown when dataset or outputQueue is null.</exception>
    public static WorkerPool<T> CreateForDataset<T>(
        IDataset<T> dataset,
        SharedQueue<T> outputQueue,
        DataLoaderConfig config,
        CancellationToken? cancellationToken = null)
    {
        if (dataset == null)
            throw new ArgumentNullException(nameof(dataset));

        if (outputQueue == null)
            throw new ArgumentNullException(nameof(outputQueue));

        // Create worker function that loads from dataset
        DataWorker<T> workerFunc = (workerId, token) =>
        {
            var context = new WorkerContext(workerId, config.NumWorkers, dataset.Length, token);

            // This is a simple partitioned worker that loads items one at a time
            // In a real implementation, you might want to batch load items
            // For this implementation, we'll just return null as a placeholder
            // The actual implementation would need to track state across calls
            throw new NotImplementedException("Dataset partitioned worker requires state management across calls.");
        };

        return new WorkerPool<T>(workerFunc, outputQueue, config.NumWorkers, cancellationToken);
    }

    /// <summary>
    /// Creates a worker pool with a simple indexing worker.
    /// </summary>
    /// <typeparam name="T">The type of data items.</typeparam>
    /// <param name="getItem">Function to get an item by index.</param>
    /// <param name="totalItems">Total number of items to process.</param>
    /// <param name="outputQueue">Queue where workers deposit completed items.</param>
    /// <param name="config">Configuration for the worker pool.</param>
    /// <param name="cancellationToken">Optional cancellation token for graceful shutdown.</param>
    /// <returns>A configured worker pool ready to start.</returns>
    /// <exception cref="ArgumentNullException">Thrown when getItem or outputQueue is null.</exception>
    public static WorkerPool<T> CreateForIndexedData<T>(
        Func<int, T> getItem,
        int totalItems,
        SharedQueue<T> outputQueue,
        DataLoaderConfig config,
        CancellationToken? cancellationToken = null)
    {
        if (getItem == null)
            throw new ArgumentNullException(nameof(getItem));

        if (outputQueue == null)
            throw new ArgumentNullException(nameof(outputQueue));

        // Create worker function that processes partitioned indices
        DataWorker<T> workerFunc = (workerId, token) =>
        {
            var context = new WorkerContext(workerId, config.NumWorkers, totalItems, token);
            T result = default!;
            bool hasItem = false;

            // Process all items in this worker's partition
            for (int index = context.StartIndex; index < context.EndIndex; index++)
            {
                if (token.IsCancellationRequested)
                    break;

                try
                {
                    result = getItem(index);
                    hasItem = true;
                }
                catch (Exception ex)
                {
                    System.Console.WriteLine($"Error loading item at index {index}: {ex.Message}");
                }
            }

            if (!hasItem)
                throw new InvalidOperationException("Worker produced no items.");

            return result;
        };

        return new WorkerPool<T>(workerFunc, outputQueue, config.NumWorkers, cancellationToken);
    }
}
