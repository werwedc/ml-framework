namespace MLFramework.Data;

/// <summary>
/// Delegate that defines the work each worker performs.
/// </summary>
/// <typeparam name="T">The type of data produced by the worker.</typeparam>
/// <param name="workerId">Unique identifier for this worker (0 to NumWorkers-1).</param>
/// <param name="cancellationToken">Token for checking cancellation during work.</param>
/// <returns>The processed data item to enqueue.</returns>
public delegate T DataWorker<T>(int workerId, CancellationToken cancellationToken);
