namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for fault tolerance handling
/// </summary>
public interface IFaultToleranceHandler
{
    /// <summary>
    /// Execute an operation with retry logic
    /// </summary>
    Task<T> ExecuteWithRetryAsync<T>(Func<Task<T>> operation, CancellationToken cancellationToken = default);

    /// <summary>
    /// Execute an operation with retry logic (non-generic)
    /// </summary>
    Task ExecuteWithRetryAsync(Func<Task> operation, CancellationToken cancellationToken = default);

    /// <summary>
    /// Execute an operation with a timeout
    /// </summary>
    Task<T> ExecuteWithTimeoutAsync<T>(Func<Task<T>> operation, TimeSpan timeout, CancellationToken cancellationToken = default);

    /// <summary>
    /// Rollback a failed checkpoint operation
    /// </summary>
    Task RollbackAsync(string checkpointPath, CancellationToken cancellationToken = default);
}
