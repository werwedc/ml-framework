namespace MachineLearning.Checkpointing;

/// <summary>
/// Handler for fault tolerance and retry logic
/// </summary>
public class FaultToleranceHandler : IFaultToleranceHandler
{
    private readonly ICheckpointStorage? _storage;
    private readonly int _maxRetries;
    private readonly TimeSpan _retryDelay;

    /// <summary>
    /// Create a new FaultToleranceHandler with storage
    /// </summary>
    public FaultToleranceHandler(ICheckpointStorage storage)
        : this(storage, maxRetries: 3, retryDelay: TimeSpan.FromSeconds(1))
    {
    }

    /// <summary>
    /// Create a new FaultToleranceHandler with custom settings
    /// </summary>
    public FaultToleranceHandler(ICheckpointStorage? storage, int maxRetries, TimeSpan retryDelay)
    {
        _storage = storage;
        _maxRetries = maxRetries;
        _retryDelay = retryDelay;
    }

    /// <summary>
    /// Execute an operation with retry logic
    /// </summary>
    public async Task<T> ExecuteWithRetryAsync<T>(Func<Task<T>> operation, CancellationToken cancellationToken = default)
    {
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        Exception? lastException = null;

        for (int attempt = 0; attempt <= _maxRetries; attempt++)
        {
            try
            {
                return await operation();
            }
            catch (Exception ex) when (IsRetryableException(ex) && attempt < _maxRetries)
            {
                lastException = ex;
                await Task.Delay(_retryDelay, cancellationToken);
            }
        }

        throw lastException ?? new InvalidOperationException("Operation failed");
    }

    /// <summary>
    /// Execute an operation with retry logic (non-generic)
    /// </summary>
    public async Task ExecuteWithRetryAsync(Func<Task> operation, CancellationToken cancellationToken = default)
    {
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        Exception? lastException = null;

        for (int attempt = 0; attempt <= _maxRetries; attempt++)
        {
            try
            {
                await operation();
                return;
            }
            catch (Exception ex) when (IsRetryableException(ex) && attempt < _maxRetries)
            {
                lastException = ex;
                await Task.Delay(_retryDelay, cancellationToken);
            }
        }

        throw lastException ?? new InvalidOperationException("Operation failed");
    }

    /// <summary>
    /// Execute an operation with a timeout
    /// </summary>
    public async Task<T> ExecuteWithTimeoutAsync<T>(Func<Task<T>> operation, TimeSpan timeout, CancellationToken cancellationToken = default)
    {
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(timeout);

        try
        {
            return await operation().WaitAsync(cts.Token);
        }
        catch (OperationCanceledException ex) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"Operation timed out after {timeout.TotalSeconds} seconds", ex);
        }
    }

    /// <summary>
    /// Determine if an exception is retryable
    /// </summary>
    private bool IsRetryableException(Exception ex)
    {
        return ex is IOException
            || ex is TimeoutException
            || ex is System.Net.Http.HttpRequestException;
    }

    /// <summary>
    /// Rollback a failed checkpoint operation
    /// </summary>
    public async Task RollbackAsync(string checkpointPath, CancellationToken cancellationToken = default)
    {
        if (_storage != null && !string.IsNullOrWhiteSpace(checkpointPath))
        {
            try
            {
                // Delete checkpoint files
                await _storage.DeleteAsync(checkpointPath, cancellationToken);
            }
            catch (Exception ex)
            {
                // Log but don't throw - rollback is best-effort
                System.Diagnostics.Debug.WriteLine($"Failed to rollback checkpoint {checkpointPath}: {ex.Message}");
            }
        }
    }
}
