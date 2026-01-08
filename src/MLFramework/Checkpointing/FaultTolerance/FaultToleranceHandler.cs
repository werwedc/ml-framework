using Microsoft.Extensions.Logging;

namespace MachineLearning.Checkpointing;

/// <summary>
/// Handler for fault tolerance and retry logic
/// </summary>
public class FaultToleranceHandler : IFaultToleranceHandler
{
    private readonly ICheckpointStorage _storage;
    private readonly RetryPolicy _retryPolicy;
    private readonly ILogger<FaultToleranceHandler>? _logger;

    /// <summary>
    /// Create a new FaultToleranceHandler with storage
    /// </summary>
    public FaultToleranceHandler(ICheckpointStorage storage, RetryPolicy? retryPolicy = null, ILogger<FaultToleranceHandler>? logger = null)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        _retryPolicy = retryPolicy ?? new RetryPolicy();
        _logger = logger;

        // Default retryable exceptions
        if (_retryPolicy.RetryableExceptions.Count == 0)
        {
            _retryPolicy.RetryableExceptions.Add(typeof(IOException));
            _retryPolicy.RetryableExceptions.Add(typeof(TimeoutException));
            _retryPolicy.RetryableExceptions.Add(typeof(TaskCanceledException));
        }
    }

    /// <summary>
    /// Execute an operation with retry logic
    /// </summary>
    public async Task<T> ExecuteWithRetryAsync<T>(Func<Task<T>> operation, CancellationToken cancellationToken = default)
    {
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        int attempt = 0;
        TimeSpan delay = _retryPolicy.InitialDelay;

        while (true)
        {
            attempt++;
            try
            {
                return await operation();
            }
            catch (Exception ex) when (attempt < _retryPolicy.MaxRetries && _retryPolicy.IsRetryable(ex))
            {
                _logger?.LogWarning(
                    ex,
                    "Operation failed (attempt {Attempt}/{MaxRetries}), retrying in {Delay}s...",
                    attempt,
                    _retryPolicy.MaxRetries,
                    delay.TotalSeconds);

                await Task.Delay(delay, cancellationToken);
                delay = CalculateBackoff(delay);
            }
        }
    }

    /// <summary>
    /// Execute an operation with retry logic (non-generic)
    /// </summary>
    public async Task ExecuteWithRetryAsync(Func<Task> operation, CancellationToken cancellationToken = default)
    {
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        int attempt = 0;
        TimeSpan delay = _retryPolicy.InitialDelay;

        while (true)
        {
            attempt++;
            try
            {
                await operation();
                return;
            }
            catch (Exception ex) when (attempt < _retryPolicy.MaxRetries && _retryPolicy.IsRetryable(ex))
            {
                _logger?.LogWarning(
                    ex,
                    "Operation failed (attempt {Attempt}/{MaxRetries}), retrying in {Delay}s...",
                    attempt,
                    _retryPolicy.MaxRetries,
                    delay.TotalSeconds);

                await Task.Delay(delay, cancellationToken);
                delay = CalculateBackoff(delay);
            }
        }
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
            throw new TimeoutException($"Operation timed out after {timeout}", ex);
        }
    }

    /// <summary>
    /// Rollback a failed checkpoint operation
    /// </summary>
    public async Task RollbackAsync(string checkpointPath, CancellationToken cancellationToken = default)
    {
        _logger?.LogInformation("Rolling back checkpoint: {CheckpointPath}", checkpointPath);

        // Check if it's a multi-shard checkpoint
        if (checkpointPath.EndsWith(".metadata.json"))
        {
            await RollbackMultiShardAsync(checkpointPath, cancellationToken);
        }
        else
        {
            await RollbackSingleFileAsync(checkpointPath, cancellationToken);
        }

        _logger?.LogInformation("Rollback completed: {CheckpointPath}", checkpointPath);
    }

    private async Task RollbackMultiShardAsync(string checkpointPath, CancellationToken cancellationToken)
    {
        var prefix = checkpointPath.Replace(".metadata.json", "");

        // Delete metadata file
        try
        {
            await _storage.DeleteAsync(checkpointPath, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "Failed to delete metadata file during rollback: {Path}", checkpointPath);
        }

        // Delete all shard files
        try
        {
            var shardFiles = await FindShardFilesAsync(prefix, cancellationToken);

            foreach (var shardFile in shardFiles)
            {
                try
                {
                    await _storage.DeleteAsync(shardFile, cancellationToken);
                }
                catch (Exception ex)
                {
                    _logger?.LogWarning(ex, "Failed to delete shard file during rollback: {Path}", shardFile);
                }
            }
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "Error during shard file cleanup: {Message}", ex.Message);
        }
    }

    private async Task RollbackSingleFileAsync(string checkpointPath, CancellationToken cancellationToken)
    {
        try
        {
            await _storage.DeleteAsync(checkpointPath, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "Failed to delete checkpoint file during rollback: {Path}", checkpointPath);
        }
    }

    private async Task<List<string>> FindShardFilesAsync(string prefix, CancellationToken cancellationToken)
    {
        var shards = new List<string>();

        // Try ranks 0 to 100 (reasonable upper limit)
        for (int rank = 0; rank < 100; rank++)
        {
            var shardPath = $"{prefix}_shard_{rank}.shard";
            if (await _storage.ExistsAsync(shardPath, cancellationToken))
            {
                shards.Add(shardPath);
            }
            else if (rank > 10 && shards.Count == 0)
            {
                // If we haven't found any shards after rank 10, stop searching
                break;
            }
        }

        return shards;
    }

    private TimeSpan CalculateBackoff(TimeSpan currentDelay)
    {
        var newDelay = TimeSpan.FromSeconds(
            currentDelay.TotalSeconds * _retryPolicy.BackoffFactor);
        return TimeSpan.FromSeconds(
            Math.Min(newDelay.TotalSeconds, _retryPolicy.MaxDelay.TotalSeconds));
    }
}
