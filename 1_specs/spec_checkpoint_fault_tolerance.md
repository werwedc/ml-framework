# Spec: Fault Tolerance and Retry Logic

## Overview
Implement fault tolerance mechanisms for distributed checkpointing including timeout handling, retry logic with exponential backoff, and rollback capabilities for failed operations.

## Scope
- 30-45 minutes coding time
- Focus on error handling and recovery
- Target: `src/MLFramework/Checkpointing/FaultTolerance/`

## Classes

### 1. IFaultToleranceHandler (Interface)
```csharp
public interface IFaultToleranceHandler
{
    /// <summary>
    /// Execute operation with automatic retry on failure
    /// </summary>
    Task<T> ExecuteWithRetryAsync<T>(
        Func<Task<T>> operation,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Execute operation with timeout
    /// </summary>
    Task<T> ExecuteWithTimeoutAsync<T>(
        Func<Task<T>> operation,
        TimeSpan timeout,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Rollback checkpoint if operation fails
    /// </summary>
    Task RollbackAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default);
}
```

### 2. RetryPolicy (Retry Configuration)
```csharp
public class RetryPolicy
{
    public int MaxRetries { get; set; } = 3;
    public TimeSpan InitialDelay { get; set; } = TimeSpan.FromSeconds(1);
    public TimeSpan MaxDelay { get; set; } = TimeSpan.FromSeconds(30);
    public double BackoffFactor { get; set; } = 2.0;
    public List<Type> RetryableExceptions { get; set; } = new();

    public bool IsRetryable(Exception ex)
    {
        return RetryableExceptions.Any(type => type.IsInstanceOfType(ex));
    }
}
```

### 3. FaultToleranceHandler (Main Implementation)
```csharp
public class FaultToleranceHandler : IFaultToleranceHandler
{
    private readonly ICheckpointStorage _storage;
    private readonly RetryPolicy _retryPolicy;
    private readonly ILogger<FaultToleranceHandler> _logger;

    public FaultToleranceHandler(
        ICheckpointStorage storage,
        RetryPolicy? retryPolicy = null,
        ILogger<FaultToleranceHandler>? logger = null)
    {
        _storage = storage;
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

    public async Task<T> ExecuteWithRetryAsync<T>(
        Func<Task<T>> operation,
        CancellationToken cancellationToken = default)
    {
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

    public async Task<T> ExecuteWithTimeoutAsync<T>(
        Func<Task<T>> operation,
        TimeSpan timeout,
        CancellationToken cancellationToken = default)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(timeout);

        try
        {
            return await operation().WithCancellation(cts.Token);
        }
        catch (OperationCanceledException ex) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"Operation timed out after {timeout}", ex);
        }
    }

    public async Task RollbackAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
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

    private async Task RollbackMultiShardAsync(
        string checkpointPath,
        CancellationToken cancellationToken)
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
            // Find all shard files
            var shardPattern = $"{prefix}_shard_*.shard";
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

    private async Task RollbackSingleFileAsync(
        string checkpointPath,
        CancellationToken cancellationToken)
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

    private async Task<List<string>> FindShardFilesAsync(
        string prefix,
        CancellationToken cancellationToken)
    {
        // This is a simplified implementation
        // In practice, you'd need to list files in the storage backend
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
```

### 4. TimeoutHandler (Timeout Management)
```csharp
public class TimeoutHandler
{
    private readonly IDistributedCoordinator _coordinator;
    private readonly TimeSpan _defaultTimeout;
    private readonly ILogger<TimeoutHandler> _logger;

    public TimeoutHandler(
        IDistributedCoordinator coordinator,
        TimeSpan? defaultTimeout = null,
        ILogger<TimeoutHandler>? logger = null)
    {
        _coordinator = coordinator;
        _defaultTimeout = defaultTimeout ?? TimeSpan.FromMinutes(10);
        _logger = logger;
    }

    /// <summary>
    /// Wait for all ranks to reach a barrier with timeout
    /// </summary>
    public async Task BarrierAsync(
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        var actualTimeout = timeout ?? _defaultTimeout;

        _logger?.LogDebug("Entering barrier (timeout: {Timeout}s)", actualTimeout.TotalSeconds);

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(actualTimeout);

        try
        {
            await _coordinator.BarrierAsync(cts.Token);
        }
        catch (OperationCanceledException ex) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"Barrier timed out after {actualTimeout}", ex);
        }

        _logger?.LogDebug("Barrier completed");
    }

    /// <summary>
    /// Wait for specific rank with timeout
    /// </summary>
    public async Task WaitForRankAsync(
        int rank,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        var actualTimeout = timeout ?? _defaultTimeout;

        _logger?.LogDebug("Waiting for rank {Rank} (timeout: {Timeout}s)", rank, actualTimeout.TotalSeconds);

        // This is a simplified implementation
        // In practice, you'd need a distributed coordination mechanism
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(actualTimeout);

        try
        {
            await Task.Delay(actualTimeout, cts.Token);
        }
        catch (OperationCanceledException ex) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"Rank {rank} did not respond within {actualTimeout}", ex);
        }
    }
}
```

### 5. CircuitBreaker (Circuit Breaker Pattern)
```csharp
public class CircuitBreaker
{
    private readonly int _failureThreshold;
    private readonly TimeSpan _recoveryTimeout;
    private int _failureCount = 0;
    private DateTime? _lastFailureTime = null;
    private CircuitState _state = CircuitState.Closed;

    public CircuitBreaker(
        int failureThreshold = 5,
        TimeSpan? recoveryTimeout = null)
    {
        _failureThreshold = failureThreshold;
        _recoveryTimeout = recoveryTimeout ?? TimeSpan.FromMinutes(1);
    }

    public async Task<T> ExecuteAsync<T>(
        Func<Task<T>> operation,
        Func<Task<T>> fallback = null)
    {
        if (_state == CircuitState.Open)
        {
            if (ShouldAttemptReset())
            {
                _state = CircuitState.HalfOpen;
            }
            else
            {
                if (fallback != null)
                {
                    return await fallback();
                }
                throw new CircuitBreakerOpenException("Circuit breaker is OPEN");
            }
        }

        try
        {
            var result = await operation();
            OnSuccess();
            return result;
        }
        catch (Exception ex)
        {
            OnFailure();
            if (fallback != null)
            {
                return await fallback();
            }
            throw;
        }
    }

    private void OnSuccess()
    {
        _failureCount = 0;
        _lastFailureTime = null;
        _state = CircuitState.Closed;
    }

    private void OnFailure()
    {
        _failureCount++;
        _lastFailureTime = DateTime.UtcNow;

        if (_failureCount >= _failureThreshold)
        {
            _state = CircuitState.Open;
        }
    }

    private bool ShouldAttemptReset()
    {
        return _lastFailureTime.HasValue &&
               DateTime.UtcNow - _lastFailureTime.Value >= _recoveryTimeout;
    }

    public CircuitState State => _state;
}

public enum CircuitState
{
    Closed,
    Open,
    HalfOpen
}

public class CircuitBreakerOpenException : Exception
{
    public CircuitBreakerOpenException(string message) : base(message)
    {
    }
}
```

### 6. CheckpointRollbackManager (Rollback Orchestration)
```csharp
public class CheckpointRollbackManager
{
    private readonly IFaultToleranceHandler _faultHandler;
    private readonly ICheckpointStorage _storage;
    private readonly ILogger<CheckpointRollbackManager> _logger;

    public CheckpointRollbackManager(
        IFaultToleranceHandler faultHandler,
        ICheckpointStorage storage,
        ILogger<CheckpointRollbackManager>? logger = null)
    {
        _faultHandler = faultHandler;
        _storage = storage;
        _logger = logger;
    }

    /// <summary>
    /// Execute checkpoint save with automatic rollback on failure
    /// </summary>
    public async Task<string> SaveWithRollbackAsync(
        string checkpointPath,
        Func<Task<byte[]>> saveOperation,
        CancellationToken cancellationToken = default)
    {
        byte[]? tempData = null;
        string? tempPath = null;

        try
        {
            // Save to temporary location first
            tempData = await _faultHandler.ExecuteWithRetryAsync(saveOperation, cancellationToken);
            tempPath = $"{checkpointPath}.tmp";
            await _storage.WriteAsync(tempPath, tempData, cancellationToken);

            // Atomic move to final location
            if (_storage is LocalFileSystemStorage localStorage)
            {
                File.Move(tempPath, checkpointPath, overwrite: true);
            }
            else
            {
                // For cloud storage, write to final location
                await _storage.WriteAsync(checkpointPath, tempData, cancellationToken);
                await _storage.DeleteAsync(tempPath, cancellationToken);
            }

            return checkpointPath;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Checkpoint save failed, rolling back: {CheckpointPath}", checkpointPath);

            // Cleanup temporary file
            if (tempPath != null)
            {
                try
                {
                    await _storage.DeleteAsync(tempPath, cancellationToken);
                }
                catch (Exception cleanupEx)
                {
                    _logger?.LogWarning(cleanupEx, "Failed to cleanup temporary file: {TempPath}", tempPath);
                }
            }

            throw;
        }
    }
}
```

## Integration Points
- Used by: `CheckpointCoordinator`, `CheckpointLoader`, `DistributedCheckpoint`
- Depends on: `ICheckpointStorage`, `IDistributedCoordinator`

## Testing Requirements
- Test retry logic with various exception types
- Test exponential backoff calculation
- Test timeout handling
- Test rollback for single-file checkpoints
- Test rollback for multi-shard checkpoints
- Test circuit breaker state transitions
- Test save with rollback

## Success Criteria
- Retries on transient failures automatically
- Exponential backoff prevents thundering herd
- Timeout handling prevents indefinite hangs
- Rollback cleans up partial checkpoints
- Circuit breaker prevents cascading failures
- Comprehensive logging for debugging
