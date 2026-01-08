namespace MLFramework.Communication.FaultTolerance;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;

/// <summary>
/// Fault-tolerant wrapper for communication operations
/// </summary>
public class FaultTolerantCommunication : ICommunicationBackend, IAsyncCommunicationBackend
{
    private readonly ICommunicationBackend _innerBackend;
    private readonly IAsyncCommunicationBackend? _innerAsyncBackend;
    private readonly TimeoutManager _timeoutManager;
    private readonly ErrorRecoveryManager _errorRecoveryManager;
    private readonly CommunicationConfig _config;
    private int _nextOperationId;
    private bool _disposed;

    public int Rank => _innerBackend.Rank;
    public int WorldSize => _innerBackend.WorldSize;
    public string BackendName => $"FT_{_innerBackend.BackendName}";

    public FaultTolerantCommunication(
        ICommunicationBackend backend,
        CommunicationConfig config)
    {
        _innerBackend = backend ?? throw new ArgumentNullException(nameof(backend));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _timeoutManager = new TimeoutManager(config.TimeoutMs);
        _errorRecoveryManager = new ErrorRecoveryManager(config.MaxRetries, TimeSpan.FromMilliseconds(config.RetryDelayMs));
        _innerAsyncBackend = backend as IAsyncCommunicationBackend;
        _nextOperationId = 1;
    }

    public void Broadcast(Tensor tensor, int rootRank)
    {
        ExecuteWithRetry<object>("Broadcast", () => { _innerBackend.Broadcast(tensor, rootRank); return null!; });
    }

    public Tensor Reduce(Tensor tensor, ReduceOp operation, int rootRank)
    {
        return ExecuteWithRetry("Reduce", () => _innerBackend.Reduce(tensor, operation, rootRank));
    }

    public Tensor AllReduce(Tensor tensor, ReduceOp operation)
    {
        return ExecuteWithRetry("AllReduce", () => _innerBackend.AllReduce(tensor, operation));
    }

    public Tensor AllGather(Tensor tensor)
    {
        return ExecuteWithRetry("AllGather", () => _innerBackend.AllGather(tensor));
    }

    public Tensor ReduceScatter(Tensor tensor, ReduceOp operation)
    {
        return ExecuteWithRetry("ReduceScatter", () => _innerBackend.ReduceScatter(tensor, operation));
    }

    public void Barrier()
    {
        ExecuteWithRetry<object>("Barrier", () => { _innerBackend.Barrier(); return null!; });
    }

    private T ExecuteWithRetry<T>(string operationType, Func<T> func)
    {
        int operationId = Interlocked.Increment(ref _nextOperationId);
        int retryCount = 0;

        while (retryCount <= _errorRecoveryManager.MaxRetries)
        {
            try
            {
                var token = _timeoutManager.StartTimeout(operationId);
                try
                {
                    return func();
                }
                finally
                {
                    // Note: We can't use using with CancellationToken, so we rely on the finally block
                    // to clean up the timeout
                }
            }
            catch (OperationCanceledException)
            {
                var error = new CommunicationError
                {
                    OperationId = operationId,
                    OperationType = operationType,
                    Severity = ErrorSeverity.Recoverable,
                    Exception = new CommunicationTimeoutException(
                        $"Operation {operationType} timed out after {_config.TimeoutMs}ms",
                        TimeSpan.FromMilliseconds(_config.TimeoutMs)),
                    Timestamp = DateTime.Now,
                    Rank = _innerBackend.Rank,
                    Context = new Dictionary<string, string> { ["RetryCount"] = retryCount.ToString() }
                };

                if (!_errorRecoveryManager.HandleError(error))
                {
                    throw error.Exception;
                }

                retryCount++;
                Thread.Sleep((int)_errorRecoveryManager.RetryDelay.TotalMilliseconds);
            }
            catch (Exception ex)
            {
                var error = new CommunicationError
                {
                    OperationId = operationId,
                    OperationType = operationType,
                    Severity = ErrorSeverity.Fatal,
                    Exception = ex,
                    Timestamp = DateTime.Now,
                    Rank = _innerBackend.Rank,
                    Context = new Dictionary<string, string> { ["RetryCount"] = retryCount.ToString() }
                };

                if (!_errorRecoveryManager.HandleError(error))
                {
                    throw error.Exception;
                }

                retryCount++;
            }
            finally
            {
                _timeoutManager.CancelTimeout(operationId);
            }
        }

        throw new CommunicationException(
            $"Operation {operationType} failed after {_errorRecoveryManager.MaxRetries} retries");
    }

    // Async operations
    public ICommunicationHandle BroadcastAsync(Tensor tensor, int rootRank)
    {
        if (_innerAsyncBackend == null)
            throw new NotSupportedException("Backend does not support async operations");

        return ExecuteWithRetryAsync("BroadcastAsync", () => _innerAsyncBackend.BroadcastAsync(tensor, rootRank));
    }

    public ICommunicationHandle AllReduceAsync(Tensor tensor, ReduceOp operation)
    {
        if (_innerAsyncBackend == null)
            throw new NotSupportedException("Backend does not support async operations");

        return ExecuteWithRetryAsync("AllReduceAsync", () => _innerAsyncBackend.AllReduceAsync(tensor, operation));
    }

    public ICommunicationHandle BarrierAsync()
    {
        if (_innerAsyncBackend == null)
            throw new NotSupportedException("Backend does not support async operations");

        return ExecuteWithRetryAsync("BarrierAsync", () => _innerAsyncBackend.BarrierAsync());
    }

    private ICommunicationHandle ExecuteWithRetryAsync(
        string operationType,
        Func<ICommunicationHandle> func)
    {
        int operationId = Interlocked.Increment(ref _nextOperationId);
        int retryCount = 0;

        while (retryCount <= _errorRecoveryManager.MaxRetries)
        {
            try
            {
                return func();
            }
            catch (Exception ex)
            {
                var error = new CommunicationError
                {
                    OperationId = operationId,
                    OperationType = operationType,
                    Severity = ErrorSeverity.Recoverable,
                    Exception = ex,
                    Timestamp = DateTime.Now,
                    Rank = _innerBackend.Rank,
                    Context = new Dictionary<string, string> { ["RetryCount"] = retryCount.ToString() }
                };

                if (!_errorRecoveryManager.HandleError(error))
                {
                    throw error.Exception;
                }

                retryCount++;
            }
        }

        throw new CommunicationException(
            $"Async operation {operationType} failed after {_errorRecoveryManager.MaxRetries} retries");
    }

    /// <summary>
    /// Get error recovery manager
    /// </summary>
    public ErrorRecoveryManager ErrorRecoveryManager => _errorRecoveryManager;

    /// <summary>
    /// Get timeout manager
    /// </summary>
    public TimeoutManager TimeoutManager => _timeoutManager;

    public void Dispose()
    {
        if (!_disposed)
        {
            _timeoutManager.Dispose();
            _errorRecoveryManager.Dispose();
            _disposed = true;
        }
    }
}
