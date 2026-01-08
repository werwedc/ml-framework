# Spec: Error Handling and Fault Tolerance

## Overview
Implement robust error handling, timeout mechanisms, and fault tolerance for distributed communication operations.

## Dependencies
- `spec_communication_interfaces.md`
- `spec_async_operations.md`

## Technical Requirements

### 1. Timeout Manager
Manage timeouts for communication operations.

```csharp
namespace MLFramework.Communication.FaultTolerance
{
    /// <summary>
    /// Manages timeouts for communication operations
    /// </summary>
    public class TimeoutManager : IDisposable
    {
        private readonly Dictionary<int, CancellationTokenSource> _operationTimeouts;
        private readonly object _lock;
        private readonly int _defaultTimeoutMs;
        private bool _disposed;

        /// <summary>
        /// Default timeout in milliseconds
        /// </summary>
        public int DefaultTimeoutMs => _defaultTimeoutMs;

        /// <summary>
        /// Create a timeout manager
        /// </summary>
        /// <param name="defaultTimeoutMs">Default timeout (default: 5 minutes)</param>
        public TimeoutManager(int defaultTimeoutMs = 300000)
        {
            _defaultTimeoutMs = defaultTimeoutMs;
            _operationTimeouts = new Dictionary<int, CancellationTokenSource>();
            _lock = new object();
        }

        /// <summary>
        /// Start a timeout for an operation
        /// </summary>
        /// <returns>Cancellation token for the operation</returns>
        public CancellationToken StartTimeout(int operationId, int? timeoutMs = null)
        {
            lock (_lock)
            {
                // Cancel existing timeout if any
                if (_operationTimeouts.ContainsKey(operationId))
                {
                    _operationTimeouts[operationId].Cancel();
                    _operationTimeouts[operationId].Dispose();
                }

                var cts = new CancellationTokenSource();
                var timeout = timeoutMs ?? _defaultTimeoutMs;

                if (timeout > 0)
                {
                    cts.CancelAfter(timeout);
                }

                _operationTimeouts[operationId] = cts;
                return cts.Token;
            }
        }

        /// <summary>
        /// Cancel timeout for an operation
        /// </summary>
        public void CancelTimeout(int operationId)
        {
            lock (_lock)
            {
                if (_operationTimeouts.TryGetValue(operationId, out var cts))
                {
                    cts.Cancel();
                    cts.Dispose();
                    _operationTimeouts.Remove(operationId);
                }
            }
        }

        /// <summary>
        /// Extend timeout for an operation
        /// </summary>
        public void ExtendTimeout(int operationId, int additionalTimeoutMs)
        {
            lock (_lock)
            {
                if (_operationTimeouts.TryGetValue(operationId, out var cts))
                {
                    if (!cts.Token.IsCancellationRequested)
                    {
                        cts.CancelAfter(additionalTimeoutMs);
                    }
                }
            }
        }

        /// <summary>
        /// Cancel all timeouts
        /// </summary>
        public void CancelAll()
        {
            lock (_lock)
            {
                foreach (var cts in _operationTimeouts.Values)
                {
                    cts.Cancel();
                    cts.Dispose();
                }
                _operationTimeouts.Clear();
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                CancelAll();
                _disposed = true;
            }
        }
    }
}
```

### 2. Error Recovery Manager
Handle and recover from communication errors.

```csharp
namespace MLFramework.Communication.FaultTolerance
{
    /// <summary>
    /// Error severity levels
    /// </summary>
    public enum ErrorSeverity
    {
        Warning,
        Recoverable,
        Fatal
    }

    /// <summary>
    /// Error information
    /// </summary>
    public class CommunicationError
    {
        public int OperationId { get; set; }
        public string OperationType { get; set; }
        public ErrorSeverity Severity { get; set; }
        public Exception Exception { get; set; }
        public DateTime Timestamp { get; set; }
        public int? Rank { get; set; }
        public Dictionary<string, string> Context { get; set; }

        public override string ToString()
        {
            return $"[{Timestamp:HH:mm:ss.fff}] {OperationType} (Rank: {Rank?.ToString() ?? "N/A"}): " +
                   $"{Severity} - {Exception.Message}";
        }
    }

    /// <summary>
    /// Error recovery strategy
    /// </summary>
    public enum RecoveryStrategy
    {
        Retry,
        FallbackToDifferentBackend,
        UseDifferentAlgorithm,
        Abort
    }

    /// <summary>
    /// Manages error recovery for communication operations
    /// </summary>
    public class ErrorRecoveryManager : IDisposable
    {
        private readonly List<CommunicationError> _errorHistory;
        private readonly Dictionary<Type, RecoveryStrategy> _recoveryStrategies;
        private readonly object _lock;
        private readonly int _maxRetries;
        private readonly TimeSpan _retryDelay;
        private bool _disposed;

        /// <summary>
        /// Maximum number of retries for recoverable errors
        /// </summary>
        public int MaxRetries => _maxRetries;

        /// <summary>
        /// Create an error recovery manager
        /// </summary>
        public ErrorRecoveryManager(int maxRetries = 3, TimeSpan? retryDelay = null)
        {
            _errorHistory = new List<CommunicationError>();
            _recoveryStrategies = new Dictionary<Type, RecoveryStrategy>();
            _lock = new object();
            _maxRetries = maxRetries;
            _retryDelay = retryDelay ?? TimeSpan.FromMilliseconds(100);

            // Set default recovery strategies
            SetDefaultStrategies();
        }

        private void SetDefaultStrategies()
        {
            _recoveryStrategies[typeof(CommunicationTimeoutException)] = RecoveryStrategy.Retry;
            _recoveryStrategies[typeof(RankMismatchException)] = RecoveryStrategy.Abort;
            _recoveryStrategies[typeof(IOException)] = RecoveryStrategy.Retry;
        }

        /// <summary>
        /// Set recovery strategy for a specific exception type
        /// </summary>
        public void SetRecoveryStrategy(Type exceptionType, RecoveryStrategy strategy)
        {
            lock (_lock)
            {
                _recoveryStrategies[exceptionType] = strategy;
            }
        }

        /// <summary>
        /// Handle a communication error
        /// </summary>
        /// <returns>True if operation should be retried, false if it should abort</returns>
        public bool HandleError(CommunicationError error)
        {
            lock (_lock)
            {
                _errorHistory.Add(error);

                // Log error
                LogError(error);

                // Determine recovery strategy
                var strategy = GetRecoveryStrategy(error.Exception.GetType());

                switch (strategy)
                {
                    case RecoveryStrategy.Retry:
                        return CanRetry(error);

                    case RecoveryStrategy.FallbackToDifferentBackend:
                    case RecoveryStrategy.UseDifferentAlgorithm:
                        // These would need additional context
                        return false;

                    case RecoveryStrategy.Abort:
                    default:
                        return false;
                }
            }
        }

        private bool CanRetry(CommunicationError error)
        {
            // Check if this operation has been retried too many times
            var retryCount = _errorHistory.Count(e =>
                e.OperationId == error.OperationId &&
                e.OperationType == error.OperationType);

            return retryCount <= _maxRetries;
        }

        private RecoveryStrategy GetRecoveryStrategy(Type exceptionType)
        {
            // Look for exact match
            if (_recoveryStrategies.TryGetValue(exceptionType, out var strategy))
            {
                return strategy;
            }

            // Look for base type match
            foreach (var kvp in _recoveryStrategies)
            {
                if (kvp.Key.IsAssignableFrom(exceptionType))
                {
                    return kvp.Value;
                }
            }

            return RecoveryStrategy.Abort;
        }

        private void LogError(CommunicationError error)
        {
            Console.WriteLine($"[ERROR] {error}");
        }

        /// <summary>
        /// Get error history
        /// </summary>
        public IReadOnlyList<CommunicationError> GetErrorHistory()
        {
            lock (_lock)
            {
                return _errorHistory.ToList();
            }
        }

        /// <summary>
        /// Get error statistics
        /// </summary>
        public ErrorStatistics GetStatistics()
        {
            lock (_lock)
            {
                return new ErrorStatistics
                {
                    TotalErrors = _errorHistory.Count,
                    ErrorsByType = _errorHistory
                        .GroupBy(e => e.Exception.GetType().Name)
                        .ToDictionary(g => g.Key, g => g.Count()),
                    ErrorsBySeverity = _errorHistory
                        .GroupBy(e => e.Severity)
                        .ToDictionary(g => g.Key, g => g.Count()),
                    ErrorsByOperation = _errorHistory
                        .GroupBy(e => e.OperationType)
                        .ToDictionary(g => g.Key, g => g.Count())
                };
            }
        }

        /// <summary>
        /// Clear error history
        /// </summary>
        public void ClearHistory()
        {
            lock (_lock)
            {
                _errorHistory.Clear();
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                lock (_lock)
                {
                    _errorHistory.Clear();
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Error statistics
    /// </summary>
    public class ErrorStatistics
    {
        public int TotalErrors { get; set; }
        public Dictionary<string, int> ErrorsByType { get; set; }
        public Dictionary<ErrorSeverity, int> ErrorsBySeverity { get; set; }
        public Dictionary<string, int> ErrorsByOperation { get; set; }

        public override string ToString()
        {
            return $"Total Errors: {TotalErrors}, " +
                   $"By Type: {string.Join(", ", ErrorsByType.Select(kvp => $"{kvp.Key}={kvp.Value}"))}, " +
                   $"By Severity: {string.Join(", ", ErrorsBySeverity.Select(kvp => $"{kvp.Key}={kvp.Value}"))}";
        }
    }
}
```

### 3. Fault Tolerant Communication Wrapper
Wrapper that adds fault tolerance to communication operations.

```csharp
namespace MLFramework.Communication.FaultTolerance
{
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
            _errorRecoveryManager = new ErrorRecoveryManager();
            _innerAsyncBackend = backend as IAsyncCommunicationBackend;
            _nextOperationId = 1;
        }

        public void Broadcast<T>(Tensor<T> tensor, int rootRank)
        {
            ExecuteWithRetry("Broadcast", () => _innerBackend.Broadcast(tensor, rootRank));
        }

        public Tensor<T> Reduce<T>(Tensor<T> tensor, ReduceOp operation, int rootRank)
        {
            return ExecuteWithRetry("Reduce", () => _innerBackend.Reduce(tensor, operation, rootRank));
        }

        public Tensor<T> AllReduce<T>(Tensor<T> tensor, ReduceOp operation)
        {
            return ExecuteWithRetry("AllReduce", () => _innerBackend.AllReduce(tensor, operation));
        }

        public Tensor<T> AllGather<T>(Tensor<T> tensor)
        {
            return ExecuteWithRetry("AllGather", () => _innerBackend.AllGather(tensor));
        }

        public Tensor<T> ReduceScatter<T>(Tensor<T> tensor, ReduceOp operation)
        {
            return ExecuteWithRetry("ReduceScatter", () => _innerBackend.ReduceScatter(tensor, operation));
        }

        public void Barrier()
        {
            ExecuteWithRetry("Barrier", () => _innerBackend.Barrier());
        }

        private T ExecuteWithRetry<T>(string operationType, Func<T> func)
        {
            int operationId = Interlocked.Increment(ref _nextOperationId);
            int retryCount = 0;

            while (retryCount <= _errorRecoveryManager.MaxRetries)
            {
                try
                {
                    using var cts = _timeoutManager.StartTimeout(operationId);
                    return func();
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
                    Thread.Sleep((int)_errorRecoveryManager._retryDelay.TotalMilliseconds);
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
        public ICommunicationHandle BroadcastAsync<T>(Tensor<T> tensor, int rootRank)
        {
            if (_innerAsyncBackend == null)
                throw new NotSupportedException("Backend does not support async operations");

            return ExecuteWithRetryAsync("BroadcastAsync", () => _innerAsyncBackend.BroadcastAsync(tensor, rootRank));
        }

        public ICommunicationHandle AllReduceAsync<T>(Tensor<T> tensor, ReduceOp operation)
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
}
```

### 4. Health Check and Heartbeat
Monitor health of communication channels.

```csharp
namespace MLFramework.Communication.FaultTolerance
{
    /// <summary>
    /// Health status of a rank
    /// </summary>
    public enum RankHealthStatus
    {
        Healthy,
        Unresponsive,
        Failed
    }

    /// <summary>
    /// Monitors health of ranks and communication channels
    /// </summary>
    public class HealthMonitor : IDisposable
    {
        private readonly ICommunicationBackend _backend;
        private readonly Dictionary<int, RankHealthStatus> _rankStatus;
        private readonly Dictionary<int, DateTime> _lastHeartbeat;
        private readonly object _lock;
        private readonly TimeSpan _heartbeatTimeout;
        private readonly CancellationTokenSource _cts;
        private Task? _heartbeatTask;
        private bool _disposed;

        public int UnresponsiveRanksCount
        {
            get
            {
                lock (_lock)
                {
                    return _rankStatus.Count(kvp => kvp.Value != RankHealthStatus.Healthy);
                }
            }
        }

        public HealthMonitor(
            ICommunicationBackend backend,
            TimeSpan? heartbeatTimeout = null)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _heartbeatTimeout = heartbeatTimeout ?? TimeSpan.FromSeconds(30);
            _rankStatus = new Dictionary<int, RankHealthStatus>();
            _lastHeartbeat = new Dictionary<int, DateTime>();
            _lock = new object();
            _cts = new CancellationTokenSource();

            // Initialize all ranks as healthy
            for (int i = 0; i < backend.WorldSize; i++)
            {
                _rankStatus[i] = RankHealthStatus.Healthy;
                _lastHeartbeat[i] = DateTime.Now;
            }
        }

        /// <summary>
        /// Start health monitoring
        /// </summary>
        public void StartMonitoring()
        {
            _heartbeatTask = Task.Run(MonitorHealthAsync, _cts.Token);
        }

        /// <summary>
        /// Stop health monitoring
        /// </summary>
        public void StopMonitoring()
        {
            _cts.Cancel();
            _heartbeatTask?.Wait();
            _heartbeatTask?.Dispose();
        }

        private async Task MonitorHealthAsync()
        {
            while (!_cts.Token.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(TimeSpan.FromSeconds(5), _cts.Token);
                    CheckHealth();
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[HealthMonitor] Error: {ex.Message}");
                }
            }
        }

        private void CheckHealth()
        {
            lock (_lock)
            {
                var now = DateTime.Now;

                foreach (var kvp in _lastHeartbeat.ToList())
                {
                    var rank = kvp.Key;
                    var lastHeartbeat = kvp.Value;

                    if (now - lastHeartbeat > _heartbeatTimeout * 2)
                    {
                        _rankStatus[rank] = RankHealthStatus.Failed;
                        Console.WriteLine($"[HealthMonitor] Rank {rank} is marked as FAILED");
                    }
                    else if (now - lastHeartbeat > _heartbeatTimeout)
                    {
                        _rankStatus[rank] = RankHealthStatus.Unresponsive;
                        Console.WriteLine($"[HealthMonitor] Rank {rank} is marked as UNRESPONSIVE");
                    }
                    else
                    {
                        _rankStatus[rank] = RankHealthStatus.Healthy;
                    }
                }
            }
        }

        /// <summary>
        /// Update heartbeat for a rank
        /// </summary>
        public void UpdateHeartbeat(int rank)
        {
            lock (_lock)
            {
                if (_rankStatus.ContainsKey(rank))
                {
                    _lastHeartbeat[rank] = DateTime.Now;
                    _rankStatus[rank] = RankHealthStatus.Healthy;
                }
            }
        }

        /// <summary>
        /// Get health status of a rank
        /// </summary>
        public RankHealthStatus GetRankHealthStatus(int rank)
        {
            lock (_lock)
            {
                return _rankStatus.TryGetValue(rank, out var status) ? status : RankHealthStatus.Failed;
            }
        }

        /// <summary>
        /// Get all healthy ranks
        /// </summary>
        public List<int> GetHealthyRanks()
        {
            lock (_lock)
            {
                return _rankStatus
                    .Where(kvp => kvp.Value == RankHealthStatus.Healthy)
                    .Select(kvp => kvp.Key)
                    .ToList();
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                StopMonitoring();
                _cts.Dispose();
                _disposed = true;
            }
        }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/FaultTolerance/TimeoutManager.cs`
   - `src/MLFramework/Communication/FaultTolerance/CommunicationError.cs`
   - `src/MLFramework/Communication/FaultTolerance/ErrorRecoveryManager.cs`
   - `src/MLFramework/Communication/FaultTolerance/FaultTolerantCommunication.cs`
   - `src/MLFramework/Communication/FaultTolerance/HealthMonitor.cs`

2. **Design Decisions:**
   - Timeout manager uses cancellation tokens
   - Error recovery manager tracks history and statistics
   - Fault-tolerant wrapper adds retries to all operations
   - Health monitor checks rank status periodically

3. **Error Handling:**
   - Graceful degradation on network failures
   - Automatic retries for transient errors
   - Health monitoring for rank failures
   - Detailed error logging and statistics

4. **Performance Considerations:**
   - Minimize overhead in happy path
   - Efficient data structures for tracking
   - Async health monitoring
   - Configurable retry delays

## Testing Requirements
- Tests for timeout manager
- Tests for error recovery strategies
- Tests for fault-tolerant wrapper with mock backend
- Tests for health monitor
- Tests for graceful degradation

## Success Criteria
- Timeout manager correctly cancels operations
- Error recovery manager handles all error types
- Fault-tolerant wrapper retries failed operations
- Health monitor correctly tracks rank status
- All components integrate correctly
