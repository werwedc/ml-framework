namespace MLFramework.Communication.FaultTolerance;

using MLFramework.Distributed.Communication;

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
    /// Delay between retries
    /// </summary>
    public TimeSpan RetryDelay => _retryDelay;

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
