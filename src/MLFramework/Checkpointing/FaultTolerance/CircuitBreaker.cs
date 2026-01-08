namespace MachineLearning.Checkpointing.FaultTolerance;

/// <summary>
/// Circuit breaker states
/// </summary>
public enum CircuitState
{
    /// <summary>
    /// Circuit is closed, operations are allowed
    /// </summary>
    Closed,

    /// <summary>
    /// Circuit is open, operations are blocked
    /// </summary>
    Open,

    /// <summary>
    /// Circuit is half-open, testing if service has recovered
    /// </summary>
    HalfOpen
}

/// <summary>
/// Exception thrown when circuit breaker is open
/// </summary>
public class CircuitBreakerOpenException : Exception
{
    public CircuitBreakerOpenException(string message) : base(message)
    {
    }
}

/// <summary>
/// Circuit breaker pattern implementation
/// </summary>
public class CircuitBreaker
{
    private readonly int _failureThreshold;
    private readonly TimeSpan _recoveryTimeout;
    private int _failureCount = 0;
    private DateTime? _lastFailureTime = null;
    private CircuitState _state = CircuitState.Closed;

    /// <summary>
    /// Create a new CircuitBreaker
    /// </summary>
    public CircuitBreaker(int failureThreshold = 5, TimeSpan? recoveryTimeout = null)
    {
        _failureThreshold = failureThreshold;
        _recoveryTimeout = recoveryTimeout ?? TimeSpan.FromMinutes(1);
    }

    /// <summary>
    /// Current circuit state
    /// </summary>
    public CircuitState State => _state;

    /// <summary>
    /// Execute an operation through the circuit breaker
    /// </summary>
    public async Task<T> ExecuteAsync<T>(
        Func<Task<T>> operation,
        Func<Task<T>>? fallback = null)
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
        catch (Exception)
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
}
