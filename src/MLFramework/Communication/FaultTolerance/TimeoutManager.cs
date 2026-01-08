namespace MLFramework.Communication.FaultTolerance;

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
