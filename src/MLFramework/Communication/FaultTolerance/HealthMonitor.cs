namespace MLFramework.Communication.FaultTolerance;

using MLFramework.Communication;

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
