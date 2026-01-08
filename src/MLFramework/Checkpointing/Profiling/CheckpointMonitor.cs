namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Real-time monitoring for checkpointing operations
/// </summary>
public class CheckpointMonitor : IDisposable
{
    /// <summary>
    /// Event raised when memory usage exceeds threshold
    /// </summary>
    public event EventHandler<MemoryThresholdEventArgs>? MemoryThresholdExceeded;

    /// <summary>
    /// Event raised when recomputation time exceeds threshold
    /// </summary>
    public event EventHandler<RecomputationThresholdEventArgs>? RecomputationThresholdExceeded;

    private readonly CheckpointProfiler _profiler;
    private readonly System.Threading.Timer _monitorTimer;
    private readonly long _memoryThresholdBytes;
    private readonly long _recomputationThresholdMs;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of CheckpointMonitor
    /// </summary>
    /// <param name="profiler">Profiler to monitor</param>
    /// <param name="memoryThresholdBytes">Memory threshold in bytes</param>
    /// <param name="recomputationThresholdMs">Recomputation threshold in milliseconds</param>
    /// <param name="monitorIntervalMs">Monitoring interval in milliseconds</param>
    public CheckpointMonitor(
        CheckpointProfiler profiler,
        long memoryThresholdBytes = 1024 * 1024 * 1024, // 1GB default
        long recomputeThresholdMs = 1000, // 1 second default
        long monitorIntervalMs = 1000) // Check every second
    {
        _profiler = profiler ?? throw new ArgumentNullException(nameof(profiler));
        _memoryThresholdBytes = memoryThresholdBytes;
        _recomputationThresholdMs = recomputeThresholdMs;
        _monitorTimer = new System.Threading.Timer(CheckThresholds, null, monitorIntervalMs, monitorIntervalMs);
        _disposed = false;
    }

    private void CheckThresholds(object? state)
    {
        var summary = _profiler.GetSummary();

        // Check memory threshold
        if (summary.TotalMemorySaved > _memoryThresholdBytes)
        {
            MemoryThresholdExceeded?.Invoke(this, new MemoryThresholdEventArgs
            {
                CurrentMemoryUsage = summary.TotalMemorySaved,
                Threshold = _memoryThresholdBytes,
                Timestamp = DateTime.UtcNow
            });
        }

        // Check recomputation threshold
        if (summary.TotalRecomputeTime > _recomputationThresholdMs)
        {
            RecomputationThresholdExceeded?.Invoke(this, new RecomputationThresholdEventArgs
            {
                CurrentRecomputeTime = summary.TotalRecomputeTime,
                Threshold = _recomputationThresholdMs,
                Timestamp = DateTime.UtcNow
            });
        }
    }

    /// <summary>
    /// Disposes the monitor
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _monitorTimer.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Event arguments for memory threshold exceeded
/// </summary>
public class MemoryThresholdEventArgs : EventArgs
{
    public long CurrentMemoryUsage { get; set; }
    public long Threshold { get; set; }
    public DateTime Timestamp { get; set; }
}

/// <summary>
/// Event arguments for recomputation threshold exceeded
/// </summary>
public class RecomputationThresholdEventArgs : EventArgs
{
    public long CurrentRecomputeTime { get; set; }
    public long Threshold { get; set; }
    public DateTime Timestamp { get; set; }
}
