namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Configuration for memory profiling
/// </summary>
public class MemoryProfilingConfiguration
{
    /// <summary>
    /// Enable memory profiling
    /// </summary>
    public bool EnableMemoryProfiling { get; set; } = false;

    /// <summary>
    /// Capture stack traces for allocations
    /// </summary>
    public bool CaptureStackTraces { get; set; } = false;

    /// <summary>
    /// Maximum depth of stack traces
    /// </summary>
    public int MaxStackTraceDepth { get; set; } = 10;

    /// <summary>
    /// Interval between memory snapshots (in milliseconds)
    /// </summary>
    public int SnapshotIntervalMs { get; set; } = 1000;

    /// <summary>
    /// Enable automatic memory snapshots
    /// </summary>
    public bool AutoSnapshot { get; set; } = true;
}
