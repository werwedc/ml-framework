namespace MLFramework.Visualization.Profiling;

/// <summary>
/// Type of profiling event
/// </summary>
public enum ProfilingEventType
{
    /// <summary>
    /// Operation started
    /// </summary>
    Start,

    /// <summary>
    /// Operation ended
    /// </summary>
    End,

    /// <summary>
    /// Instantaneous event (e.g., checkpoint)
    /// </summary>
    Instant
}
