namespace MLFramework.Visualization.Profiling;

/// <summary>
/// Interface for a profiling scope that tracks the duration of an operation
/// </summary>
public interface IProfilingScope : IDisposable
{
    /// <summary>
    /// Gets the name of the operation being profiled
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the metadata associated with this profiling scope
    /// </summary>
    Dictionary<string, string> Metadata { get; }

    /// <summary>
    /// Ends the profiling scope and records the duration
    /// </summary>
    void End();
}

/// <summary>
/// Interface for profiling operations to measure performance
/// </summary>
public interface IProfiler
{
    /// <summary>
    /// Starts profiling an operation
    /// </summary>
    /// <param name="name">Name of the operation to profile</param>
    /// <returns>A profiling scope that will record the duration when disposed</returns>
    IProfilingScope StartProfile(string name);

    /// <summary>
    /// Starts profiling an operation with metadata
    /// </summary>
    /// <param name="name">Name of the operation to profile</param>
    /// <param name="metadata">Additional metadata for this profiling operation</param>
    /// <returns>A profiling scope that will record the duration when disposed</returns>
    IProfilingScope StartProfile(string name, Dictionary<string, string> metadata);

    /// <summary>
    /// Records an instant event (e.g., a checkpoint or milestone)
    /// </summary>
    /// <param name="name">Name of the instant event</param>
    void RecordInstant(string name);

    /// <summary>
    /// Records an instant event with metadata
    /// </summary>
    /// <param name="name">Name of the instant event</param>
    /// <param name="metadata">Additional metadata for this event</param>
    void RecordInstant(string name, Dictionary<string, string> metadata);

    /// <summary>
    /// Gets whether profiling is enabled
    /// </summary>
    bool IsEnabled { get; set; }
}
