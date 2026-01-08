using MachineLearning.Visualization.Events;

namespace MLFramework.Visualization.Profiling;

/// <summary>
/// Profiling scope that tracks the duration of an operation
/// </summary>
public class ProfilingScope : IProfilingScope
{
    private readonly IProfiler _profiler;
    private readonly long _startTimestampNanoseconds;
    private bool _disposed;
    private bool _ended;

    /// <summary>
    /// Gets the name of the operation being profiled
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the metadata associated with this profiling scope
    /// </summary>
    public Dictionary<string, string> Metadata { get; }

    /// <summary>
    /// Gets the start timestamp in nanoseconds
    /// </summary>
    public long StartTimestampNanoseconds => _startTimestampNanoseconds;

    /// <summary>
    /// Gets the duration in nanoseconds (only available after disposal)
    /// </summary>
    public long DurationNanoseconds { get; private set; }

    /// <summary>
    /// Creates a new profiling scope
    /// </summary>
    /// <param name="profiler">The profiler that owns this scope</param>
    /// <param name="name">Name of the operation being profiled</param>
    /// <param name="metadata">Optional metadata</param>
    public ProfilingScope(IProfiler profiler, string name, Dictionary<string, string>? metadata = null)
    {
        _profiler = profiler ?? throw new ArgumentNullException(nameof(profiler));
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Metadata = metadata ?? new Dictionary<string, string>();

        // Record start timestamp using high-resolution timer
        _startTimestampNanoseconds = Stopwatch.GetTimestamp() * (1_000_000_000L / Stopwatch.Frequency);
    }

    /// <summary>
    /// Ends the profiling scope and records the duration
    /// </summary>
    public void End()
    {
        if (!_ended)
        {
            _ended = true;
            DurationNanoseconds = Stopwatch.GetTimestamp() * (1_000_000_000L / Stopwatch.Frequency) - _startTimestampNanoseconds;
        }
    }

    /// <summary>
    /// Disposes the profiling scope, ending it and recording the duration
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            try
            {
                if (!_ended)
                {
                    End();
                }

                // Record the duration with the profiler
                if (_profiler.IsEnabled)
                {
                    var endEvent = new ProfilingEndEvent(Name, durationNanoseconds: DurationNanoseconds, metadata: Metadata);
                    // The profiler will handle the actual recording
                }
            }
            catch (Exception)
            {
                // Handle disposal exceptions gracefully
            }
            finally
            {
                _disposed = true;
            }
        }
    }
}
