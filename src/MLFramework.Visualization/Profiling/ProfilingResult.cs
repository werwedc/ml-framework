namespace MLFramework.Visualization.Profiling;

/// <summary>
/// Result of profiling operations with statistical summaries
/// </summary>
public class ProfilingResult
{
    /// <summary>
    /// Gets the name of the profiled operation
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the total duration in nanoseconds across all invocations
    /// </summary>
    public long TotalDurationNanoseconds { get; }

    /// <summary>
    /// Gets the number of times the operation was profiled
    /// </summary>
    public long Count { get; }

    /// <summary>
    /// Gets the minimum duration in nanoseconds
    /// </summary>
    public long MinDurationNanoseconds { get; }

    /// <summary>
    /// Gets the maximum duration in nanoseconds
    /// </summary>
    public long MaxDurationNanoseconds { get; }

    /// <summary>
    /// Gets the average duration in nanoseconds
    /// </summary>
    public double AverageDurationNanoseconds { get; }

    /// <summary>
    /// Gets the standard deviation in nanoseconds
    /// </summary>
    public double StdDevNanoseconds { get; }

    /// <summary>
    /// Gets the 50th percentile (median) in nanoseconds
    /// </summary>
    public long P50Nanoseconds { get; }

    /// <summary>
    /// Gets the 90th percentile in nanoseconds
    /// </summary>
    public long P90Nanoseconds { get; }

    /// <summary>
    /// Gets the 95th percentile in nanoseconds
    /// </summary>
    public long P95Nanoseconds { get; }

    /// <summary>
    /// Gets the 99th percentile in nanoseconds
    /// </summary>
    public long P99Nanoseconds { get; }

    /// <summary>
    /// Creates a new profiling result
    /// </summary>
    public ProfilingResult(
        string name,
        long totalDurationNanoseconds,
        long count,
        long minDurationNanoseconds,
        long maxDurationNanoseconds,
        double averageDurationNanoseconds,
        double stdDevNanoseconds,
        long p50Nanoseconds,
        long p90Nanoseconds,
        long p95Nanoseconds,
        long p99Nanoseconds)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        TotalDurationNanoseconds = totalDurationNanoseconds;
        Count = count;
        MinDurationNanoseconds = minDurationNanoseconds;
        MaxDurationNanoseconds = maxDurationNanoseconds;
        AverageDurationNanoseconds = averageDurationNanoseconds;
        StdDevNanoseconds = stdDevNanoseconds;
        P50Nanoseconds = p50Nanoseconds;
        P90Nanoseconds = p90Nanoseconds;
        P95Nanoseconds = p95Nanoseconds;
        P99Nanoseconds = p99Nanoseconds;
    }

    /// <summary>
    /// Returns a string representation of the profiling result
    /// </summary>
    public override string ToString()
    {
        return $"{Name}: Count={Count}, Avg={AverageDurationNanoseconds:F2}ns, Min={MinDurationNanoseconds}ns, Max={MaxDurationNanoseconds}ns, P95={P95Nanoseconds}ns";
    }
}
