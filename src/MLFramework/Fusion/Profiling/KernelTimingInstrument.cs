using MLFramework.Fusion.Backends;

namespace MLFramework.Fusion.Profiling;

/// <summary>
/// Interface for instrumenting kernel timing
/// </summary>
public interface IKernelTimingInstrument
{
    /// <summary>
    /// Records kernel launch and completion
    /// </summary>
    void RecordKernelLaunch(string kernelName, KernelLaunchConfiguration config);

    /// <summary>
    /// Records kernel completion
    /// </summary>
    void RecordKernelComplete(string kernelName, double durationMs);

    /// <summary>
    /// Gets timing statistics for a kernel
    /// </summary>
    KernelTimingStatistics GetTimingStatistics(string kernelName);
}

/// <summary>
/// Timing statistics for a kernel
/// </summary>
public record KernelTimingStatistics
{
    public required string KernelName { get; init; }
    public required int ExecutionCount { get; init; }
    public required double TotalTimeMs { get; init; }
    public required double AverageTimeMs { get; init; }
    public required double MinTimeMs { get; init; }
    public required double MaxTimeMs { get; init; }
    public required double StdDevMs { get; init; }
}

/// <summary>
/// Implementation of kernel timing instrumentation
/// </summary>
public class KernelTimingInstrument : IKernelTimingInstrument
{
    private readonly Dictionary<string, List<double>> _timingRecords = new();
    private readonly Dictionary<string, KernelLaunchConfiguration> _launchConfigs = new();
    private readonly object _lock = new();

    public void RecordKernelLaunch(string kernelName, KernelLaunchConfiguration config)
    {
        lock (_lock)
        {
            _launchConfigs[kernelName] = config;
        }
    }

    public void RecordKernelComplete(string kernelName, double durationMs)
    {
        lock (_lock)
        {
            if (!_timingRecords.ContainsKey(kernelName))
            {
                _timingRecords[kernelName] = new List<double>();
            }
            _timingRecords[kernelName].Add(durationMs);
        }
    }

    public KernelTimingStatistics GetTimingStatistics(string kernelName)
    {
        lock (_lock)
        {
            if (!_timingRecords.TryGetValue(kernelName, out var timings))
            {
                return new KernelTimingStatistics
                {
                    KernelName = kernelName,
                    ExecutionCount = 0,
                    TotalTimeMs = 0,
                    AverageTimeMs = 0,
                    MinTimeMs = 0,
                    MaxTimeMs = 0,
                    StdDevMs = 0
                };
            }

            var count = timings.Count;
            var total = timings.Sum();
            var average = total / count;
            var min = timings.Min();
            var max = timings.Max();
            var variance = timings.Sum(t => Math.Pow(t - average, 2)) / count;
            var stdDev = Math.Sqrt(variance);

            return new KernelTimingStatistics
            {
                KernelName = kernelName,
                ExecutionCount = count,
                TotalTimeMs = total,
                AverageTimeMs = average,
                MinTimeMs = min,
                MaxTimeMs = max,
                StdDevMs = stdDev
            };
        }
    }
}
