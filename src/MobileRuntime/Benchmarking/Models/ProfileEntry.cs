using System;

namespace MobileRuntime.Benchmarking.Models;

public class ProfileEntry
{
    public string Name { get; set; } = string.Empty;
    public int CallCount { get; set; }
    public TimeSpan TotalTime { get; set; }
    public TimeSpan MinTime { get; set; }
    public TimeSpan MaxTime { get; set; }
    public TimeSpan AverageTime { get; set; }
    public long TotalMemoryBytes { get; set; }
}
