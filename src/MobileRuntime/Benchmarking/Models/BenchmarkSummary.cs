using System;

namespace MobileRuntime.Benchmarking.Models;

public class BenchmarkSummary
{
    public TimeSpan TotalTime { get; set; }
    public long TotalMemoryBytes { get; set; }
    public double TotalEnergyJoules { get; set; }
    public int PassedCount { get; set; }
    public int FailedCount { get; set; }
}
