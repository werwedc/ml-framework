using System;
using System.Collections.Generic;

namespace MobileRuntime.Benchmarking.Models;

public class BenchmarkResult
{
    public string Name { get; set; } = string.Empty;
    public int Iterations { get; set; }
    public TimeSpan MinTime { get; set; }
    public TimeSpan MaxTime { get; set; }
    public TimeSpan AverageTime { get; set; }
    public TimeSpan MedianTime { get; set; }
    public double StdDev { get; set; }
    public long MinMemoryBytes { get; set; }
    public long MaxMemoryBytes { get; set; }
    public long AverageMemoryBytes { get; set; }
    public double MinEnergyJoules { get; set; }
    public double MaxEnergyJoules { get; set; }
    public double AverageEnergyJoules { get; set; }
    public DateTime Timestamp { get; set; }
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}
