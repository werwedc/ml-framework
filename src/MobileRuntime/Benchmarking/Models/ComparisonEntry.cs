using MobileRuntime.Benchmarking.Models;

using System;

namespace MobileRuntime.Benchmarking.Models;

public class ComparisonEntry
{
    public string Name { get; set; } = string.Empty;
    public double BaselineTimeMs { get; set; }
    public double CurrentTimeMs { get; set; }
    public double TimeChangePercent { get; set; }
    public double BaselineMemoryMB { get; set; }
    public double CurrentMemoryMB { get; set; }
    public double MemoryChangePercent { get; set; }
    public bool IsRegression { get; set; }
}
