using MobileRuntime.Benchmarking.Models;

using System.Collections.Generic;

namespace MobileRuntime.Benchmarking.Models;

public class ComparisonReport
{
    public BenchmarkResults Baseline { get; set; } = new BenchmarkResults();
    public BenchmarkResults Current { get; set; } = new BenchmarkResults();
    public List<ComparisonEntry> Entries { get; set; } = new List<ComparisonEntry>();
    public List<ComparisonEntry> Regressions { get; set; } = new List<ComparisonEntry>();
    public List<ComparisonEntry> Improvements { get; set; } = new List<ComparisonEntry>();
}
