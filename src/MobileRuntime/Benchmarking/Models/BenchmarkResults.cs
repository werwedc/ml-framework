using System.Collections.Generic;

namespace MobileRuntime.Benchmarking.Models;

public class BenchmarkResults
{
    public string SuiteName { get; set; } = string.Empty;
    public List<BenchmarkResult> Results { get; set; } = new List<BenchmarkResult>();
    public BenchmarkSummary Summary { get; set; } = new BenchmarkSummary();
}
