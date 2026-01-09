using System;
using System.Collections.Generic;

namespace MobileRuntime.Benchmarking.Models;

public class BenchmarkConfig
{
    public string Name { get; set; } = string.Empty;
    public Action? Benchmark { get; set; }
    public int Iterations { get; set; } = 10;
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}
