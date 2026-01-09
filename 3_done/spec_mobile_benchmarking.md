# Spec: Benchmarking and Profiling Utilities

## Overview
Implement benchmarking and profiling utilities for performance measurement and optimization of the mobile runtime.

## Requirements
- Measure inference latency
- Profile operator execution times
- Memory usage tracking
- Energy efficiency measurement (mobile)
- Comparison across backends
- Performance regression detection
- Export results to various formats

## Classes to Implement

### 1. `BenchmarkRunner` Class
```csharp
public class BenchmarkRunner : IDisposable
{
    private readonly List<BenchmarkResult> _results;
    private readonly Stopwatch _stopwatch;
    private readonly IMemoryMonitor _memoryMonitor;
    private readonly IEnergyMonitor _energyMonitor;

    public BenchmarkRunner(IMemoryMonitor memoryMonitor = null, IEnergyMonitor energyMonitor = null);

    public BenchmarkResult RunBenchmark(string name, Action benchmark, int iterations = 10);
    public BenchmarkResult RunBenchmark<T>(string name, Func<T> benchmark, int iterations = 10);
    public BenchmarkResults RunBenchmarkSuite(string suiteName, params BenchmarkConfig[] benchmarks);

    public BenchmarkResults GetResults();
    public void Reset();
    public void ExportResults(string filePath, ReportFormat format = ReportFormat.Json);

    public void Dispose();
}

public class BenchmarkConfig
{
    public string Name { get; set; }
    public Action Benchmark { get; set; }
    public int Iterations { get; set; } = 10;
    public Dictionary<string, string> Metadata { get; set; }
}

public class BenchmarkResult
{
    public string Name { get; set; }
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
    public Dictionary<string, string> Metadata { get; set; }
}

public class BenchmarkResults
{
    public string SuiteName { get; set; }
    public List<BenchmarkResult> Results { get; set; }
    public BenchmarkSummary Summary { get; set; }
}

public class BenchmarkSummary
{
    public TimeSpan TotalTime { get; set; }
    public long TotalMemoryBytes { get; set; }
    public double TotalEnergyJoules { get; set; }
    public int PassedCount { get; set; }
    public int FailedCount { get; set; }
}
```

### 2. `Profiler` Class
```csharp
public class Profiler : IDisposable
{
    private readonly Dictionary<string, ProfileEntry> _profiles;
    private readonly Stack<ProfileScope> _activeScopes;
    private readonly Stopwatch _stopwatch;

    public Profiler();

    public ProfileScope BeginScope(string name);
    public void EndScope();
    public T Profile<T>(string name, Func<T> action);
    public void Profile(string name, Action action);

    public ProfileReport GetReport();
    public void Reset();
    public void ExportReport(string filePath, ReportFormat format = ReportFormat.Json);
    public void PrintReport();

    public void Dispose();
}

public class ProfileScope : IDisposable
{
    private readonly Profiler _profiler;
    private readonly string _name;
    private readonly Stopwatch _stopwatch;

    internal ProfileScope(Profiler profiler, string name);

    public void Dispose();
}

public class ProfileEntry
{
    public string Name { get; set; }
    public int CallCount { get; set; }
    public TimeSpan TotalTime { get; set; }
    public TimeSpan MinTime { get; set; }
    public TimeSpan MaxTime { get; set; }
    public TimeSpan AverageTime { get; set; }
    public long TotalMemoryBytes { get; set; }
}

public class ProfileReport
{
    public List<ProfileEntry> Entries { get; set; }
    public TimeSpan TotalTime { get; set; }
    public DateTime Timestamp { get; set; }
}
```

### 3. `IMemoryMonitor` Interface and Implementation

```csharp
public interface IMemoryMonitor
{
    void StartMonitoring();
    void StopMonitoring();
    MemorySnapshot GetSnapshot();
    void Reset();
}

public class MemoryMonitor : IMemoryMonitor
{
    private readonly Timer _monitorTimer;
    private readonly List<MemorySnapshot> _snapshots;
    private bool _isMonitoring;

    public MemoryMonitor(int sampleIntervalMs = 100);

    public void StartMonitoring();
    public void StopMonitoring();
    public MemorySnapshot GetSnapshot();
    public void Reset();

    private void SampleMemory();
}

public class MemorySnapshot
{
    public long WorkingSetBytes { get; set; }
    public long PrivateMemoryBytes { get; set; }
    public long GCMemoryBytes { get; set; }
    public long Gen0Collections { get; set; }
    public long Gen1Collections { get; set; }
    public long Gen2Collections { get; set; }
    public DateTime Timestamp { get; set; }
}
```

### 4. `IEnergyMonitor` Interface and Implementation

```csharp
public interface IEnergyMonitor
{
    void StartMonitoring();
    void StopMonitoring();
    EnergySnapshot GetSnapshot();
    void Reset();
}

public class EnergyMonitor : IEnergyMonitor
{
    private readonly Timer _monitorTimer;
    private readonly List<EnergySnapshot> _snapshots;
    private bool _isMonitoring;

    public EnergyMonitor(int sampleIntervalMs = 100);

    public void StartMonitoring();
    public void StopMonitoring();
    public EnergySnapshot GetSnapshot();
    public void Reset();

    private void SampleEnergy();
}

public class EnergySnapshot
{
    public double EnergyJoules { get; set; }
    public double PowerWatts { get; set; }
    public double VoltageVolts { get; set; }
    public double CurrentAmperes { get; set; }
    public DateTime Timestamp { get; set; }
}
```

### 5. `PerformanceComparator` Class

```csharp
public class PerformanceComparator
{
    public ComparisonReport Compare(BenchmarkResults baseline, BenchmarkResults current);
    public ComparisonReport CompareProfiles(ProfileReport baseline, ProfileReport current);

    private ComparisonEntry CompareEntries(BenchmarkResult baseline, BenchmarkResult current);
    private bool IsRegression(BenchmarkResult baseline, BenchmarkResult current, double thresholdPercent);
}

public class ComparisonReport
{
    public BenchmarkResults Baseline { get; set; }
    public BenchmarkResults Current { get; set; }
    public List<ComparisonEntry> Entries { get; set; }
    public List<ComparisonEntry> Regressions { get; set; }
    public List<ComparisonEntry> Improvements { get; set; }
}

public class ComparisonEntry
{
    public string Name { get; set; }
    public double BaselineTimeMs { get; set; }
    public double CurrentTimeMs { get; set; }
    public double TimeChangePercent { get; set; }
    public double BaselineMemoryMB { get; set; }
    public double CurrentMemoryMB { get; set; }
    public double MemoryChangePercent { get; set; }
    public bool IsRegression { get; set; }
}
```

### 6. `BenchmarkExporter` Class

```csharp
public static class BenchmarkExporter
{
    public static void ExportToJson(BenchmarkResults results, string filePath);
    public static void ExportToCsv(BenchmarkResults results, string filePath);
    public static void ExportToMarkdown(BenchmarkResults results, string filePath);
    public static void ExportToHtml(BenchmarkResults results, string filePath);

    private static void CreateHtmlReport(BenchmarkResults results, string filePath);
}
```

### 7. `InferenceBenchmark` Helper Class

```csharp
public class InferenceBenchmark
{
    public static BenchmarkResult BenchmarkModel(
        IModel model,
        ITensor[] inputs,
        int iterations = 10,
        string name = "Inference"
    );

    public static BenchmarkResult BenchmarkOperator(
        IBackend backend,
        OperatorDescriptor op,
        ITensor[] inputs,
        int iterations = 100,
        string name = "Operator"
    );

    public static BenchmarkResults BenchmarkAllOperators(
        IBackend backend,
        Dictionary<OperatorType, OperatorDescriptor> operators,
        Dictionary<OperatorType, ITensor[]> inputs
    );
}
```

## Implementation Notes

### Timing
- Use `Stopwatch` with high-resolution timing
- Use `QueryPerformanceCounter` on Windows for maximum precision
- Measure both CPU time and wall clock time
- Exclude warmup iterations from results

### Memory Monitoring
- Use `Process.GetCurrentProcess()` for process memory
- Use `GC.GetTotalMemory()` for managed memory
- Sample at regular intervals (e.g., 100ms)
- Track peaks and averages

### Energy Monitoring
- Platform-specific implementations
- Android: BatteryStats API
- iOS: IOKit framework
- Windows: Performance counters
- Desktop platforms: Use power meters if available

### Statistical Analysis
- Calculate median for robust timing
- Calculate standard deviation for consistency
- Exclude outliers (e.g., > 3σ)
- Use geometric mean for multiple metrics

## Usage Examples

### Benchmark model inference
```csharp
using (var runner = new BenchmarkRunner())
{
    var model = runtime.LoadModel("model.mob");
    var inputs = new[] { tensorFactory.CreateTensor(data, shape) };

    var result = runner.RunBenchmark("Model Inference", () =>
    {
        model.Predict(inputs);
    }, iterations: 100);

    Console.WriteLine($"Average: {result.AverageTime.TotalMilliseconds}ms");
    Console.WriteLine($"Min: {result.MinTime.TotalMilliseconds}ms");
    Console.WriteLine($"Max: {result.MaxTime.TotalMilliseconds}ms");
}
```

### Profile operator execution
```csharp
using (var profiler = new Profiler())
{
    profiler.Profile("Conv2D", () =>
    {
        backend.Execute(convOp, inputs, parameters);
    });

    profiler.Profile("Relu", () =>
    {
        backend.Execute(reluOp, inputs, parameters);
    });

    profiler.PrintReport();
}
```

### Compare backends
```csharp
var cpuBackend = new CpuBackend(memoryPool, tensorFactory);
var gpuBackend = new MetalBackend(tensorFactory);

var cpuResult = InferenceBenchmark.BenchmarkModel(model, inputs, 100, "CPU");
var gpuResult = InferenceBenchmark.BenchmarkModel(model, inputs, 100, "GPU");

Console.WriteLine($"CPU: {cpuResult.AverageTime.TotalMilliseconds}ms");
Console.WriteLine($"GPU: {gpuResult.AverageTime.TotalMilliseconds}ms");
Console.WriteLine($"Speedup: {cpuResult.AverageTime / gpuResult.AverageTime:F2}x");
```

### Export results
```csharp
var results = runner.RunBenchmarkSuite("Model Suite", benchmarks);
runner.ExportResults("results.json", ReportFormat.Json);
runner.ExportResults("results.csv", ReportFormat.Csv);
runner.ExportResults("results.html", ReportFormat.Html);
```

## File Structure

```
src/MobileRuntime/Benchmarking/
├── BenchmarkRunner.cs
├── Profiler.cs
├── Interfaces/
│   ├── IMemoryMonitor.cs
│   └── IEnergyMonitor.cs
├── Monitoring/
│   ├── MemoryMonitor.cs
│   └── EnergyMonitor.cs
├── Comparison/
│   ├── PerformanceComparator.cs
│   └── ComparisonReport.cs
├── Export/
│   └── BenchmarkExporter.cs
├── Helpers/
│   └── InferenceBenchmark.cs
└── Models/
    ├── BenchmarkConfig.cs
    ├── BenchmarkResult.cs
    ├── BenchmarkResults.cs
    ├── BenchmarkSummary.cs
    ├── ProfileEntry.cs
    ├── ProfileReport.cs
    ├── ProfileScope.cs
    ├── MemorySnapshot.cs
    └── EnergySnapshot.cs
```

## Success Criteria

- All benchmarks run successfully
- Timing measurements are accurate
- Memory monitoring works on all platforms
- Energy monitoring works on mobile platforms
- Reports are correctly generated
- Performance regressions are detected

## Dependencies

- spec_mobile_runtime_core (interfaces)
- spec_mobile_model_loader (IModel)
- spec_mobile_backend_cpu/metal/vulkan (IBackend)
- System.Diagnostics (Stopwatch)
- System.IO (file export)
- System.Text.Json (JSON export)

## Testing Requirements

- Unit tests for benchmark runner
- Unit tests for profiler
- Unit tests for memory monitor
- Integration tests with real models
- Cross-platform tests
- Accuracy verification tests

## Performance Targets

- Benchmark overhead: < 1%
- Timing precision: < 100μs
- Memory sampling overhead: < 5%
- Energy sampling overhead: < 10%
- Report generation: < 1s for 1000 entries

## Platform Notes

- Windows: Use `QueryPerformanceCounter` for high-precision timing
- Linux: Use `clock_gettime(CLOCK_MONOTONIC)`
- macOS: Use `mach_absolute_time`
- Android: BatteryStats API for energy monitoring
- iOS: IOKit framework for energy monitoring
