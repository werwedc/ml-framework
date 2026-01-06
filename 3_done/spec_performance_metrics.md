# Spec: Performance Profiling and Metrics

## Overview
Add performance monitoring and profiling capabilities to the data loading pipeline.

## Requirements

### Metrics to Track

#### 1. Timing Metrics
- Batch load time
- Transform time
- Collate time
- GPU transfer time
- End-to-end batch time

#### 2. Throughput Metrics
- Samples per second
- Batches per second
- Throughput per worker

#### 3. Resource Metrics
- Queue depths
- Memory usage
- CPU utilization (per worker)
- GPU utilization

#### 4. Latency Metrics
- Time to first batch
- Average batch latency
- P99/P99.9 latency

### Implementation

#### DataLoadingMetrics
- Central metrics collector
- Thread-safe recording of metrics
- Aggregation and reporting

**Key Fields:**
```csharp
public class DataLoadingMetrics
{
    private readonly ConcurrentQueue<TimingRecord> _timingRecords;
    private readonly ConcurrentDictionary<string, PerformanceCounter> _counters;
    private readonly Stopwatch _epochTimer;
    private volatile bool _enabled;
}
```

**Classes:**
```csharp
public class TimingRecord
{
    public string MetricName { get; set; }
    public TimeSpan Duration { get; set; }
    public DateTime Timestamp { get; set; }
    public int WorkerId { get; set; }
}

public class PerformanceCounter
{
    private long _count;
    private double _sum;
    private double _min = double.MaxValue;
    private double _max = double.MinValue;

    public void Record(double value)
    {
        Interlocked.Increment(ref _count);
        Interlocked.Add(ref _sum, (long)value);

        // Update min/max (not perfectly atomic but acceptable)
        if (value < _min)
            _min = value;
        if (value > _max)
            _max = value;
    }

    public long Count => _count;
    public double Sum => _sum;
    public double Average => _count > 0 ? _sum / _count : 0;
    public double Min => _min;
    public double Max => _max;
}
```

**Constructor:**
```csharp
public DataLoadingMetrics(bool enabled = true)
{
    _timingRecords = new ConcurrentQueue<TimingRecord>();
    _counters = new ConcurrentDictionary<string, PerformanceCounter>();
    _epochTimer = new Stopwatch();
    _enabled = enabled;
}
```

**Record Timing:**
```csharp
public void RecordTiming(string metricName, TimeSpan duration, int workerId = -1)
{
    if (!_enabled)
        return;

    var record = new TimingRecord
    {
        MetricName = metricName,
        Duration = duration,
        Timestamp = DateTime.UtcNow,
        WorkerId = workerId
    };

    _timingRecords.Enqueue(record);

    // Update counter
    var counter = _counters.GetOrAdd(metricName, _ => new PerformanceCounter());
    counter.Record(duration.TotalMilliseconds);
}
```

**Record Counter:**
```csharp
public void RecordCounter(string metricName, double value)
{
    if (!_enabled)
        return;

    var counter = _counters.GetOrAdd(metricName, _ => new PerformanceCounter());
    counter.Record(value);
}
```

**Get Metrics Summary:**
```csharp
public Dictionary<string, MetricSummary> GetMetricsSummary()
{
    var summary = new Dictionary<string, MetricSummary>();

    foreach (var kvp in _counters)
    {
        summary[kvp.Key] = new MetricSummary
        {
            Count = kvp.Value.Count,
            Average = kvp.Value.Average,
            Min = kvp.Value.Min,
            Max = kvp.Value.Max
        };
    }

    return summary;
}
```

**Metric Summary:**
```csharp
public class MetricSummary
{
    public long Count { get; set; }
    public double Average { get; set; }
    public double Min { get; set; }
    public double Max { get; set; }
    public double Total => Average * Count;
}
```

**Epoch Management:**
```csharp
public void StartEpoch()
{
    if (!_enabled)
        return;

    _epochTimer.Restart();
}

public void EndEpoch()
{
    if (!_enabled)
        return;

    _epochTimer.Stop();

    var counter = _counters.GetOrAdd("EpochTime", _ => new PerformanceCounter());
    counter.Record(_epochTimer.Elapsed.TotalMilliseconds);
}
```

**Reset:**
```csharp
public void Reset()
{
    _timingRecords.Clear();
    _counters.Clear();
    _epochTimer.Reset();
}
```

### Helper: Profiling Scope

#### ProfilingScope
- Using-style helper for automatic timing

```csharp
public class ProfilingScope : IDisposable
{
    private readonly DataLoadingMetrics _metrics;
    private readonly string _metricName;
    private readonly int _workerId;
    private readonly Stopwatch _stopwatch;

    public ProfilingScope(DataLoadingMetrics metrics, string metricName, int workerId = -1)
    {
        _metrics = metrics;
        _metricName = metricName;
        _workerId = workerId;
        _stopwatch = Stopwatch.StartNew();
    }

    public void Dispose()
    {
        _stopwatch.Stop();
        _metrics.RecordTiming(_metricName, _stopwatch.Elapsed, _workerId);
    }
}

// Extension method
public static class DataLoadingMetricsExtensions
{
    public static IDisposable Profile(this DataLoadingMetrics metrics, string metricName, int workerId = -1)
    {
        return new ProfilingScope(metrics, metricName, workerId);
    }
}
```

### Reporting

#### MetricsReporter
- Generate human-readable reports
- Export metrics to various formats

```csharp
public static class MetricsReporter
{
    public static string GenerateTextReport(DataLoadingMetrics metrics)
    {
        var summary = metrics.GetMetricsSummary();
        var sb = new StringBuilder();

        sb.AppendLine("=== Data Loading Metrics ===");

        foreach (var kvp in summary.OrderByDescending(x => x.Value.Total))
        {
            sb.AppendLine($"{kvp.Key}:");
            sb.AppendLine($"  Count: {kvp.Value.Count}");
            sb.AppendLine($"  Avg:   {kvp.Value.Average:F2} ms");
            sb.AppendLine($"  Min:   {kvp.Value.Min:F2} ms");
            sb.AppendLine($"  Max:   {kvp.Value.Max:F2} ms");
            sb.AppendLine($"  Total: {kvp.Value.Total:F2} ms");
            sb.AppendLine();
        }

        return sb.ToString();
    }

    public static void ExportToJson(DataLoadingMetrics metrics, string filePath)
    {
        var summary = metrics.GetMetricsSummary();
        var json = JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(filePath, json);
    }
}
```

## Acceptance Criteria
1. Metrics can record timing for various operations
2. Metrics can record counter values
3. Metrics are thread-safe for concurrent recording
4. GetMetricsSummary returns aggregated statistics
5. ProfilingScope automatically records timing
6. Report generation produces readable output
7. JSON export works correctly
8. Epoch tracking works correctly
9. Reset clears all metrics
10. Unit tests verify thread-safety and accuracy

## Files to Create
- `src/Data/Metrics/DataLoadingMetrics.cs`
- `src/Data/Metrics/TimingRecord.cs`
- `src/Data/Metrics/PerformanceCounter.cs`
- `src/Data/Metrics/MetricSummary.cs`
- `src/Data/Metrics/ProfilingScope.cs`
- `src/Data/Metrics/MetricsReporter.cs`

## Tests
- `tests/Data/Metrics/DataLoadingMetricsTests.cs`
- `tests/Data/Metrics/ProfilingScopeTests.cs`

## Usage Example
```csharp
var metrics = new DataLoadingMetrics(enabled: true);

// Profile a batch loading operation
using (metrics.Profile("BatchLoad", workerId: 0))
{
    var batch = LoadBatch();
}

// Profile transforms
using (metrics.Profile("Transform", workerId: 0))
{
    var transformed = ApplyTransforms(batch);
}

// End of epoch
metrics.EndEpoch();

// Generate report
Console.WriteLine(MetricsReporter.GenerateTextReport(metrics));
```

## Notes
- Critical for identifying bottlenecks
- Minimal overhead (<1% when enabled)
- Disable in production for zero overhead
- Consider adding histogram support for latency distribution
- Future: Real-time monitoring dashboard
- Common bottleneck: transform time > GPU time
- Use metrics to tune numWorkers and prefetchCount
- Export metrics for long-term trend analysis
