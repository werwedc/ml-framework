# Spec: Per-Version Metrics Collection

## Purpose
Collect and export performance metrics per model version for monitoring, observability, and analysis.

## Technical Requirements

### Core Functionality
- Track inference metrics per version (latency, throughput, error rate)
- Track system metrics (memory, CPU, GPU)
- Aggregate metrics over time windows
- Export metrics to monitoring systems (Prometheus, StatsD, etc.)
- Support metric tagging (model_name, version, etc.)
- Real-time metric queries

### Data Structures
```csharp
public class VersionMetrics
{
    public string ModelName { get; }
    public string Version { get; }
    public DateTime WindowStart { get; }
    public DateTime WindowEnd { get; }
    public long RequestCount { get; }
    public double RequestsPerSecond { get; }
    public double AverageLatencyMs { get; }
    public double P50LatencyMs { get; }
    public double P95LatencyMs { get; }
    public double P99LatencyMs { get; }
    public double ErrorRate { get; }
    public long ActiveConnections { get; }
    public double MemoryUsageMB { get; }
}

public interface IMetricsCollector
{
    void RecordInference(string modelName, string version, double latencyMs, bool success, string errorType = null);
    void RecordActiveConnections(string modelName, string version, int count);
    void RecordMemoryUsage(string modelName, string version, long bytes);
    VersionMetrics GetMetrics(string modelName, string version, TimeSpan window);
    Dictionary<string, VersionMetrics> GetAllMetrics(TimeSpan window);
    void ExportMetrics();
    void SetExporter(IMetricsExporter exporter);
    void StartAutoExport(TimeSpan interval);
    void StopAutoExport();
}

public interface IMetricsExporter
{
    void Export(Dictionary<string, VersionMetrics> metrics);
    Task ExportAsync(Dictionary<string, VersionMetrics> metrics);
}

// Example exporters
public class PrometheusExporter : IMetricsExporter
{
    public void Export(Dictionary<string, VersionMetrics> metrics);
}

public class StatsDExporter : IMetricsExporter
{
    public void Export(Dictionary<string, VersionMetrics> metrics);
}

public class ConsoleExporter : IMetricsExporter
{
    public void Export(Dictionary<string, VersionMetrics> metrics);
}
```

### Metric Collection
- Increment counters on each inference
- Track latency histograms (for percentiles)
- Calculate rolling averages
- Windowed aggregation (e.g., last 1 minute, 5 minutes)

## Dependencies
- `spec_model_loader.md` (to track loaded models)
- `spec_reference_counting.md` (to get active connections)

## Testing Requirements
- Record inference, verify metrics updated
- Record multiple inferences, verify aggregations correct
- Get metrics for different time windows (1m, 5m)
- Export metrics to console, verify format
- Auto-export test (periodic export)
- Concurrent recording test (100 threads)
- Export to Prometheus format
- Export to StatsD format
- Performance test: Record 10,000 inferences in < 1s

## Success Criteria
- [ ] Accurately tracks all metric types
- [ ] Calculates correct percentiles and averages
- [ ] Supports multiple export formats
- [ ] Auto-export works reliably
- [ ] Windowed metrics aggregation works
- [ ] Thread-safe under high concurrency
- [ ] Records 10,000+ inferences per second
- [ ] Export completes in < 100ms

## Implementation Notes
- Use thread-safe data structures for concurrent updates
- Use reservoir sampling for percentile calculation
- Implement metric expiration (old windows removed)
- Add metric labels/tags for filtering
- Consider using OpenTelemetry (optional)
- Add histogram buckets for Prometheus (optional)
- Implement metric flushing to reduce memory usage

## Performance Targets
- RecordInference: < 0.01ms
- GetMetrics: < 10ms (even with 100k records)
- Export: < 100ms (for 100+ models)
- Support 100,000+ inferences per second

## Metric Definitions
- `model_inference_requests_total{model_name, version}` - Counter
- `model_inference_latency_ms{model_name, version}` - Histogram
- `model_inference_errors_total{model_name, version, error_type}` - Counter
- `model_memory_usage_bytes{model_name, version}` - Gauge
- `model_active_connections{model_name, version}` - Gauge

## Edge Cases
- Recording metrics for non-existent version
- Very large time windows (hours)
- Export failures (should not block collection)
- Concurrent export operations
- Metric updates during export
