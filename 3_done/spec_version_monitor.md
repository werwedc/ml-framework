# Spec: Version Monitor Implementation

## Overview
Implement IVersionMonitor for tracking per-version metrics, comparing versions, and alerting on anomalies.

## Tasks

### 1. Create IVersionMonitor Interface
**File:** `src/ModelVersioning/IVersionMonitor.cs`

```csharp
public interface IVersionMonitor
{
    void RecordMetric(string modelId, string version, MetricSample sample);
    VersionMetrics GetMetrics(string modelId, string version);
    MetricComparison CompareVersions(string modelId, string v1, string v2);
    void SubscribeToAlerts(Action<VersionAlert> callback);
    void UnsubscribeFromAlerts(Action<VersionAlert> callback);
    void ClearMetrics(string modelId, string version);
}
```

### 2. Implement VersionMonitor Class
**File:** `src/ModelVersioning/VersionMonitor.cs`

```csharp
public class VersionMonitor : IVersionMonitor
{
    private readonly ConcurrentDictionary<string, List<MetricSample>> _metrics;
    private readonly ConcurrentDictionary<string, Action<VersionAlert>> _alertSubscribers;
    private readonly object _lock;
    private readonly Timer _alertCheckTimer;

    public VersionMonitor()
    {
        _metrics = new ConcurrentDictionary<string, List<MetricSample>>();
        _alertSubscribers = new ConcurrentDictionary<string, Action<VersionAlert>>();
        _lock = new object();
        // Start background timer for alert checking
    }

    public void RecordMetric(string modelId, string version, MetricSample sample)
    {
        // Store metric sample
        // Check for anomalies
    }

    public VersionMetrics GetMetrics(string modelId, string version)
    {
        // Aggregate samples into VersionMetrics
    }

    public MetricComparison CompareVersions(string modelId, string v1, string v2)
    {
        // Compare metrics between versions
    }

    public void SubscribeToAlerts(Action<VersionAlert> callback)
    {
        // Subscribe to alerts
    }

    public void UnsubscribeFromAlerts(Action<VersionAlert> callback)
    {
        // Unsubscribe from alerts
    }

    public void ClearMetrics(string modelId, string version)
    {
        // Clear metrics for version
    }
}
```

### 3. Implement Metric Aggregation
- Calculate average latency from samples
- Calculate percentiles (P50, P95, P99)
- Calculate error rate (failed requests / total)
- Calculate throughput (requests / time)
- Track memory usage

### 4. Implement Alert Detection
- Check for high latency (> threshold)
- Check for high error rate (> threshold)
- Check for low throughput (< threshold)
- Check for memory exceeded (> threshold)
- Detect anomalies (statistical deviation)

### 5. Implement Alert Broadcasting
- Notify all subscribers when alert triggered
- Include context (metrics, thresholds)
- Handle multiple subscribers safely

### 6. Implement Version Comparison
- Calculate absolute differences
- Calculate percentage changes
- Determine direction (better/worse)
- Lower latency = better
- Lower error rate = better
- Higher throughput = better

## Validation
- Metric aggregation accuracy
- Alert threshold configuration
- Thread-safe metric recording
- Alert subscription/unsubscription

## Testing
**File:** `tests/ModelVersioning/VersionMonitorTests.cs`

Create unit tests for:
1. RecordMetric and store samples
2. GetMetrics aggregates samples correctly
3. GetMetrics calculates percentiles correctly
4. CompareVersions calculates deltas correctly
5. CompareVersions determines direction correctly
6. SubscribeToAlerts receives notifications
7. UnsubscribeFromAlerts stops notifications
8. Alert detection for high latency
9. Alert detection for high error rate
10. Alert detection for low throughput
11. ClearMetrics removes data
12. Concurrent metric recording
13. Multiple alert subscribers

## Dependencies
- Spec: spec_model_data_models.md
- Spec: spec_monitoring_data_models.md
- System.Collections.Concurrent
- System.Threading.Timer
