# Spec: A/B Testing Experiment Tracker

## Purpose
Track experiment IDs and collect performance metrics per model version to support A/B testing analysis.

## Technical Requirements

### Core Functionality
- Associate inference requests with experiment IDs
- Track metrics per version within experiment
- Record latency, success/failure, custom metrics
- Query aggregated metrics by version
- Support experiment lifecycle (start, end, results)

### Data Structures
```csharp
public class ExperimentMetrics
{
    public string ExperimentId { get; set; }
    public string ModelName { get; set; }
    public string Version { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public int RequestCount { get; set; }
    public int SuccessCount { get; set; }
    public int ErrorCount { get; set; }
    public double AverageLatencyMs { get; set; }
    public double P50LatencyMs { get; set; }
    public double P95LatencyMs { get; set; }
    public double P99LatencyMs { get; set; }
    public Dictionary<string, double> CustomMetrics { get; set; }
}

public interface IExperimentTracker
{
    void StartExperiment(string experimentId, string modelName, Dictionary<string, float> versionTraffic);
    void EndExperiment(string experimentId);
    void RecordInference(string experimentId, string version, double latencyMs, bool success, Dictionary<string, double> customMetrics = null);
    ExperimentMetrics GetMetrics(string experimentId, string version);
    Dictionary<string, ExperimentMetrics> GetAllMetrics(string experimentId);
    Dictionary<string, double> CompareVersions(string experimentId);
}

public interface IInferenceTracker : IDisposable
{
    void RecordSuccess(double latencyMs);
    void RecordError(double latencyMs, string errorType);
    void AddCustomMetric(string name, double value);
}
```

### Metrics Collection
- Track each inference with version and timestamp
- Calculate percentiles (P50, P95, P99) from latency samples
- Aggregate metrics in memory (or flush periodically)
- Support custom metrics per request

## Dependencies
- `spec_traffic_splitting.md` (used to configure experiment traffic splits)

## Testing Requirements
- Start experiment, verify it's tracked
- Record inference, verify metrics updated
- Record multiple inferences, verify aggregations correct
- Calculate percentiles from sample latencies
- Compare versions, verify statistical comparison returned
- End experiment, verify it's marked complete
- Get metrics for non-existent experiment (should throw)
- Record inference for ended experiment (should throw)
- Performance test: Record 10,000 inferences in < 1 second
- Concurrent tracking test (100 threads recording)

## Success Criteria
- [ ] Accurately tracks requests per version
- [ ] Calculates correct latency percentiles
- [ ] Aggregates metrics correctly
- [ ] Supports 10+ concurrent experiments
- [ ] Records 10,000+ inferences per second
- [ ] Memory usage scales linearly with request count
- [ ] Provides meaningful version comparison

## Implementation Notes
- Use thread-safe collections for concurrent metric updates
- Consider using reservoir sampling for percentile calculation
- Store raw latency samples or use streaming percentile algorithm
- Add metric persistence (optional, for long experiments)
- Implement metric flush to database (optional future)
- Add experiment metadata (description, owner, etc.)
- Consider adding statistical significance testing (optional)

## Performance Targets
- RecordInference: < 0.1ms
- GetMetrics: < 10ms (even with 100,000 records)
- Start/End Experiment: < 1ms
- Support 100,000+ inferences per second per experiment

## Statistical Analysis
- Support t-test for comparing mean latencies
- Support chi-square test for comparing error rates
- Calculate confidence intervals
- Provide statistical significance flags (optional future)

## Edge Cases
- Recording inference for non-existent experiment
- Recording inference for non-tracked version
- Very large experiments (1M+ inferences)
- Concurrent experiment lifecycle operations
