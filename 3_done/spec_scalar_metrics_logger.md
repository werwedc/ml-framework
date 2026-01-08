# Spec: Scalar Metrics Logger

## Overview
Implement a specialized logger for scalar metrics (loss, accuracy, learning rate) that provides efficient storage and retrieval of time-series data.

## Objectives
- Efficient logging of scalar values with minimal overhead
- Support multiple scalar metrics with hierarchical naming
- Provide smoothing and aggregation capabilities
- Enable comparison of metrics across runs

## API Design

```csharp
// Scalar metric entry
public class ScalarEntry
{
    public long Step { get; }
    public float Value { get; }
    public DateTime Timestamp { get; }
}

// Scalar series (all values for a single metric)
public class ScalarSeries
{
    public string Name { get; }
    public List<ScalarEntry> Entries { get; }
    public float? Min { get; }
    public float? Max { get; }
    public float Average { get; }

    public void Add(ScalarEntry entry);
    public IEnumerable<ScalarEntry> GetRange(long startStep, long endStep);
    public ScalarSeries Smoothed(int windowSize);
    public ScalarSeries Resampled(int targetCount);
}

// Scalar metrics logger
public interface IScalarLogger
{
    void LogScalar(string name, float value, long step = -1);
    void LogScalar(string name, double value, long step = -1);
    Task LogScalarAsync(string name, float value, long step = -1);

    // Retrieval
    ScalarSeries GetSeries(string name);
    IEnumerable<ScalarSeries> GetAllSeries();
    Task<ScalarSeries> GetSeriesAsync(string name);

    // Smoothing and aggregation
    ScalarSeries GetSmoothedSeries(string name, int windowSize);
    Dictionary<string, float> GetLatestValues();

    // Comparison
    void TagRun(string runName, Dictionary<string, string> tags);
}

public class ScalarLogger : IScalarLogger
{
    public ScalarLogger(IStorageBackend storage);
    public ScalarLogger(IEventPublisher eventPublisher);

    // Configuration
    public bool AutoSmooth { get; set; }
    public int DefaultSmoothingWindow { get; set; }
    public int MaxEntriesPerSeries { get; set; }
}
```

## Implementation Requirements

### 1. ScalarEntry and ScalarSeries (30-45 min)
- Implement `ScalarEntry` with step, value, and timestamp
- Implement `ScalarSeries` with:
  - Thread-safe entry addition
  - Automatic min/max/average calculation
  - Range query support (get entries between steps)
  - Smoothing with moving average algorithm
  - Resampling to fixed number of points for visualization
- Ensure efficient storage (use `List<ScalarEntry>` or similar)

### 2. ScalarLogger Core (45-60 min)
- Implement `IScalarLogger` interface
- Maintain dictionary of series by name
- Support hierarchical naming (e.g., "train/loss", "val/accuracy")
- Auto-generate step numbers if not provided (counter)
- Convert double to float for storage efficiency
- Integrate with event system (publish `ScalarMetricEvent`)
- Integrate with storage backend (persist events)
- Implement async logging with minimal overhead

### 3. Smoothing and Aggregation (30-45 min)
- Implement moving average smoothing algorithm:
  - `Smoothed(int windowSize)` returns new series
  - Handle edge cases (series shorter than window)
- Implement resampling algorithm:
  - `Resampled(int targetCount)` returns evenly spaced points
  - Use interpolation between actual data points
- Compute statistics:
  - Min, max, average for each series
  - Latest values for all metrics

### 4. Run Comparison Support (20-30 min)
- Add run tagging system
- Store metadata (hyperparameters, model architecture info)
- Support multiple runs in same logger instance
- Enable querying by run tag

## File Structure
```
src/
  MLFramework.Visualization/
    Scalars/
      ScalarEntry.cs
      ScalarSeries.cs
      IScalarLogger.cs
      ScalarLogger.cs
      Smoothing/
        MovingAverageSmoother.cs
        Resampler.cs

tests/
  MLFramework.Visualization.Tests/
    Scalars/
      ScalarLoggerTests.cs
      ScalarSeriesTests.cs
      SmoothingTests.cs
```

## Dependencies
- `MLFramework.Visualization.Events` (Event system)
- `MLFramework.Visualization.Storage` (Storage backend)

## Integration Points
- Used by training loops to log metrics
- Integrated with Visualizer main API
- Data consumed by metrics dashboard visualization

## Success Criteria
- Logging 1,000,000 scalar values completes in <100ms
- Memory usage scales linearly with number of entries
- Smoothing produces correct results (unit tests verify)
- Resampling preserves important data points
- Thread-safe under concurrent logging (10+ threads)
- Unit tests cover all scenarios including edge cases
