# Spec: Profiling Core

## Overview
Implement the core profiling infrastructure that enables tracking execution time of operations, identifying bottlenecks, and building performance timelines.

## Objectives
- Create low-overhead profiling system for timing operations
- Support hierarchical profiling (nested operations)
- Enable timing aggregation and statistics
- Provide scope-based profiling (using statements)

## API Design

```csharp
// Profiling data
public class ProfilingEvent
{
    public string Name { get; }
    public ProfilingEventType Type { get; } // Start, End, Instant
    public long TimestampNanoseconds { get; }
    public long Step { get; }
    public int ThreadId { get; }
    public Dictionary<string, string> Metadata { get; }
}

public enum ProfilingEventType
{
    Start,   // Operation started
    End,     // Operation ended
    Instant  // Instantaneous event (e.g., checkpoint)
}

// Profiling scope (for using statements)
public interface IProfilingScope : IDisposable
{
    string Name { get; }
    long StartTimestampNanoseconds { get; }
    long DurationNanoseconds { get; }
}

// Profiling result
public class ProfilingResult
{
    public string Name { get; }
    public long TotalDurationNanoseconds { get; }
    public long Count { get; }
    public long MinDurationNanoseconds { get; }
    public long MaxDurationNanoseconds { get; }
    public double AverageDurationNanoseconds { get; }
    public double StdDevNanoseconds { get; }

    // Percentiles
    public long P50Nanoseconds { get; }
    public long P90Nanoseconds { get; }
    public long P95Nanoseconds { get; }
    public long P99Nanoseconds { get; }
}

// Profiler interface
public interface IProfiler
{
    // Start a profiling scope
    IProfilingScope StartProfile(string name);
    IProfilingScope StartProfile(string name, Dictionary<string, string> metadata);

    // Record an instant event
    void RecordInstant(string name);
    void RecordInstant(string name, Dictionary<string, string> metadata);

    // Get results
    ProfilingResult GetResult(string name);
    Dictionary<string, ProfilingResult> GetAllResults();

    // Advanced
    void SetParentScope(string childName, string parentName);
    void Enable();
    void Disable();
    bool IsEnabled { get; }
}

// Profiler implementation
public class Profiler : IProfiler
{
    public Profiler(IStorageBackend storage);
    public Profiler(IEventPublisher eventPublisher);

    // Configuration
    public bool EnableAutomatic { get; set; } = true;
    public int MaxStoredOperations { get; set; } = 10000;
}
```

## Implementation Requirements

### 1. ProfilingEvent and ProfilingEventType (20-30 min)
- Implement event types for start, end, and instant events
- Use high-resolution timer (`Stopwatch.GetTimestamp()`)
- Store timestamps in nanoseconds for precision
- Include thread ID for multi-threaded scenarios
- Add metadata dictionary for extensibility

### 2. IProfilingScope and ProfilingScope (30-45 min)
- Implement `ProfilingScope` class:
  - Record start timestamp on construction
  - Record end timestamp on disposal
  - Calculate duration automatically
  - Publish start and end events
- Ensure disposal is idempotent
- Handle disposal exceptions gracefully
- Track nested scopes (parent-child relationships)

### 3. Profiler Core (45-60 min)
- Implement `IProfiler` interface
- Maintain dictionary of profiling data:
  - Track all durations for each operation name
  - Compute statistics (min, max, average, std dev)
  - Compute percentiles (p50, p90, p95, p99)
- Implement enable/disable flag to skip profiling when not needed
- Support automatic profiling (when enabled) vs manual (explicit calls)
- Track parent-child relationships for hierarchical profiling
- Integrate with event system (publish `ProfilingEvent`)
- Integrate with storage backend (persist events)

### 4. Statistics and Percentiles (30-45 min)
- Implement efficient statistics computation:
  - Maintain running sum and sum of squares for avg/std
  - Use reservoir sampling or approximation for large datasets
- Implement percentiles:
  - Use selection algorithm for exact percentiles on small datasets
  - Use approximation (t-digest or similar) for large datasets
- Update statistics incrementally as new data arrives
- Cache computed statistics to avoid recomputation

## File Structure
```
src/
  MLFramework.Visualization/
    Profiling/
      ProfilingEvent.cs
      IProfilingScope.cs
      ProfilingScope.cs
      ProfilingResult.cs
      IProfiler.cs
      Profiler.cs
      Statistics/
        DurationTracker.cs
        PercentileCalculator.cs

tests/
  MLFramework.Visualization.Tests/
    Profiling/
      ProfilerTests.cs
      ProfilingScopeTests.cs
      StatisticsTests.cs
```

## Dependencies
- `MLFramework.Visualization.Events` (Event system)
- `MLFramework.Visualization.Storage` (Storage backend)

## Integration Points
- Used by tensor operations to profile computation
- Integrated with training loops for end-to-end timing
- Data consumed by hardware profiler integration

## Success Criteria
- Profiling overhead < 0.1% (measured by self-profiling)
- Creating 1M profiling scopes completes in <500ms
- Computing statistics for 1M durations in <100ms
- Thread-safe under concurrent profiling (100+ threads)
- Nested scopes correctly track parent-child relationships
- Unit tests verify correctness of all statistics
