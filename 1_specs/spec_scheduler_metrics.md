# Spec: Scheduler Metrics for Continuous Batching

## Overview
Implement comprehensive metrics collection and reporting for the continuous batch scheduler. Metrics include request statistics, batch utilization, latency distributions, and GPU resource usage.

## Class: SchedulerMetricsCollector : ISchedulerMetrics
```csharp
public class SchedulerMetricsCollector : ISchedulerMetrics
{
    private readonly MetricsConfiguration _config;
    private readonly CircularBuffer<RequestMetrics> _requestMetrics;
    private readonly CircularBuffer<IterationMetrics> _iterationMetrics;
    private readonly CircularBuffer<BatchMetrics> _batchMetrics;
    private readonly SlidingWindowCounter _requestCounter;
    private readonly SlidingWindowCounter _tokenCounter;
    private readonly object _lock;

    public SchedulerMetricsCollector(MetricsConfiguration config)
    {
        _config = config;
        _requestMetrics = new CircularBuffer<RequestMetrics>(config.MaxRequestSamples);
        _iterationMetrics = new CircularBuffer<IterationMetrics>(config.MaxIterationSamples);
        _batchMetrics = new CircularBuffer<BatchMetrics>(config.MaxBatchSamples);
        _requestCounter = new SlidingWindowCounter(config.CounterWindowSeconds);
        _tokenCounter = new SlidingWindowCounter(config.CounterWindowSeconds);
        _lock = new object();
    }

    // Record iteration completion
    public void RecordIteration(IterationResult result);

    // Record request completion
    public void RecordRequestCompletion(RequestResult result);

    // Record batch utilization
    public void RecordBatchUtilization(double utilization);

    // Record error
    public void RecordError(string errorType, Exception exception);

    // Get request statistics
    public RequestStatistics GetRequestStatistics();

    // Get iteration statistics
    public IterationStatistics GetIterationStatistics();

    // Get batch statistics
    public BatchStatistics GetBatchStatistics();

    // Get error statistics
    public ErrorStatistics GetErrorStatistics();

    // Get all metrics as a snapshot
    public MetricsSnapshot GetSnapshot();

    // Reset all metrics
    public void Reset();
}
```

---

## Class: MetricsConfiguration
```csharp
public record class MetricsConfiguration(
    int MaxRequestSamples,              // Max request metrics to keep
    int MaxIterationSamples,            // Max iteration metrics to keep
    int MaxBatchSamples,                // Max batch metrics to keep
    int CounterWindowSeconds,          // Window for rate counters
    int PercentilePrecision,            // Precision for percentile calculation
    bool EnableDetailedLogging,         // Log detailed metrics
    int DetailedLogIntervalSeconds      // Interval for detailed logs
)
{
    public static readonly MetricsConfiguration Default = new(
        MaxRequestSamples: 10000,
        MaxIterationSamples: 10000,
        MaxBatchSamples: 10000,
        CounterWindowSeconds: 60,
        PercentilePrecision: 2,
        EnableDetailedLogging: false,
        DetailedLogIntervalSeconds: 30
    );
}
```

**Purpose**: Configure metrics collection behavior.

---

## Class: RequestMetrics
```csharp
public record class RequestMetrics(
    RequestId RequestId,
    int TokensGenerated,
    CompletionReason Reason,
    TimeSpan QueueTime,
    TimeSpan ProcessingTime,
    TimeSpan TotalTime,
    DateTime CompletedTime
)
```

**Purpose**: Metrics for a single request.

---

## Class: IterationMetrics
```csharp
public record class IterationMetrics(
    int IterationNumber,
    int RequestCount,
    int TokensGenerated,
    int RequestsCompleted,
    TimeSpan ProcessingTime,
    long MemoryBytesUsed,
    DateTime Timestamp
)
```

**Purpose**: Metrics for a single iteration.

---

## Class: BatchMetrics
```csharp
public record class BatchMetrics(
    int BatchId,
    int RequestCount,
    double Utilization,
    long MemoryBytesUsed,
    DateTime Timestamp
)
```

**Purpose**: Metrics for a single batch.

---

## Class: RequestStatistics
```csharp
public record class RequestStatistics(
    long TotalRequests,
    long CompletedRequests,
    long FailedRequests,
    long CancelledRequests,
    double AverageTokensPerRequest,
    double P50Latency,
    double P95Latency,
    double P99Latency,
    double RequestsPerSecond,
    double TokensPerSecond,
    TimeSpan AverageQueueTime,
    TimeSpan AverageProcessingTime
)
```

**Purpose**: Aggregated request statistics.

---

## Class: IterationStatistics
```csharp
public record class IterationStatistics(
    long TotalIterations,
    int AverageRequestsPerIteration,
    int AverageTokensPerIteration,
    double AverageMemoryBytesPerIteration,
    double AverageProcessingTimeMs,
    double IterationsPerSecond,
    double AverageUtilization
)
```

**Purpose**: Aggregated iteration statistics.

---

## Class: BatchStatistics
```csharp
public record class BatchStatistics(
    long TotalBatches,
    double AverageBatchSize,
    double AverageUtilization,
    double AverageMemoryBytesPerBatch,
    int MaxBatchSize,
    int MinBatchSize
)
```

**Purpose**: Aggregated batch statistics.

---

## Class: ErrorStatistics
```csharp
public record class ErrorStatistics(
    long TotalErrors,
    Dictionary<string, long> ErrorsByType,
    double ErrorRate,
    DateTime LastErrorTime
)
```

**Purpose**: Error statistics.

---

## Class: MetricsSnapshot
```csharp
public record class MetricsSnapshot(
    RequestStatistics RequestStats,
    IterationStatistics IterationStats,
    BatchStatistics BatchStats,
    ErrorStatistics ErrorStats,
    DateTime SnapshotTime
)
```

**Purpose**: Complete metrics snapshot.

---

## Implementation Details

### RecordIteration
```csharp
public void RecordIteration(IterationResult result)
{
    lock (_lock)
    {
        var metrics = new IterationMetrics(
            result.IterationNumber,
            result.RequestCount,
            result.TokensGenerated,
            result.RequestsCompleted,
            result.ProcessingTime,
            result.MemoryBytesUsed,
            DateTime.UtcNow
        );

        _iterationMetrics.Add(metrics);

        // Update token counter
        _tokenCounter.Add(result.TokensGenerated);
    }
}
```

**Requirements**:
- Create metrics from result
- Add to circular buffer
- Update token counter
- Thread-safe

---

### RecordRequestCompletion
```csharp
public void RecordRequestCompletion(RequestResult result)
{
    lock (_lock)
    {
        // Calculate queue time (estimated as total time - processing time)
        // In production, this would be tracked more precisely
        var queueTime = TimeSpan.FromTicks(
            Math.Max(0, result.ProcessingTime.Ticks - result.ProcessingTime.Ticks / 2)
        );

        var processingTime = result.ProcessingTime - queueTime;

        var metrics = new RequestMetrics(
            result.RequestId,
            result.TokensGenerated,
            result.Reason,
            queueTime,
            processingTime,
            result.ProcessingTime,
            DateTime.UtcNow
        );

        _requestMetrics.Add(metrics);

        // Update request counter
        _requestCounter.Add(1);

        // Update token counter
        _tokenCounter.Add(result.TokensGenerated);
    }
}
```

**Requirements**:
- Create metrics from result
- Add to circular buffer
- Update counters
- Thread-safe

---

### RecordBatchUtilization
```csharp
public void RecordBatchUtilization(double utilization)
{
    lock (_lock)
    {
        // Need to associate with current batch
        // For now, record with placeholder batch ID
        var metrics = new BatchMetrics(
            -1, // Would need actual batch ID
            0,  // Would need actual request count
            utilization,
            0,  // Would need actual memory
            DateTime.UtcNow
        );

        _batchMetrics.Add(metrics);
    }
}
```

**Requirements**:
- Record utilization
- Add to circular buffer
- Thread-safe

---

### RecordError
```csharp
private readonly Dictionary<string, long> _errorCounts = new();
private DateTime _lastErrorTime = DateTime.MinValue;

public void RecordError(string errorType, Exception exception)
{
    lock (_lock)
    {
        // Update error count
        if (!_errorCounts.ContainsKey(errorType))
        {
            _errorCounts[errorType] = 0;
        }
        _errorCounts[errorType]++;

        _lastErrorTime = DateTime.UtcNow;
    }
}
```

**Requirements**:
- Track error count by type
- Update last error time
- Thread-safe

---

### GetRequestStatistics
```csharp
public RequestStatistics GetRequestStatistics()
{
    lock (_lock)
    {
        var completed = _requestMetrics
            .Where(m => m.Reason != CompletionReason.Cancelled)
            .ToList();

        var failed = completed
            .Where(m => m.Reason == CompletionReason.Cancelled)
            .Count();

        var cancelled = _requestMetrics
            .Where(m => m.Reason == CompletionReason.Cancelled)
            .Count();

        double avgTokens = completed.Count > 0
            ? completed.Average(m => m.TokensGenerated)
            : 0.0;

        double p50Latency = CalculatePercentile(completed, 50);
        double p95Latency = CalculatePercentile(completed, 95);
        double p99Latency = CalculatePercentile(completed, 99);

        double requestsPerSec = _requestCounter.GetRate();
        double tokensPerSec = _tokenCounter.GetRate();

        TimeSpan avgQueueTime = completed.Count > 0
            ? TimeSpan.FromTicks((long)completed.Average(m => m.QueueTime.Ticks))
            : TimeSpan.Zero;

        TimeSpan avgProcessingTime = completed.Count > 0
            ? TimeSpan.FromTicks((long)completed.Average(m => m.ProcessingTime.Ticks))
            : TimeSpan.Zero;

        return new RequestStatistics(
            TotalRequests: _requestMetrics.Count,
            CompletedRequests: completed.Count,
            FailedRequests: failed,
            CancelledRequests: cancelled,
            AverageTokensPerRequest: avgTokens,
            P50Latency: p50Latency,
            P95Latency: p95Latency,
            P99Latency: p99Latency,
            RequestsPerSecond: requestsPerSec,
            TokensPerSecond: tokensPerSec,
            AverageQueueTime: avgQueueTime,
            AverageProcessingTime: avgProcessingTime
        );
    }
}
```

**Requirements**:
- Calculate from collected metrics
- Calculate percentiles
- Calculate averages
- Thread-safe

---

### GetIterationStatistics
```csharp
public IterationStatistics GetIterationStatistics()
{
    lock (_lock)
    {
        if (_iterationMetrics.Count == 0)
        {
            return new IterationStatistics(
                0, 0, 0, 0.0, 0.0, 0.0, 0.0
            );
        }

        double avgRequests = _iterationMetrics.Average(m => m.RequestCount);
        double avgTokens = _iterationMetrics.Average(m => m.TokensGenerated);
        double avgMemory = _iterationMetrics.Average(m => m.MemoryBytesUsed);
        double avgTimeMs = _iterationMetrics.Average(m => m.ProcessingTime.TotalMilliseconds);

        // Calculate iterations per second (using last 60 seconds)
        var recentMetrics = _iterationMetrics
            .Where(m => (DateTime.UtcNow - m.Timestamp).TotalSeconds <= 60)
            .ToList();
        double iterationsPerSec = recentMetrics.Count / 60.0;

        // Calculate average utilization
        double avgUtilization = avgRequests / 32.0; // Assuming max batch size of 32

        return new IterationStatistics(
            TotalIterations: _iterationMetrics.Count,
            AverageRequestsPerIteration: (int)avgRequests,
            AverageTokensPerIteration: (int)avgTokens,
            AverageMemoryBytesPerIteration: avgMemory,
            AverageProcessingTimeMs: avgTimeMs,
            IterationsPerSecond: iterationsPerSec,
            AverageUtilization: avgUtilization
        );
    }
}
```

**Requirements**:
- Calculate from collected metrics
- Calculate rates
- Thread-safe

---

### GetBatchStatistics
```csharp
public BatchStatistics GetBatchStatistics()
{
    lock (_lock)
    {
        if (_batchMetrics.Count == 0)
        {
            return new BatchStatistics(
                0, 0.0, 0.0, 0.0, 0, 0
            );
        }

        double avgSize = _batchMetrics.Average(m => m.RequestCount);
        double avgUtilization = _batchMetrics.Average(m => m.Utilization);
        double avgMemory = _batchMetrics.Average(m => m.MemoryBytesUsed);
        int maxSize = _batchMetrics.Max(m => m.RequestCount);
        int minSize = _batchMetrics.Min(m => m.RequestCount);

        return new BatchStatistics(
            TotalBatches: _batchMetrics.Count,
            AverageBatchSize: avgSize,
            AverageUtilization: avgUtilization,
            AverageMemoryBytesPerBatch: avgMemory,
            MaxBatchSize: maxSize,
            MinBatchSize: minSize
        );
    }
}
```

**Requirements**:
- Calculate from collected metrics
- Calculate min/max
- Thread-safe

---

### GetErrorStatistics
```csharp
public ErrorStatistics GetErrorStatistics()
{
    lock (_lock)
    {
        long totalErrors = _errorCounts.Values.Sum();
        long totalRequests = _requestMetrics.Count;
        double errorRate = totalRequests > 0
            ? (double)totalErrors / totalRequests
            : 0.0;

        return new ErrorStatistics(
            TotalErrors: totalErrors,
            ErrorsByType: new Dictionary<string, long>(_errorCounts),
            ErrorRate: errorRate,
            LastErrorTime: _lastErrorTime
        );
    }
}
```

**Requirements**:
- Calculate error rate
- Return error breakdown
- Thread-safe

---

### GetSnapshot
```csharp
public MetricsSnapshot GetSnapshot()
{
    return new MetricsSnapshot(
        RequestStats: GetRequestStatistics(),
        IterationStats: GetIterationStatistics(),
        BatchStats: GetBatchStatistics(),
        ErrorStats: GetErrorStatistics(),
        SnapshotTime: DateTime.UtcNow
    );
}
```

**Requirements**:
- Collect all statistics
- Return complete snapshot

---

### Reset
```csharp
public void Reset()
{
    lock (_lock)
    {
        _requestMetrics.Clear();
        _iterationMetrics.Clear();
        _batchMetrics.Clear();
        _requestCounter.Reset();
        _tokenCounter.Reset();
        _errorCounts.Clear();
        _lastErrorTime = DateTime.MinValue;
    }
}
```

**Requirements**:
- Clear all collected metrics
- Reset counters
- Thread-safe

---

## Helper Classes

### CircularBuffer<T>
```csharp
public class CircularBuffer<T> : IEnumerable<T>
{
    private readonly T[] _buffer;
    private int _head;
    private int _tail;
    private int _count;

    public CircularBuffer(int capacity)
    {
        _buffer = new T[capacity];
        _head = 0;
        _tail = 0;
        _count = 0;
    }

    public int Count => _count;
    public int Capacity => _buffer.Length;

    public void Add(T item)
    {
        _buffer[_tail] = item;
        _tail = (_tail + 1) % _buffer.Length;

        if (_count < _buffer.Length)
        {
            _count++;
        }
        else
        {
            _head = (_head + 1) % _buffer.Length;
        }
    }

    public IEnumerator<T> GetEnumerator()
    {
        for (int i = 0; i < _count; i++)
        {
            yield return _buffer[(_head + i) % _buffer.Length];
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public void Clear()
    {
        _head = 0;
        _tail = 0;
        _count = 0;
    }
}
```

---

### SlidingWindowCounter
```csharp
public class SlidingWindowCounter
{
    private readonly Queue<(DateTime Timestamp, int Count)> _events;
    private readonly TimeSpan _window;

    public SlidingWindowCounter(int windowSeconds)
    {
        _events = new Queue<(DateTime, int)>();
        _window = TimeSpan.FromSeconds(windowSeconds);
    }

    public void Add(int count)
    {
        _events.Enqueue((DateTime.UtcNow, count));
        CleanOldEvents();
    }

    public double GetRate()
    {
        CleanOldEvents();
        int total = _events.Sum(e => e.Count);
        return total / _window.TotalSeconds;
    }

    public void Reset()
    {
        _events.Clear();
    }

    private void CleanOldEvents()
    {
        var cutoff = DateTime.UtcNow - _window;
        while (_events.Count > 0 && _events.Peek().Timestamp < cutoff)
        {
            _events.Dequeue();
        }
    }
}
```

---

## Files to Create

### Implementation
- `src/MLFramework/Inference/ContinuousBatching/Metrics/SchedulerMetricsCollector.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/MetricsConfiguration.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/RequestMetrics.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/IterationMetrics.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/BatchMetrics.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/RequestStatistics.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/IterationStatistics.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/BatchStatistics.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/ErrorStatistics.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/MetricsSnapshot.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/CircularBuffer.cs`
- `src/MLFramework/Inference/ContinuousBatching/Metrics/SlidingWindowCounter.cs`

### Tests
- `tests/MLFramework.Tests/Inference/ContinuousBatching/Metrics/SchedulerMetricsCollectorTests.cs`

---

## Dependencies
- `spec_continuous_batching_models.md` (RequestResult, CompletionReason)
- `spec_continuous_scheduler_core.md` (IterationResult)

---

## Testing Requirements

### Unit Tests
1. **Basic Recording**:
   - Record iteration metrics
   - Record request completion
   - Record batch utilization
   - Record errors

2. **Statistics Calculation**:
   - GetRequestStatistics returns correct values
   - GetIterationStatistics returns correct values
   - GetBatchStatistics returns correct values
   - GetErrorStatistics returns correct values

3. **Percentile Calculation**:
   - Calculate P50 correctly
   - Calculate P95 correctly
   - Calculate P99 correctly
   - Handle edge cases (empty, single value)

4. **Circular Buffer**:
   - Add items correctly
   - Overwrite old items when full
   - Iterate items correctly
   - Clear buffer

5. **Sliding Window Counter**:
   - Add counts correctly
   - Calculate rate correctly
   - Clean old events correctly
   - Reset counter

6. **Snapshot**:
   - GetSnapshot returns complete data
   - Snapshot includes all statistics
   - Snapshot timestamp is current

7. **Reset**:
   - Reset clears all metrics
   - Reset clears counters
   - Reset clears error counts

8. **Thread Safety**:
   - Concurrent metric recording
   - Concurrent statistics retrieval
   - Race conditions handling

---

## Success Criteria
- [ ] All public methods implemented
- [ ] Metrics collection accurate
- [ ] Statistics calculation correct
- [ ] Percentile calculation correct
- [ ] Circular buffer works correctly
- [ ] Sliding window counter works correctly
- [ ] Thread-safe operations
- [ ] Unit tests cover all scenarios
