namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Collects and manages metrics for the continuous batch scheduler.
/// </summary>
public class SchedulerMetricsCollector : ISchedulerMetrics
{
    private readonly MetricsConfiguration _config;
    private readonly CircularBuffer<RequestMetrics> _requestMetrics;
    private readonly CircularBuffer<IterationMetrics> _iterationMetrics;
    private readonly CircularBuffer<BatchMetrics> _batchMetrics;
    private readonly SlidingWindowCounter _requestCounter;
    private readonly SlidingWindowCounter _tokenCounter;
    private readonly Dictionary<string, long> _errorCounts;
    private DateTime _lastErrorTime;
    private readonly object _lock;

    /// <summary>
    /// Creates a new metrics collector.
    /// </summary>
    /// <param name="config">Metrics configuration.</param>
    public SchedulerMetricsCollector(MetricsConfiguration config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _requestMetrics = new CircularBuffer<RequestMetrics>(config.MaxRequestSamples);
        _iterationMetrics = new CircularBuffer<IterationMetrics>(config.MaxIterationSamples);
        _batchMetrics = new CircularBuffer<BatchMetrics>(config.MaxBatchSamples);
        _requestCounter = new SlidingWindowCounter(config.CounterWindowSeconds);
        _tokenCounter = new SlidingWindowCounter(config.CounterWindowSeconds);
        _errorCounts = new Dictionary<string, long>();
        _lastErrorTime = DateTime.MinValue;
        _lock = new object();
    }

    /// <summary>
    /// Records iteration completion.
    /// </summary>
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

    /// <summary>
    /// Records request completion.
    /// </summary>
    public void RecordRequestCompletion(RequestResult result)
    {
        lock (_lock)
        {
            var queueTime = TimeSpan.Zero; // Simplified
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

            // Update counters
            _requestCounter.Add(1);
            _tokenCounter.Add(result.TokensGenerated);
        }
    }

    /// <summary>
    /// Records batch utilization.
    /// </summary>
    public void RecordBatchUtilization(double utilization)
    {
        lock (_lock)
        {
            var metrics = new BatchMetrics(
                -1, // Placeholder batch ID
                0,  // Placeholder request count
                utilization,
                0,  // Placeholder memory
                DateTime.UtcNow
            );

            _batchMetrics.Add(metrics);
        }
    }

    /// <summary>
    /// Records an error.
    /// </summary>
    public void RecordError(string errorType, Exception exception)
    {
        lock (_lock)
        {
            if (!_errorCounts.ContainsKey(errorType))
            {
                _errorCounts[errorType] = 0;
            }
            _errorCounts[errorType]++;

            _lastErrorTime = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Gets request statistics.
    /// </summary>
    public RequestStatistics GetRequestStatistics()
    {
        lock (_lock)
        {
            var completed = _requestMetrics
                .Where(m => m.Reason != CompletionReason.Cancelled)
                .ToList();

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
                FailedRequests: 0, // Simplified
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

    /// <summary>
    /// Gets iteration statistics.
    /// </summary>
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

            var recentMetrics = _iterationMetrics
                .Where(m => (DateTime.UtcNow - m.Timestamp).TotalSeconds <= 60)
                .ToList();
            double iterationsPerSec = recentMetrics.Count / 60.0;

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

    /// <summary>
    /// Gets batch statistics.
    /// </summary>
    public BatchStatistics GetBatchStatistics()
    {
        lock (_lock)
        {
            if (_batchMetrics.Count == 0)
            {
                return new BatchStatistics(
                    0, 0.0, 0.0, 0.0, 0, 0, 0.0
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
                MinBatchSize: minSize,
                MaxBatchSizeRaw: maxSize
            );
        }
    }

    /// <summary>
    /// Gets error statistics.
    /// </summary>
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

    /// <summary>
    /// Gets a complete metrics snapshot.
    /// </summary>
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

    /// <summary>
    /// Resets all metrics.
    /// </summary>
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

    /// <summary>
    /// Calculates a percentile from latency values.
    /// </summary>
    private double CalculatePercentile(List<RequestMetrics> metrics, int percentile)
    {
        if (metrics.Count == 0)
            return 0.0;

        var values = metrics
            .Select(m => m.TotalTime.TotalMilliseconds)
            .OrderBy(v => v)
            .ToList();

        int index = (int)Math.Ceiling(values.Count * percentile / 100.0) - 1;
        index = Math.Max(0, Math.Min(index, values.Count - 1));

        return values[index];
    }
}
