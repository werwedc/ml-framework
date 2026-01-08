using System;

namespace MLFramework.Serving;

/// <summary>
/// Snapshot of batching metrics at a point in time
/// </summary>
public class BatchingMetrics
{
    /// <summary>
    /// Total number of requests processed
    /// </summary>
    public long TotalRequests { get; set; }

    /// <summary>
    /// Total number of batches processed
    /// </summary>
    public long TotalBatches { get; set; }

    /// <summary>
    /// Average batch size (requests per batch)
    /// </summary>
    public double AverageBatchSize { get; set; }

    /// <summary>
    /// Current queue depth
    /// </summary>
    public int CurrentQueueDepth { get; set; }

    /// <summary>
    /// Maximum queue depth observed
    /// </summary>
    public int MaxQueueDepth { get; set; }

    /// <summary>
    /// Number of times queue was full (rejections)
    /// </summary>
    public long QueueFullRejections { get; set; }

    /// <summary>
    /// Average wait time in queue (milliseconds)
    /// </summary>
    public double AverageQueueWaitMs { get; set; }

    /// <summary>
    /// Average batch processing time (milliseconds)
    /// </summary>
    public double AverageBatchProcessingMs { get; set; }

    /// <summary>
    /// Timestamp when metrics were captured
    /// </summary>
    public DateTime CapturedAt { get; set; }
}

/// <summary>
/// Histogram of batch size distribution
/// </summary>
public class BatchSizeDistribution
{
    /// <summary>
    /// Count of batches with 1-5 requests
    /// </summary>
    public int VerySmall { get; set; }

    /// <summary>
    /// Count of batches with 6-15 requests
    /// </summary>
    public int Small { get; set; }

    /// <summary>
    /// Count of batches with 16-30 requests
    /// </summary>
    public int Medium { get; set; }

    /// <summary>
    /// Count of batches with 31-63 requests
    /// </summary>
    public int Large { get; set; }

    /// <summary>
    /// Count of batches with 64+ requests
    /// </summary>
    public int VeryLarge { get; set; }
}

/// <summary>
/// Collects and tracks batching metrics
/// </summary>
public class BatchingMetricsCollector
{
    private long _totalRequests;
    private long _totalBatches;
    private long _totalRequestsInBatches;
    private long _maxQueueDepth;
    private long _queueFullRejections;
    private long _totalQueueWaitMs;
    private long _totalBatchProcessingMs;

    private readonly object _lock = new object();
    private readonly BatchingSizeHistogram _histogram;

    public BatchingMetricsCollector()
    {
        _histogram = new BatchingSizeHistogram();
    }

    /// <summary>
    /// Record a batch being processed
    /// </summary>
    /// <param name="batchSize">Number of requests in the batch</param>
    /// <param name="queueWaitMs">Average time requests waited in queue</param>
    /// <param name="processingMs">Time taken to process the batch</param>
    public void RecordBatch(int batchSize, double queueWaitMs, double processingMs)
    {
        lock (_lock)
        {
            _totalBatches++;
            _totalRequestsInBatches += batchSize;
            _totalQueueWaitMs += (long)queueWaitMs;
            _totalBatchProcessingMs += (long)processingMs;
            _histogram.Record(batchSize);
        }
    }

    /// <summary>
    /// Record a request being enqueued
    /// </summary>
    public void RecordRequestEnqueued(int currentQueueDepth)
    {
        lock (_lock)
        {
            _totalRequests++;

            if (currentQueueDepth > _maxQueueDepth)
            {
                _maxQueueDepth = currentQueueDepth;
            }
        }
    }

    /// <summary>
    /// Record a queue rejection (queue full)
    /// </summary>
    public void RecordQueueRejection()
    {
        System.Threading.Interlocked.Increment(ref _queueFullRejections);
    }

    /// <summary>
    /// Get current metrics snapshot
    /// </summary>
    public BatchingMetrics GetSnapshot(int currentQueueDepth)
    {
        lock (_lock)
        {
            return new BatchingMetrics
            {
                TotalRequests = _totalRequests,
                TotalBatches = _totalBatches,
                AverageBatchSize = _totalBatches > 0
                    ? (double)_totalRequestsInBatches / _totalBatches
                    : 0,
                CurrentQueueDepth = currentQueueDepth,
                MaxQueueDepth = (int)_maxQueueDepth,
                QueueFullRejections = _queueFullRejections,
                AverageQueueWaitMs = _totalBatches > 0
                    ? (double)_totalQueueWaitMs / _totalBatches
                    : 0,
                AverageBatchProcessingMs = _totalBatches > 0
                    ? (double)_totalBatchProcessingMs / _totalBatches
                    : 0,
                CapturedAt = DateTime.UtcNow
            };
        }
    }

    /// <summary>
    /// Get batch size distribution histogram
    /// </summary>
    public BatchSizeDistribution GetBatchSizeDistribution()
    {
        return _histogram.GetDistribution();
    }

    /// <summary>
    /// Reset all metrics (for testing)
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _totalRequests = 0;
            _totalBatches = 0;
            _totalRequestsInBatches = 0;
            _maxQueueDepth = 0;
            _queueFullRejections = 0;
            _totalQueueWaitMs = 0;
            _totalBatchProcessingMs = 0;
            _histogram.Reset();
        }
    }
}

/// <summary>
/// Internal histogram implementation for batch size distribution
/// </summary>
internal class BatchingSizeHistogram
{
    private int _verySmall;
    private int _small;
    private int _medium;
    private int _large;
    private int _veryLarge;

    public void Record(int batchSize)
    {
        if (batchSize <= 5)
            _verySmall++;
        else if (batchSize <= 15)
            _small++;
        else if (batchSize <= 30)
            _medium++;
        else if (batchSize <= 63)
            _large++;
        else
            _veryLarge++;
    }

    public BatchSizeDistribution GetDistribution()
    {
        return new BatchSizeDistribution
        {
            VerySmall = _verySmall,
            Small = _small,
            Medium = _medium,
            Large = _large,
            VeryLarge = _veryLarge
        };
    }

    public void Reset()
    {
        _verySmall = 0;
        _small = 0;
        _medium = 0;
        _large = 0;
        _veryLarge = 0;
    }
}
