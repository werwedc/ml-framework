using System.Collections.Concurrent;

namespace MLFramework.Data.Metrics;

/// <summary>
/// Thread-safe collector for per-version model metrics.
/// </summary>
public class MetricsCollector : IMetricsCollector, IDisposable
{
    private readonly string _name;
    private readonly ConcurrentDictionary<string, VersionMetricData> _versionMetrics;
    private readonly ReaderWriterLockSlim _exportLock;
    private IMetricsExporter _exporter;
    private Timer _autoExportTimer;
    private readonly object _timerLock;
    private bool _disposed;

    // Performance target: RecordInference should be < 0.01ms
    // We use lock-free data structures where possible

    private class VersionMetricData
    {
        public string ModelName { get; }
        public string Version { get; }
        public ConcurrentQueue<InferenceRecord> InferenceRecords { get; }
        private long _activeConnections;
        private long _memoryUsageBytes;
        public DateTime LastUpdate { get; set; }

        public long ActiveConnections
        {
            get => _activeConnections;
            set => _activeConnections = value;
        }

        public long MemoryUsageBytes
        {
            get => _memoryUsageBytes;
            set => _memoryUsageBytes = value;
        }

        public VersionMetricData(string modelName, string version)
        {
            ModelName = modelName;
            Version = version;
            InferenceRecords = new ConcurrentQueue<InferenceRecord>();
            _activeConnections = 0;
            _memoryUsageBytes = 0;
            LastUpdate = DateTime.UtcNow;
        }
    }

    private class InferenceRecord
    {
        public DateTime Timestamp { get; }
        public double LatencyMs { get; }
        public bool Success { get; }
        public string ErrorType { get; }

        public InferenceRecord(DateTime timestamp, double latencyMs, bool success, string errorType)
        {
            Timestamp = timestamp;
            LatencyMs = latencyMs;
            Success = success;
            ErrorType = errorType;
        }
    }

    public MetricsCollector(string name = "default")
    {
        _name = name;
        _versionMetrics = new ConcurrentDictionary<string, VersionMetricData>();
        _exportLock = new ReaderWriterLockSlim();
        _exporter = new ConsoleExporter();
        _timerLock = new object();
    }

    public string Name => _name;

    public void RecordInference(string modelName, string version, double latencyMs, bool success, string errorType = null)
    {
        if (string.IsNullOrEmpty(modelName))
            throw new ArgumentNullException(nameof(modelName));
        if (string.IsNullOrEmpty(version))
            throw new ArgumentNullException(nameof(version));

        var key = GetKey(modelName, version);
        var record = new InferenceRecord(DateTime.UtcNow, latencyMs, success, errorType);

        var data = _versionMetrics.GetOrAdd(key, k => new VersionMetricData(modelName, version));
        data.InferenceRecords.Enqueue(record);
        data.LastUpdate = DateTime.UtcNow;

        // Periodic cleanup of old records to prevent memory leaks
        if (data.InferenceRecords.Count > 100000)
        {
            CleanupOldRecords(data);
        }
    }

    public void RecordActiveConnections(string modelName, string version, int count)
    {
        if (string.IsNullOrEmpty(modelName))
            throw new ArgumentNullException(nameof(modelName));
        if (string.IsNullOrEmpty(version))
            throw new ArgumentNullException(nameof(version));

        var key = GetKey(modelName, version);
        var data = _versionMetrics.GetOrAdd(key, k => new VersionMetricData(modelName, version));
        data.ActiveConnections = count;
        data.LastUpdate = DateTime.UtcNow;
    }

    public void RecordMemoryUsage(string modelName, string version, long bytes)
    {
        if (string.IsNullOrEmpty(modelName))
            throw new ArgumentNullException(nameof(modelName));
        if (string.IsNullOrEmpty(version))
            throw new ArgumentNullException(nameof(version));

        var key = GetKey(modelName, version);
        var data = _versionMetrics.GetOrAdd(key, k => new VersionMetricData(modelName, version));
        data.MemoryUsageBytes = bytes;
        data.LastUpdate = DateTime.UtcNow;
    }

    public VersionMetrics GetMetrics(string modelName, string version, TimeSpan window)
    {
        if (string.IsNullOrEmpty(modelName))
            throw new ArgumentNullException(nameof(modelName));
        if (string.IsNullOrEmpty(version))
            throw new ArgumentNullException(nameof(version));

        var key = GetKey(modelName, version);
        if (!_versionMetrics.TryGetValue(key, out var data))
        {
            return CreateEmptyMetrics(modelName, version, window);
        }

        return CalculateMetrics(data, window);
    }

    public Dictionary<string, VersionMetrics> GetAllMetrics(TimeSpan window)
    {
        var result = new Dictionary<string, VersionMetrics>();

        foreach (var kvp in _versionMetrics)
        {
            var metrics = CalculateMetrics(kvp.Value, window);
            result[kvp.Key] = metrics;
        }

        return result;
    }

    public void ExportMetrics()
    {
        var window = TimeSpan.FromMinutes(5);
        var metrics = GetAllMetrics(window);

        try
        {
            _exportLock.EnterWriteLock();
            _exporter.Export(metrics);
        }
        finally
        {
            _exportLock.ExitWriteLock();
        }
    }

    public void SetExporter(IMetricsExporter exporter)
    {
        if (exporter == null)
            throw new ArgumentNullException(nameof(exporter));

        try
        {
            _exportLock.EnterWriteLock();
            _exporter = exporter;
        }
        finally
        {
            _exportLock.ExitWriteLock();
        }
    }

    public void StartAutoExport(TimeSpan interval)
    {
        if (interval <= TimeSpan.Zero)
            throw new ArgumentException("Interval must be positive", nameof(interval));

        lock (_timerLock)
        {
            StopAutoExport(); // Stop existing timer if any
            _autoExportTimer = new Timer(_ => ExportMetrics(), null, interval, interval);
        }
    }

    public void StopAutoExport()
    {
        lock (_timerLock)
        {
            _autoExportTimer?.Dispose();
            _autoExportTimer = null;
        }
    }

    private VersionMetrics CalculateMetrics(VersionMetricData data, TimeSpan window)
    {
        var windowEnd = DateTime.UtcNow;
        var windowStart = windowEnd - window;

        // Filter records within the window
        var records = new List<InferenceRecord>();
        var tempRecords = new List<InferenceRecord>();

        // Dequeue and filter
        while (data.InferenceRecords.TryDequeue(out var record))
        {
            if (record.Timestamp >= windowStart)
            {
                records.Add(record);
            }
            // Old records are discarded
        }

        // Re-queue records that are still within the window
        foreach (var record in records)
        {
            data.InferenceRecords.Enqueue(record);
        }

        if (records.Count == 0)
        {
            return CreateEmptyMetrics(data.ModelName, data.Version, window);
        }

        // Calculate metrics
        var requestCount = records.Count;
        var errorCount = records.Count(r => !r.Success);
        var errorRate = (double)errorCount / requestCount * 100;
        var latencies = records.Select(r => r.LatencyMs).OrderBy(l => l).ToList();

        // Calculate percentiles
        var p50 = CalculatePercentile(latencies, 0.50);
        var p95 = CalculatePercentile(latencies, 0.95);
        var p99 = CalculatePercentile(latencies, 0.99);

        var averageLatency = latencies.Average();

        // Calculate requests per second
        var windowDuration = (records.Last().Timestamp - records.First().Timestamp).TotalSeconds;
        var requestsPerSecond = windowDuration > 0 ? requestCount / windowDuration : requestCount;

        // Group errors by type
        var errorCountsByType = records
            .Where(r => !r.Success && r.ErrorType != null)
            .GroupBy(r => r.ErrorType)
            .ToDictionary(g => g.Key, g => g.Count());

        return new VersionMetrics(
            data.ModelName,
            data.Version,
            windowStart,
            windowEnd,
            requestCount,
            requestsPerSecond,
            averageLatency,
            p50,
            p95,
            p99,
            errorRate,
            data.ActiveConnections,
            data.MemoryUsageBytes / (1024.0 * 1024.0), // Convert to MB
            errorCount,
            ConvertErrorCounts(errorCountsByType)
        );
    }

    private VersionMetrics CreateEmptyMetrics(string modelName, string version, TimeSpan window)
    {
        var now = DateTime.UtcNow;
        return new VersionMetrics(
            modelName,
            version,
            now - window,
            now,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            new Dictionary<string, long>()
        );
    }

    private Dictionary<string, long> ConvertErrorCounts(Dictionary<string, int> source)
    {
        var result = new Dictionary<string, long>();
        foreach (var kvp in source)
        {
            result[kvp.Key] = kvp.Value;
        }
        return result;
    }

    private double CalculatePercentile(List<double> sortedValues, double percentile)
    {
        if (sortedValues.Count == 0)
            return 0;

        if (percentile <= 0)
            return sortedValues[0];
        if (percentile >= 1)
            return sortedValues[sortedValues.Count - 1];

        var index = (int)(percentile * (sortedValues.Count - 1));
        return sortedValues[index];
    }

    private void CleanupOldRecords(VersionMetricData data)
    {
        var cutoff = DateTime.UtcNow.AddHours(-1);
        var tempRecords = new List<InferenceRecord>();

        // Remove records older than 1 hour
        while (data.InferenceRecords.TryDequeue(out var record))
        {
            if (record.Timestamp >= cutoff)
            {
                tempRecords.Add(record);
            }
        }

        // Re-queue remaining records
        foreach (var record in tempRecords)
        {
            data.InferenceRecords.Enqueue(record);
        }
    }

    private string GetKey(string modelName, string version)
    {
        return $"{modelName}:{version}";
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        StopAutoExport();
        _exportLock?.Dispose();
        _disposed = true;
    }
}
