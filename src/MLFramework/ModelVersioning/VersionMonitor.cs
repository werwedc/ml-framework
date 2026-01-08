using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Monitors model version metrics, compares versions, and alerts on anomalies.
    /// </summary>
    public class VersionMonitor : IVersionMonitor, IDisposable
    {
        private readonly Dictionary<string, List<MetricSample>> _metrics;
        private readonly Dictionary<string, Action<VersionAlert>> _alertSubscribers;
        private readonly object _lock;
        private readonly Timer _alertCheckTimer;
        private bool _disposed;

        // Alert thresholds (configurable)
        public double HighLatencyThresholdMs { get; set; } = 1000.0;
        public double HighErrorRateThreshold { get; set; } = 0.05; // 5%
        public double LowThroughputThreshold { get; set; } = 10.0; // requests per second
        public double MemoryThresholdMB { get; set; } = 1024.0; // 1 GB
        public double AnomalyDetectionThreshold { get; set; } = 2.0; // Standard deviations

        /// <summary>
        /// Initializes a new instance of VersionMonitor.
        /// </summary>
        public VersionMonitor()
        {
            _metrics = new Dictionary<string, List<MetricSample>>();
            _alertSubscribers = new Dictionary<string, Action<VersionAlert>>();
            _lock = new object();
            // Start background timer for alert checking every 30 seconds
            _alertCheckTimer = new Timer(CheckForAlerts, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
        }

        /// <summary>
        /// Records a metric sample for a specific model version.
        /// </summary>
        public void RecordMetric(string modelId, string version, MetricSample sample)
        {
            if (string.IsNullOrEmpty(modelId))
            {
                throw new ArgumentException("Model ID cannot be null or empty", nameof(modelId));
            }

            if (string.IsNullOrEmpty(version))
            {
                throw new ArgumentException("Version cannot be null or empty", nameof(version));
            }

            if (sample == null)
            {
                throw new ArgumentNullException(nameof(sample));
            }

            string key = GetKey(modelId, version);

            lock (_lock)
            {
                if (!_metrics.ContainsKey(key))
                {
                    _metrics[key] = new List<MetricSample>();
                }

                _metrics[key].Add(sample);
            }

            // Check for immediate alerts based on this single sample
            CheckForImmediateAlerts(modelId, version, sample);
        }

        /// <summary>
        /// Gets aggregated metrics for a specific model version.
        /// </summary>
        public VersionMetrics GetMetrics(string modelId, string version)
        {
            if (string.IsNullOrEmpty(modelId))
            {
                throw new ArgumentException("Model ID cannot be null or empty", nameof(modelId));
            }

            if (string.IsNullOrEmpty(version))
            {
                throw new ArgumentException("Version cannot be null or empty", nameof(version));
            }

            string key = GetKey(modelId, version);

            List<MetricSample> samples;
            lock (_lock)
            {
                if (!_metrics.ContainsKey(key) || _metrics[key].Count == 0)
                {
                    return new VersionMetrics
                    {
                        ModelId = modelId,
                        Version = version,
                        StartTime = DateTime.UtcNow,
                        EndTime = DateTime.UtcNow,
                        TotalRequests = 0
                    };
                }

                samples = new List<MetricSample>(_metrics[key]);
            }

            return AggregateSamples(modelId, version, samples);
        }

        /// <summary>
        /// Compares metrics between two versions of a model.
        /// </summary>
        public MetricComparison CompareVersions(string modelId, string v1, string v2)
        {
            if (string.IsNullOrEmpty(modelId))
            {
                throw new ArgumentException("Model ID cannot be null or empty", nameof(modelId));
            }

            if (string.IsNullOrEmpty(v1))
            {
                throw new ArgumentException("Version 1 cannot be null or empty", nameof(v1));
            }

            if (string.IsNullOrEmpty(v2))
            {
                throw new ArgumentException("Version 2 cannot be null or empty", nameof(v2));
            }

            VersionMetrics metrics1 = GetMetrics(modelId, v1);
            VersionMetrics metrics2 = GetMetrics(modelId, v2);

            return MetricComparison.Create(metrics1, metrics2);
        }

        /// <summary>
        /// Subscribes to version alerts.
        /// </summary>
        public void SubscribeToAlerts(Action<VersionAlert> callback)
        {
            if (callback == null)
            {
                throw new ArgumentNullException(nameof(callback));
            }

            string subscriberId = callback.GetHashCode().ToString();

            lock (_lock)
            {
                _alertSubscribers[subscriberId] = callback;
            }
        }

        /// <summary>
        /// Unsubscribes from version alerts.
        /// </summary>
        public void UnsubscribeFromAlerts(Action<VersionAlert> callback)
        {
            if (callback == null)
            {
                throw new ArgumentNullException(nameof(callback));
            }

            string subscriberId = callback.GetHashCode().ToString();

            lock (_lock)
            {
                _alertSubscribers.Remove(subscriberId);
            }
        }

        /// <summary>
        /// Clears all metrics for a specific model version.
        /// </summary>
        public void ClearMetrics(string modelId, string version)
        {
            if (string.IsNullOrEmpty(modelId))
            {
                throw new ArgumentException("Model ID cannot be null or empty", nameof(modelId));
            }

            if (string.IsNullOrEmpty(version))
            {
                throw new ArgumentException("Version cannot be null or empty", nameof(version));
            }

            string key = GetKey(modelId, version);

            lock (_lock)
            {
                if (_metrics.ContainsKey(key))
                {
                    _metrics[key].Clear();
                }
            }
        }

        /// <summary>
        /// Aggregates a list of metric samples into VersionMetrics.
        /// </summary>
        private VersionMetrics AggregateSamples(string modelId, string version, List<MetricSample> samples)
        {
            if (samples == null || samples.Count == 0)
            {
                return new VersionMetrics
                {
                    ModelId = modelId,
                    Version = version,
                    StartTime = DateTime.UtcNow,
                    EndTime = DateTime.UtcNow,
                    TotalRequests = 0
                };
            }

            // Sort by timestamp
            var sortedSamples = samples.OrderBy(s => s.Timestamp).ToList();

            // Calculate basic metrics
            int totalRequests = samples.Count;
            int failedRequests = samples.Count(s => !s.Success);
            double errorRate = (double)failedRequests / totalRequests;

            // Calculate latency statistics
            var latencies = samples.Select(s => s.LatencyMs).ToList();
            double averageLatency = latencies.Average();
            latencies.Sort();

            // Calculate percentiles
            double p50 = CalculatePercentile(latencies, 0.50);
            double p95 = CalculatePercentile(latencies, 0.95);
            double p99 = CalculatePercentile(latencies, 0.99);

            // Calculate throughput
            TimeSpan duration = sortedSamples.Last().Timestamp - sortedSamples.First().Timestamp;
            double throughput = duration.TotalSeconds > 0 ? totalRequests / duration.TotalSeconds : 0;

            // Calculate memory usage
            double memoryUsage = samples.Average(s => s.MemoryUsageMB);

            return new VersionMetrics
            {
                ModelId = modelId,
                Version = version,
                StartTime = sortedSamples.First().Timestamp,
                EndTime = sortedSamples.Last().Timestamp,
                TotalRequests = totalRequests,
                AverageLatencyMs = averageLatency,
                P50LatencyMs = p50,
                P95LatencyMs = p95,
                P99LatencyMs = p99,
                ErrorRate = errorRate,
                Throughput = throughput,
                MemoryUsageMB = memoryUsage
            };
        }

        /// <summary>
        /// Calculates a percentile from a sorted list of values.
        /// </summary>
        private double CalculatePercentile(List<double> sortedValues, double percentile)
        {
            if (sortedValues == null || sortedValues.Count == 0)
            {
                return 0;
            }

            double index = (sortedValues.Count - 1) * percentile;
            int lowerIndex = (int)Math.Floor(index);
            int upperIndex = (int)Math.Ceiling(index);

            if (lowerIndex == upperIndex)
            {
                return sortedValues[lowerIndex];
            }

            double weight = index - lowerIndex;
            return sortedValues[lowerIndex] * (1 - weight) + sortedValues[upperIndex] * weight;
        }

        /// <summary>
        /// Checks for alerts based on aggregated metrics.
        /// </summary>
        private void CheckForAlerts(object? state)
        {
            if (_disposed)
            {
                return;
            }

            Dictionary<string, List<MetricSample>> metricsCopy;
            lock (_lock)
            {
                metricsCopy = new Dictionary<string, List<MetricSample>>();
                foreach (var kvp in _metrics)
                {
                    metricsCopy[kvp.Key] = new List<MetricSample>(kvp.Value);
                }
            }

            // Check each model/version for alerts
            foreach (var kvp in metricsCopy)
            {
                var (modelId, version) = ParseKey(kvp.Key);

                if (kvp.Value.Count == 0)
                {
                    continue;
                }

                VersionMetrics metrics = AggregateSamples(modelId, version, kvp.Value);

                // Check for high latency
                if (metrics.AverageLatencyMs > HighLatencyThresholdMs)
                {
                    TriggerAlert(VersionAlert.Create(
                        modelId,
                        version,
                        AlertType.HighLatency,
                        $"Average latency {metrics.AverageLatencyMs:F2}ms exceeds threshold {HighLatencyThresholdMs:F2}ms",
                        AlertSeverity.Warning,
                        new Dictionary<string, object>
                        {
                            { "averageLatency", metrics.AverageLatencyMs },
                            { "threshold", HighLatencyThresholdMs },
                            { "p95Latency", metrics.P95LatencyMs }
                        }
                    ));
                }

                // Check for high error rate
                if (metrics.ErrorRate > HighErrorRateThreshold)
                {
                    TriggerAlert(VersionAlert.Create(
                        modelId,
                        version,
                        AlertType.HighErrorRate,
                        $"Error rate {metrics.ErrorRate:P2} exceeds threshold {HighErrorRateThreshold:P2}",
                        AlertSeverity.Critical,
                        new Dictionary<string, object>
                        {
                            { "errorRate", metrics.ErrorRate },
                            { "threshold", HighErrorRateThreshold },
                            { "failedRequests", (int)(metrics.TotalRequests * metrics.ErrorRate) },
                            { "totalRequests", metrics.TotalRequests }
                        }
                    ));
                }

                // Check for low throughput
                if (metrics.Throughput < LowThroughputThreshold && metrics.TotalRequests > 10)
                {
                    TriggerAlert(VersionAlert.Create(
                        modelId,
                        version,
                        AlertType.LowThroughput,
                        $"Throughput {metrics.Throughput:F2} req/s below threshold {LowThroughputThreshold:F2} req/s",
                        AlertSeverity.Warning,
                        new Dictionary<string, object>
                        {
                            { "throughput", metrics.Throughput },
                            { "threshold", LowThroughputThreshold },
                            { "totalRequests", metrics.TotalRequests }
                        }
                    ));
                }

                // Check for memory exceeded
                if (metrics.MemoryUsageMB > MemoryThresholdMB)
                {
                    TriggerAlert(VersionAlert.Create(
                        modelId,
                        version,
                        AlertType.MemoryExceeded,
                        $"Memory usage {metrics.MemoryUsageMB:F2}MB exceeds threshold {MemoryThresholdMB:F2}MB",
                        AlertSeverity.Warning,
                        new Dictionary<string, object>
                        {
                            { "memoryUsage", metrics.MemoryUsageMB },
                            { "threshold", MemoryThresholdMB }
                        }
                    ));
                }

                // Check for anomalies (statistical deviation)
                CheckForAnomalies(modelId, version, kvp.Value);
            }
        }

        /// <summary>
        /// Checks for anomalies based on statistical deviation from baseline.
        /// </summary>
        private void CheckForAnomalies(string modelId, string version, List<MetricSample> samples)
        {
            if (samples.Count < 30)
            {
                return; // Not enough samples for statistical significance
            }

            var recentLatencies = samples.TakeLast(30).Select(s => s.LatencyMs).ToList();
            double mean = recentLatencies.Average();
            double stdDev = CalculateStandardDeviation(recentLatencies, mean);

            // Check if any samples are outside the threshold
            var anomalies = recentLatencies.Where(latency => Math.Abs(latency - mean) > AnomalyDetectionThreshold * stdDev).ToList();

            if (anomalies.Count > 0)
            {
                double anomalyPercentage = (double)anomalies.Count / recentLatencies.Count;
                if (anomalyPercentage > 0.1) // More than 10% anomalies
                {
                    TriggerAlert(VersionAlert.Create(
                        modelId,
                        version,
                        AlertType.AnomalyDetected,
                        $"{anomalies.Count} anomalies detected in recent 30 samples ({anomalyPercentage:P2})",
                        AlertSeverity.Warning,
                        new Dictionary<string, object>
                        {
                            { "anomalyCount", anomalies.Count },
                            { "totalSamples", recentLatencies.Count },
                            { "mean", mean },
                            { "stdDev", stdDev },
                            { "threshold", AnomalyDetectionThreshold }
                        }
                    ));
                }
            }
        }

        /// <summary>
        /// Calculates standard deviation of a list of values.
        /// </summary>
        private double CalculateStandardDeviation(List<double> values, double mean)
        {
            double sumOfSquares = values.Sum(val => Math.Pow(val - mean, 2));
            return Math.Sqrt(sumOfSquares / values.Count);
        }

        /// <summary>
        /// Checks for immediate alerts based on a single sample.
        /// </summary>
        private void CheckForImmediateAlerts(string modelId, string version, MetricSample sample)
        {
            // Check for critical latency spike
            if (sample.LatencyMs > HighLatencyThresholdMs * 2)
            {
                TriggerAlert(VersionAlert.Create(
                    modelId,
                    version,
                    AlertType.HighLatency,
                    $"Critical latency spike: {sample.LatencyMs:F2}ms",
                    AlertSeverity.Critical,
                    new Dictionary<string, object>
                    {
                        { "latency", sample.LatencyMs },
                        { "threshold", HighLatencyThresholdMs * 2 }
                    }
                ));
            }

            // Check for critical memory spike
            if (sample.MemoryUsageMB > MemoryThresholdMB * 1.5)
            {
                TriggerAlert(VersionAlert.Create(
                    modelId,
                    version,
                    AlertType.MemoryExceeded,
                    $"Critical memory spike: {sample.MemoryUsageMB:F2}MB",
                    AlertSeverity.Critical,
                    new Dictionary<string, object>
                    {
                        { "memoryUsage", sample.MemoryUsageMB },
                        { "threshold", MemoryThresholdMB * 1.5 }
                    }
                ));
            }
        }

        /// <summary>
        /// Triggers an alert to all subscribers.
        /// </summary>
        private void TriggerAlert(VersionAlert alert)
        {
            List<Action<VersionAlert>> subscribers;
            lock (_lock)
            {
                subscribers = _alertSubscribers.Values.ToList();
            }

            // Notify subscribers on thread pool
            Task.Run(() =>
            {
                foreach (var subscriber in subscribers)
                {
                    try
                    {
                        subscriber?.Invoke(alert);
                    }
                    catch (Exception)
                    {
                        // Log error but continue notifying other subscribers
                    }
                }
            });
        }

        /// <summary>
        /// Creates a composite key for model ID and version.
        /// </summary>
        private string GetKey(string modelId, string version)
        {
            return $"{modelId}:{version}";
        }

        /// <summary>
        /// Parses a composite key back into model ID and version.
        /// </summary>
        private (string modelId, string version) ParseKey(string key)
        {
            int colonIndex = key.IndexOf(':');
            if (colonIndex < 0)
            {
                return (key, string.Empty);
            }

            string modelId = key.Substring(0, colonIndex);
            string version = key.Substring(colonIndex + 1);
            return (modelId, version);
        }

        /// <summary>
        /// Disposes the version monitor and stops the alert timer.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _alertCheckTimer?.Dispose();
                lock (_lock)
                {
                    _metrics.Clear();
                    _alertSubscribers.Clear();
                }
                _disposed = true;
            }
        }
    }
}
