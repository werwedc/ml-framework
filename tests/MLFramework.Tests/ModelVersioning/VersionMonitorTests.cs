using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using MLFramework.ModelVersioning;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Unit tests for VersionMonitor.
    /// </summary>
    public class VersionMonitorTests : IDisposable
    {
        private readonly VersionMonitor _monitor;
        private readonly List<VersionAlert> _capturedAlerts;
        private readonly Action<VersionAlert> _alertHandler;

        public VersionMonitorTests()
        {
            _monitor = new VersionMonitor();
            _capturedAlerts = new List<VersionAlert>();
            _alertHandler = alert => _capturedAlerts.Add(alert);
            _monitor.SubscribeToAlerts(_alertHandler);

            // Set thresholds for testing
            _monitor.HighLatencyThresholdMs = 500.0;
            _monitor.HighErrorRateThreshold = 0.1; // 10%
            _monitor.LowThroughputThreshold = 5.0;
            _monitor.MemoryThresholdMB = 512.0;
        }

        public void Dispose()
        {
            _monitor?.Dispose();
        }

        [Fact]
        public void RecordMetric_StoresSamples()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";
            var sample = MetricSample.Create(100.0, true, 200.0);

            // Act
            _monitor.RecordMetric(modelId, version, sample);

            // Assert
            var metrics = _monitor.GetMetrics(modelId, version);
            Assert.Equal(1, metrics.TotalRequests);
        }

        [Fact]
        public void GetMetrics_AggregatesSamplesCorrectly()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";

            // Record multiple samples
            _monitor.RecordMetric(modelId, version, MetricSample.Create(100.0, true, 200.0));
            _monitor.RecordMetric(modelId, version, MetricSample.Create(150.0, true, 210.0));
            _monitor.RecordMetric(modelId, version, MetricSample.Create(200.0, false, 220.0));

            // Act
            var metrics = _monitor.GetMetrics(modelId, version);

            // Assert
            Assert.Equal(3, metrics.TotalRequests);
            Assert.Equal(150.0, metrics.AverageLatencyMs);
            Assert.Equal(1.0 / 3.0, metrics.ErrorRate, 2); // 1 error out of 3
            Assert.Equal(210.0, metrics.MemoryUsageMB, 1);
        }

        [Fact]
        public void GetMetrics_CalculatesPercentilesCorrectly()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";

            // Record samples with known latency values
            var latencies = new[] { 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0 };
            foreach (var latency in latencies)
            {
                _monitor.RecordMetric(modelId, version, MetricSample.Create(latency, true, 200.0));
            }

            // Act
            var metrics = _monitor.GetMetrics(modelId, version);

            // Assert
            Assert.Equal(10, metrics.TotalRequests);
            Assert.Equal(550.0, metrics.AverageLatencyMs); // (100 + 1000) / 2
            Assert.Equal(500.0, metrics.P50LatencyMs); // Median
            Assert.Equal(950.0, metrics.P95LatencyMs);
            Assert.Equal(1000.0, metrics.P99LatencyMs);
        }

        [Fact]
        public void CompareVersions_CalculatesDeltasCorrectly()
        {
            // Arrange
            string modelId = "test-model";

            // Version 1: Slower, higher error rate
            _monitor.RecordMetric(modelId, "v1.0", MetricSample.Create(200.0, true, 200.0));
            _monitor.RecordMetric(modelId, "v1.0", MetricSample.Create(200.0, false, 200.0));

            // Version 2: Faster, lower error rate
            _monitor.RecordMetric(modelId, "v2.0", MetricSample.Create(100.0, true, 200.0));
            _monitor.RecordMetric(modelId, "v2.0", MetricSample.Create(100.0, true, 200.0));

            // Act
            var comparison = _monitor.CompareVersions(modelId, "v1.0", "v2.0");

            // Assert
            Assert.NotNull(comparison);
            Assert.NotNull(comparison.Version1);
            Assert.NotNull(comparison.Version2);
            Assert.Equal("v1.0", comparison.Version1.Version);
            Assert.Equal("v2.0", comparison.Version2.Version);
            Assert.Equal(200.0, comparison.Version1.AverageLatencyMs);
            Assert.Equal(100.0, comparison.Version2.AverageLatencyMs);
            Assert.Equal(0.5, comparison.Version1.ErrorRate); // 1 error out of 2
            Assert.Equal(0.0, comparison.Version2.ErrorRate); // 0 errors
        }

        [Fact]
        public void CompareVersions_DeterminesDirectionCorrectly()
        {
            // Arrange
            string modelId = "test-model";

            // Version 1: Slower, higher error rate, lower throughput
            _monitor.RecordMetric(modelId, "v1.0", MetricSample.Create(200.0, true, 200.0));
            _monitor.RecordMetric(modelId, "v1.0", MetricSample.Create(200.0, false, 200.0));

            // Version 2: Faster, lower error rate, higher throughput
            _monitor.RecordMetric(modelId, "v2.0", MetricSample.Create(100.0, true, 200.0));
            _monitor.RecordMetric(modelId, "v2.0", MetricSample.Create(100.0, true, 200.0));

            // Act
            var comparison = _monitor.CompareVersions(modelId, "v1.0", "v2.0");

            // Assert
            Assert.Equal("better", comparison.LatencyDelta.Direction); // Lower latency is better
            Assert.Equal("better", comparison.ErrorRateDelta.Direction); // Lower error rate is better
            Assert.True(comparison.IsVersion2Better());
        }

        [Fact]
        public void CompareVersions_DetectsWorsePerformance()
        {
            // Arrange
            string modelId = "test-model";

            // Version 1: Better performance
            _monitor.RecordMetric(modelId, "v1.0", MetricSample.Create(100.0, true, 200.0));
            _monitor.RecordMetric(modelId, "v1.0", MetricSample.Create(100.0, true, 200.0));

            // Version 2: Worse performance
            _monitor.RecordMetric(modelId, "v2.0", MetricSample.Create(200.0, false, 200.0));
            _monitor.RecordMetric(modelId, "v2.0", MetricSample.Create(200.0, false, 200.0));

            // Act
            var comparison = _monitor.CompareVersions(modelId, "v1.0", "v2.0");

            // Assert
            Assert.Equal("worse", comparison.LatencyDelta.Direction); // Higher latency is worse
            Assert.Equal("worse", comparison.ErrorRateDelta.Direction); // Higher error rate is worse
            Assert.False(comparison.IsVersion2Better());
        }

        [Fact]
        public async Task SubscribeToAlerts_ReceivesNotifications()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";
            _capturedAlerts.Clear();

            // Record samples that will trigger an alert
            for (int i = 0; i < 10; i++)
            {
                _monitor.RecordMetric(modelId, version, MetricSample.Create(600.0, true, 200.0));
            }

            // Wait for alert check timer to run
            await Task.Delay(35000); // 30 seconds + buffer

            // Assert
            Assert.NotEmpty(_capturedAlerts);
            Assert.Contains(_capturedAlerts, a => a.ModelId == modelId && a.Version == version);
        }

        [Fact]
        public async Task UnsubscribeFromAlerts_StopsNotifications()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";
            _capturedAlerts.Clear();

            // Unsubscribe from alerts
            _monitor.UnsubscribeFromAlerts(_alertHandler);

            // Record samples that would trigger an alert
            for (int i = 0; i < 10; i++)
            {
                _monitor.RecordMetric(modelId, version, MetricSample.Create(600.0, true, 200.0));
            }

            // Wait for alert check timer to run
            await Task.Delay(35000);

            // Assert - Should not have received any alerts
            Assert.Empty(_capturedAlerts);
        }

        [Fact]
        public async Task AlertDetection_HighLatency()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";
            _capturedAlerts.Clear();

            // Record samples with high latency
            for (int i = 0; i < 10; i++)
            {
                _monitor.RecordMetric(modelId, version, MetricSample.Create(600.0, true, 200.0));
            }

            // Wait for alert check timer
            await Task.Delay(35000);

            // Assert
            var highLatencyAlerts = _capturedAlerts.Where(a => a.Type == AlertType.HighLatency).ToList();
            Assert.NotEmpty(highLatencyAlerts);
        }

        [Fact]
        public async Task AlertDetection_HighErrorRate()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";
            _capturedAlerts.Clear();

            // Record samples with high error rate (50%)
            for (int i = 0; i < 10; i++)
            {
                _monitor.RecordMetric(modelId, version, MetricSample.Create(100.0, i % 2 == 0, 200.0));
            }

            // Wait for alert check timer
            await Task.Delay(35000);

            // Assert
            var highErrorRateAlerts = _capturedAlerts.Where(a => a.Type == AlertType.HighErrorRate).ToList();
            Assert.NotEmpty(highErrorRateAlerts);
        }

        [Fact]
        public async Task AlertDetection_LowThroughput()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";
            _capturedAlerts.Clear();

            // Record samples over a long period to create low throughput
            var startTime = DateTime.UtcNow;
            for (int i = 0; i < 5; i++)
            {
                _monitor.RecordMetric(modelId, version, MetricSample.Create(100.0, true, 200.0));
                await Task.Delay(5000); // 5 seconds between requests
            }

            // Wait for alert check timer
            await Task.Delay(35000);

            // Assert
            var lowThroughputAlerts = _capturedAlerts.Where(a => a.Type == AlertType.LowThroughput).ToList();
            Assert.NotEmpty(lowThroughputAlerts);
        }

        [Fact]
        public void ClearMetrics_RemovesData()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";

            // Record samples
            _monitor.RecordMetric(modelId, version, MetricSample.Create(100.0, true, 200.0));
            _monitor.RecordMetric(modelId, version, MetricSample.Create(150.0, true, 210.0));

            // Verify metrics exist
            var metricsBefore = _monitor.GetMetrics(modelId, version);
            Assert.Equal(2, metricsBefore.TotalRequests);

            // Act
            _monitor.ClearMetrics(modelId, version);

            // Assert
            var metricsAfter = _monitor.GetMetrics(modelId, version);
            Assert.Equal(0, metricsAfter.TotalRequests);
        }

        [Fact]
        public void ConcurrentMetricRecording_ThreadSafe()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";
            int numThreads = 10;
            int samplesPerThread = 100;

            // Act - Record metrics from multiple threads concurrently
            var tasks = new Task[numThreads];
            for (int i = 0; i < numThreads; i++)
            {
                int threadId = i;
                tasks[i] = Task.Run(() =>
                {
                    for (int j = 0; j < samplesPerThread; j++)
                    {
                        _monitor.RecordMetric(modelId, version, MetricSample.Create(
                            100.0 + threadId,
                            true,
                            200.0 + threadId
                        ));
                    }
                });
            }

            Task.WaitAll(tasks);

            // Assert
            var metrics = _monitor.GetMetrics(modelId, version);
            Assert.Equal(numThreads * samplesPerThread, metrics.TotalRequests);
        }

        [Fact]
        public void MultipleAlertSubscribers_AllReceiveNotifications()
        {
            // Arrange
            var monitor = new VersionMonitor();
            var alerts1 = new List<VersionAlert>();
            var alerts2 = new List<VersionAlert>();

            Action<VersionAlert> handler1 = alert => alerts1.Add(alert);
            Action<VersionAlert> handler2 = alert => alerts2.Add(alert);

            monitor.SubscribeToAlerts(handler1);
            monitor.SubscribeToAlerts(handler2);

            // Record a sample that triggers an immediate alert
            monitor.RecordMetric("test-model", "v1.0", MetricSample.Create(1100.0, true, 200.0));

            // Give time for async alert processing
            Thread.Sleep(100);

            // Assert
            Assert.NotEmpty(alerts1);
            Assert.NotEmpty(alerts2);
            Assert.Equal(alerts1.Count, alerts2.Count);

            monitor.Dispose();
        }

        [Fact]
        public void RecordMetric_NullModelId_ThrowsException()
        {
            // Arrange
            string modelId = null;
            string version = "v1.0";
            var sample = MetricSample.Create(100.0, true, 200.0);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _monitor.RecordMetric(modelId, version, sample));
        }

        [Fact]
        public void RecordMetric_EmptyModelId_ThrowsException()
        {
            // Arrange
            string modelId = "";
            string version = "v1.0";
            var sample = MetricSample.Create(100.0, true, 200.0);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _monitor.RecordMetric(modelId, version, sample));
        }

        [Fact]
        public void RecordMetric_NullVersion_ThrowsException()
        {
            // Arrange
            string modelId = "test-model";
            string version = null;
            var sample = MetricSample.Create(100.0, true, 200.0);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _monitor.RecordMetric(modelId, version, sample));
        }

        [Fact]
        public void RecordMetric_NullSample_ThrowsException()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";
            MetricSample sample = null;

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _monitor.RecordMetric(modelId, version, sample));
        }

        [Fact]
        public void CompareVersions_NullModelId_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _monitor.CompareVersions(null, "v1.0", "v2.0"));
        }

        [Fact]
        public void CompareVersions_NullV1_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _monitor.CompareVersions("test-model", null, "v2.0"));
        }

        [Fact]
        public void CompareVersions_NullV2_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _monitor.CompareVersions("test-model", "v1.0", null));
        }

        [Fact]
        public void SubscribeToAlerts_NullCallback_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _monitor.SubscribeToAlerts(null));
        }

        [Fact]
        public void UnsubscribeFromAlerts_NullCallback_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _monitor.UnsubscribeFromAlerts(null));
        }

        [Fact]
        public void ClearMetrics_NullModelId_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _monitor.ClearMetrics(null, "v1.0"));
        }

        [Fact]
        public void ClearMetrics_NullVersion_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _monitor.ClearMetrics("test-model", null));
        }

        [Fact]
        public void GetMetrics_ReturnsEmptyMetricsForNonExistentVersion()
        {
            // Arrange
            string modelId = "non-existent-model";
            string version = "v1.0";

            // Act
            var metrics = _monitor.GetMetrics(modelId, version);

            // Assert
            Assert.NotNull(metrics);
            Assert.Equal(modelId, metrics.ModelId);
            Assert.Equal(version, metrics.Version);
            Assert.Equal(0, metrics.TotalRequests);
        }

        [Fact]
        public void GetMetrics_CalculatesThroughputCorrectly()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";

            var startTime = DateTime.UtcNow;
            _monitor.RecordMetric(modelId, version, MetricSample.Create(100.0, true, 200.0));
            Task.Delay(100).Wait();
            _monitor.RecordMetric(modelId, version, MetricSample.Create(100.0, true, 200.0));
            Task.Delay(100).Wait();
            _monitor.RecordMetric(modelId, version, MetricSample.Create(100.0, true, 200.0));

            // Act
            var metrics = _monitor.GetMetrics(modelId, version);

            // Assert
            Assert.Equal(3, metrics.TotalRequests);
            // 3 requests in ~0.2 seconds should be around 15 req/s
            Assert.True(metrics.Throughput > 10.0 && metrics.Throughput < 20.0);
        }

        [Fact]
        public void VersionMonitor_DisposesCorrectly()
        {
            // Arrange
            var monitor = new VersionMonitor();

            // Act
            monitor.Dispose();

            // Assert - Should not throw exception
            monitor.Dispose();
        }

        [Fact]
        public void GetMetrics_SingleSampleCalculations()
        {
            // Arrange
            string modelId = "test-model";
            string version = "v1.0";
            var sample = MetricSample.Create(150.0, true, 200.0);

            // Act
            _monitor.RecordMetric(modelId, version, sample);
            var metrics = _monitor.GetMetrics(modelId, version);

            // Assert
            Assert.Equal(1, metrics.TotalRequests);
            Assert.Equal(150.0, metrics.AverageLatencyMs);
            Assert.Equal(150.0, metrics.P50LatencyMs);
            Assert.Equal(150.0, metrics.P95LatencyMs);
            Assert.Equal(150.0, metrics.P99LatencyMs);
            Assert.Equal(0.0, metrics.ErrorRate);
        }
    }
}
