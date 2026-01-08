using System;
using System.Text.Json;
using Xunit;
using MLFramework.ModelVersioning;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Unit tests for monitoring data models.
    /// </summary>
    public class MonitoringDataModelsTests
    {
        [Fact]
        public void VersionMetrics_Creation_WithValidData()
        {
            // Arrange & Act
            var metrics = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 50.5,
                P50LatencyMs = 45.0,
                P95LatencyMs = 120.0,
                P99LatencyMs = 200.0,
                ErrorRate = 0.01,
                Throughput = 100.0,
                MemoryUsageMB = 500.0
            };

            // Assert
            Assert.Equal("model-123", metrics.ModelId);
            Assert.Equal("v1.0.0", metrics.Version);
            Assert.True(metrics.IsValid());
        }

        [Fact]
        public void VersionMetrics_InvalidTimeRange_Invalidates()
        {
            // Arrange & Act
            var metrics = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = DateTime.UtcNow,
                EndTime = DateTime.UtcNow.AddMinutes(-10), // End time before start time
                TotalRequests = 1000,
                AverageLatencyMs = 50.0,
                ErrorRate = 0.01,
                Throughput = 100.0,
                MemoryUsageMB = 500.0
            };

            // Assert
            Assert.False(metrics.IsValid());
        }

        [Fact]
        public void VersionMetrics_NegativeValues_Invalidates()
        {
            // Arrange & Act
            var metrics = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = -10.0, // Negative latency
                ErrorRate = 0.01,
                Throughput = 100.0,
                MemoryUsageMB = 500.0
            };

            // Assert
            Assert.False(metrics.IsValid());
        }

        [Fact]
        public void VersionMetrics_ErrorRateOutOfBounds_Invalidates()
        {
            // Arrange & Act
            var metrics = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 50.0,
                ErrorRate = 1.5, // Error rate > 1
                Throughput = 100.0,
                MemoryUsageMB = 500.0
            };

            // Assert
            Assert.False(metrics.IsValid());
        }

        [Fact]
        public void VersionMetrics_GetDuration_ReturnsCorrectSpan()
        {
            // Arrange
            var startTime = DateTime.UtcNow.AddMinutes(-10);
            var endTime = DateTime.UtcNow;
            var metrics = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = startTime,
                EndTime = endTime,
                TotalRequests = 1000,
                AverageLatencyMs = 50.0,
                ErrorRate = 0.01,
                Throughput = 100.0,
                MemoryUsageMB = 500.0
            };

            // Act
            var duration = metrics.GetDuration();

            // Assert
            Assert.Equal(TimeSpan.FromMinutes(10), duration);
        }

        [Fact]
        public void VersionMetrics_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 50.0,
                ErrorRate = 0.01,
                Throughput = 100.0,
                MemoryUsageMB = 500.0
            };

            // Act
            var clone = original.Clone();
            clone.AverageLatencyMs = 75.0;

            // Assert
            Assert.Equal(50.0, original.AverageLatencyMs);
            Assert.Equal(75.0, clone.AverageLatencyMs);
        }

        [Fact]
        public void MetricComparison_Calculation_CorrectDeltaAndPercentage()
        {
            // Arrange
            var version1 = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 100.0,
                P50LatencyMs = 90.0,
                P95LatencyMs = 200.0,
                P99LatencyMs = 300.0,
                ErrorRate = 0.05,
                Throughput = 50.0,
                MemoryUsageMB = 500.0
            };

            var version2 = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v2.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 80.0, // 20% improvement
                P50LatencyMs = 75.0,
                P95LatencyMs = 160.0,
                P99LatencyMs = 240.0,
                ErrorRate = 0.03, // 40% improvement
                Throughput = 60.0, // 20% improvement
                MemoryUsageMB = 480.0
            };

            // Act
            var comparison = MetricComparison.Create(version1, version2);

            // Assert
            Assert.NotNull(comparison);
            Assert.Equal(-20.0, comparison.LatencyDelta.AbsoluteDifference);
            Assert.Equal(-20.0, comparison.LatencyDelta.PercentageChange);
            Assert.Equal("better", comparison.LatencyDelta.Direction);

            Assert.Equal(-0.02, comparison.ErrorRateDelta.AbsoluteDifference);
            Assert.Equal(-40.0, comparison.ErrorRateDelta.PercentageChange);
            Assert.Equal("better", comparison.ErrorRateDelta.Direction);

            Assert.Equal(10.0, comparison.ThroughputDelta.AbsoluteDifference);
            Assert.Equal(20.0, comparison.ThroughputDelta.PercentageChange);
            Assert.Equal("better", comparison.ThroughputDelta.Direction);
        }

        [Fact]
        public void MetricDelta_Direction_BetterForLowerLatency()
        {
            // Arrange & Act
            var delta = MetricDelta.CreateForLatency(100.0, 80.0);

            // Assert
            Assert.Equal(-20.0, delta.AbsoluteDifference);
            Assert.Equal(-20.0, delta.PercentageChange);
            Assert.Equal("better", delta.Direction);
        }

        [Fact]
        public void MetricDelta_Direction_WorseForHigherLatency()
        {
            // Arrange & Act
            var delta = MetricDelta.CreateForLatency(80.0, 100.0);

            // Assert
            Assert.Equal(20.0, delta.AbsoluteDifference);
            Assert.Equal(25.0, delta.PercentageChange);
            Assert.Equal("worse", delta.Direction);
        }

        [Fact]
        public void MetricDelta_Direction_NeutralForEqualValues()
        {
            // Arrange & Act
            var delta = MetricDelta.CreateForLatency(100.0, 100.0);

            // Assert
            Assert.Equal(0.0, delta.AbsoluteDifference);
            Assert.Equal(0.0, delta.PercentageChange);
            Assert.Equal("neutral", delta.Direction);
        }

        [Fact]
        public void MetricDelta_ErrorRate_BetterForLowerRate()
        {
            // Arrange & Act
            var delta = MetricDelta.CreateForErrorRate(0.05, 0.02);

            // Assert
            Assert.Equal(-0.03, delta.AbsoluteDifference);
            Assert.Equal(-60.0, delta.PercentageChange);
            Assert.Equal("better", delta.Direction);
        }

        [Fact]
        public void MetricDelta_Throughput_BetterForHigherRate()
        {
            // Arrange & Act
            var delta = MetricDelta.CreateForThroughput(50.0, 75.0);

            // Assert
            Assert.Equal(25.0, delta.AbsoluteDifference);
            Assert.Equal(50.0, delta.PercentageChange);
            Assert.Equal("better", delta.Direction);
        }

        [Fact]
        public void MetricDelta_Throughput_WorseForLowerRate()
        {
            // Arrange & Act
            var delta = MetricDelta.CreateForThroughput(75.0, 50.0);

            // Assert
            Assert.Equal(-25.0, delta.AbsoluteDifference);
            Assert.Equal(-33.333, delta.PercentageChange, 3);
            Assert.Equal("worse", delta.Direction);
        }

        [Fact]
        public void MetricComparison_IsVersion2Better_WithAllBetterMetrics()
        {
            // Arrange
            var version1 = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 100.0,
                ErrorRate = 0.05,
                Throughput = 50.0,
                MemoryUsageMB = 500.0
            };

            var version2 = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v2.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 80.0,
                ErrorRate = 0.03,
                Throughput = 60.0,
                MemoryUsageMB = 480.0
            };

            // Act
            var comparison = MetricComparison.Create(version1, version2);

            // Assert
            Assert.True(comparison.IsVersion2Better());
        }

        [Fact]
        public void MetricComparison_IsVersion2Better_WithMixedMetrics()
        {
            // Arrange
            var version1 = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 100.0,
                ErrorRate = 0.05,
                Throughput = 50.0,
                MemoryUsageMB = 500.0
            };

            var version2 = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v2.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 80.0, // Better
                ErrorRate = 0.06, // Worse
                Throughput = 60.0, // Better
                MemoryUsageMB = 480.0
            };

            // Act
            var comparison = MetricComparison.Create(version1, version2);

            // Assert
            Assert.False(comparison.IsVersion2Better());
        }

        [Fact]
        public void VersionAlert_Creation_ForEachAlertType()
        {
            // Arrange
            var modelId = "model-123";
            var version = "v1.0.0";

            // Act & Assert
            var latencyAlert = VersionAlert.Create(modelId, version, AlertType.HighLatency, "Latency exceeds threshold", AlertSeverity.Critical);
            Assert.Equal(AlertType.HighLatency, latencyAlert.Type);
            Assert.Equal(AlertSeverity.Critical, latencyAlert.Severity);
            Assert.True(latencyAlert.IsValid());

            var errorAlert = VersionAlert.Create(modelId, version, AlertType.HighErrorRate, "Error rate exceeds threshold", AlertSeverity.Warning);
            Assert.Equal(AlertType.HighErrorRate, errorAlert.Type);
            Assert.Equal(AlertSeverity.Warning, errorAlert.Severity);
            Assert.True(errorAlert.IsValid());

            var throughputAlert = VersionAlert.Create(modelId, version, AlertType.LowThroughput, "Throughput below threshold", AlertSeverity.Warning);
            Assert.Equal(AlertType.LowThroughput, throughputAlert.Type);
            Assert.Equal(AlertSeverity.Warning, throughputAlert.Severity);
            Assert.True(throughputAlert.IsValid());

            var memoryAlert = VersionAlert.Create(modelId, version, AlertType.MemoryExceeded, "Memory usage exceeded", AlertSeverity.Critical);
            Assert.Equal(AlertType.MemoryExceeded, memoryAlert.Type);
            Assert.Equal(AlertSeverity.Critical, memoryAlert.Severity);
            Assert.True(memoryAlert.IsValid());

            var anomalyAlert = VersionAlert.Create(modelId, version, AlertType.AnomalyDetected, "Statistical anomaly detected", AlertSeverity.Info);
            Assert.Equal(AlertType.AnomalyDetected, anomalyAlert.Type);
            Assert.Equal(AlertSeverity.Info, anomalyAlert.Severity);
            Assert.True(anomalyAlert.IsValid());
        }

        [Fact]
        public void VersionAlert_CreationWithContext_StoresContextCorrectly()
        {
            // Arrange
            var context = new Dictionary<string, object>
            {
                { "threshold", 100.0 },
                { "actualValue", 150.0 },
                { "exceededBy", 50.0 }
            };

            // Act
            var alert = VersionAlert.Create("model-123", "v1.0.0", AlertType.HighLatency, "Latency exceeded", AlertSeverity.Warning, context);

            // Assert
            Assert.Equal(3, alert.Context.Count);
            Assert.Equal(100.0, alert.Context["threshold"]);
            Assert.Equal(150.0, alert.Context["actualValue"]);
            Assert.Equal(50.0, alert.Context["exceededBy"]);
        }

        [Fact]
        public void VersionAlert_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = VersionAlert.Create("model-123", "v1.0.0", AlertType.HighLatency, "Latency exceeded", AlertSeverity.Warning);
            original.Context.Add("key", "value");

            // Act
            var clone = original.Clone();
            clone.Context["key"] = "modified";

            // Assert
            Assert.Equal("value", original.Context["key"]);
            Assert.Equal("modified", clone.Context["key"]);
            Assert.Equal(original.AlertId, clone.AlertId);
        }

        [Fact]
        public void VersionAlert_ToString_FormatsCorrectly()
        {
            // Arrange
            var alert = VersionAlert.Create("model-123", "v1.0.0", AlertType.HighLatency, "Latency exceeded threshold", AlertSeverity.Critical);

            // Act
            var alertString = alert.ToString();

            // Assert
            Assert.Contains("[Critical]", alertString);
            Assert.Contains("HighLatency", alertString);
            Assert.Contains("Latency exceeded threshold", alertString);
            Assert.Contains("model-123", alertString);
            Assert.Contains("v1.0.0", alertString);
        }

        [Fact]
        public void MetricSample_Creation_WithCreateMethod()
        {
            // Arrange & Act
            var sample = MetricSample.Create(150.5, true, 512.0);

            // Assert
            Assert.True(sample.Timestamp <= DateTime.UtcNow);
            Assert.True(sample.Timestamp > DateTime.UtcNow.AddSeconds(-1));
            Assert.Equal(150.5, sample.LatencyMs);
            Assert.True(sample.Success);
            Assert.Equal(512.0, sample.MemoryUsageMB);
        }

        [Fact]
        public void MetricSample_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = MetricSample.Create(150.5, true, 512.0);

            // Act
            var clone = original.Clone();
            clone.LatencyMs = 200.0;

            // Assert
            Assert.Equal(150.5, original.LatencyMs);
            Assert.Equal(200.0, clone.LatencyMs);
            Assert.NotEqual(original.Timestamp, clone.Timestamp); // Timestamp should be copied as-is
        }

        [Fact]
        public void VersionMetrics_JsonSerialization_DeserializesCorrectly()
        {
            // Arrange
            var original = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 50.5,
                P50LatencyMs = 45.0,
                P95LatencyMs = 120.0,
                P99LatencyMs = 200.0,
                ErrorRate = 0.01,
                Throughput = 100.0,
                MemoryUsageMB = 500.0
            };

            // Act
            var json = JsonSerializer.Serialize(original);
            var deserialized = JsonSerializer.Deserialize<VersionMetrics>(json);

            // Assert
            Assert.NotNull(deserialized);
            Assert.Equal(original.ModelId, deserialized.ModelId);
            Assert.Equal(original.Version, deserialized.Version);
            Assert.Equal(original.TotalRequests, deserialized.TotalRequests);
            Assert.Equal(original.AverageLatencyMs, deserialized.AverageLatencyMs);
            Assert.Equal(original.ErrorRate, deserialized.ErrorRate);
        }

        [Fact]
        public void MetricDelta_PercentageChange_EdgeCaseZeroBaseline()
        {
            // Arrange & Act
            var delta = MetricDelta.CreateForLatency(0.0, 50.0);

            // Assert
            Assert.Equal(50.0, delta.AbsoluteDifference);
            Assert.Equal(0.0, delta.PercentageChange); // Can't calculate percentage when baseline is 0
            Assert.Equal("worse", delta.Direction);
        }

        [Fact]
        public void MetricDelta_PercentageChange_EdgeCaseSmallBaseline()
        {
            // Arrange & Act
            var delta = MetricDelta.CreateForLatency(0.001, 0.002);

            // Assert
            Assert.Equal(0.001, delta.AbsoluteDifference);
            Assert.Equal(100.0, delta.PercentageChange);
            Assert.Equal("worse", delta.Direction);
        }

        [Fact]
        public void MetricComparison_GetSummary_ReturnsCorrectData()
        {
            // Arrange
            var version1 = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 100.0,
                ErrorRate = 0.05,
                Throughput = 50.0,
                MemoryUsageMB = 500.0
            };

            var version2 = new VersionMetrics
            {
                ModelId = "model-123",
                Version = "v2.0.0",
                StartTime = DateTime.UtcNow.AddMinutes(-10),
                EndTime = DateTime.UtcNow,
                TotalRequests = 1000,
                AverageLatencyMs = 80.0,
                ErrorRate = 0.03,
                Throughput = 60.0,
                MemoryUsageMB = 480.0
            };

            var comparison = MetricComparison.Create(version1, version2);

            // Act
            var summary = comparison.GetSummary();

            // Assert
            Assert.Equal("v1.0.0", summary["version1"]);
            Assert.Equal("v2.0.0", summary["version2"]);
            Assert.Equal(-20.0, summary["latencyDelta"]);
            Assert.Equal(-20.0, summary["latencyPercentChange"]);
            Assert.Equal("better", summary["latencyDirection"]);
            Assert.True((bool)summary["isVersion2Better"]);
        }

        [Fact]
        public void VersionAlert_InvalidWithoutRequiredFields()
        {
            // Arrange
            var alert = new VersionAlert
            {
                ModelId = "model-123",
                Version = "v1.0.0",
                Type = AlertType.HighLatency,
                Message = "Test",
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>(),
                Severity = AlertSeverity.Warning
                // Missing AlertId
            };

            // Act
            var isValid = alert.IsValid();

            // Assert
            Assert.False(isValid);
        }
    }
}
