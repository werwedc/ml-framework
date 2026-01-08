using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Serving;
using Xunit;

namespace MLFramework.Tests.Serving
{
    /// <summary>
    /// Unit tests for the ExperimentTracker class.
    /// </summary>
    public class ExperimentTrackerTests
    {
        [Fact]
        public void StartExperiment_ValidInput_CreatesExperiment()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var modelName = "TestModel";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 0.5f },
                { "v2", 0.5f }
            };

            // Act
            tracker.StartExperiment(experimentId, modelName, versionTraffic);

            // Assert
            var metrics = tracker.GetAllMetrics(experimentId);
            Assert.Equal(2, metrics.Count);
            Assert.Contains("v1", metrics.Keys);
            Assert.Contains("v2", metrics.Keys);
        }

        [Fact]
        public void StartExperiment_DuplicateExperimentId_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var modelName = "TestModel";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };

            // Act
            tracker.StartExperiment(experimentId, modelName, versionTraffic);

            // Assert
            Assert.Throws<InvalidOperationException>(() => 
                tracker.StartExperiment(experimentId, modelName, versionTraffic));
        }

        [Fact]
        public void StartExperiment_InvalidTrafficSplit_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 0.3f },
                { "v2", 0.3f }
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => 
                tracker.StartExperiment("test_exp_1", "TestModel", versionTraffic));
        }

        [Fact]
        public void StartExperiment_NullExperimentId_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => 
                tracker.StartExperiment(null, "TestModel", versionTraffic));
        }

        [Fact]
        public void StartExperiment_NullVersionTraffic_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => 
                tracker.StartExperiment("test_exp_1", "TestModel", null));
        }

        [Fact]
        public void EndExperiment_ActiveExperiment_EndsExperiment()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            // Act
            tracker.EndExperiment(experimentId);

            // Assert
            var metrics = tracker.GetMetrics(experimentId, "v1");
            Assert.NotNull(metrics.EndTime);
            Assert.True(metrics.EndTime >= metrics.StartTime);
        }

        [Fact]
        public void EndExperiment_NonExistentExperiment_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() => tracker.EndExperiment("non_existent"));
        }

        [Fact]
        public void EndExperiment_AlreadyEndedExperiment_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            tracker.EndExperiment(experimentId);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => tracker.EndExperiment(experimentId));
        }

        [Fact]
        public void RecordInference_ValidInput_UpdatesMetrics()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            // Act
            tracker.RecordInference(experimentId, "v1", 100.0, true);

            // Assert
            var metrics = tracker.GetMetrics(experimentId, "v1");
            Assert.Equal(1, metrics.RequestCount);
            Assert.Equal(1, metrics.SuccessCount);
            Assert.Equal(0, metrics.ErrorCount);
            Assert.Equal(100.0, metrics.AverageLatencyMs);
        }

        [Fact]
        public void RecordInference_NonExistentExperiment_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() => 
                tracker.RecordInference("non_existent", "v1", 100.0, true));
        }

        [Fact]
        public void RecordInference_NonExistentVersion_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() => 
                tracker.RecordInference(experimentId, "v2", 100.0, true));
        }

        [Fact]
        public void RecordInference_EndedExperiment_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            tracker.EndExperiment(experimentId);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => 
                tracker.RecordInference(experimentId, "v1", 100.0, true));
        }

        [Fact]
        public void RecordInference_RecordsErrorCorrectly()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            // Act
            tracker.RecordInference(experimentId, "v1", 150.0, false);

            // Assert
            var metrics = tracker.GetMetrics(experimentId, "v1");
            Assert.Equal(1, metrics.RequestCount);
            Assert.Equal(0, metrics.SuccessCount);
            Assert.Equal(1, metrics.ErrorCount);
            Assert.Equal(150.0, metrics.AverageLatencyMs);
        }

        [Fact]
        public void RecordInference_WithCustomMetrics_RecordsCustomMetrics()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            var customMetrics = new Dictionary<string, double>
            {
                { "cpu_usage", 75.5 },
                { "memory_usage", 1024.0 }
            };

            // Act
            tracker.RecordInference(experimentId, "v1", 100.0, true, customMetrics);

            // Assert
            var metrics = tracker.GetMetrics(experimentId, "v1");
            Assert.Equal(75.5, metrics.CustomMetrics["cpu_usage"]);
            Assert.Equal(1024.0, metrics.CustomMetrics["memory_usage"]);
        }

        [Fact]
        public void RecordMultipleInferences_AggregatesCorrectly()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            // Act
            tracker.RecordInference(experimentId, "v1", 100.0, true);
            tracker.RecordInference(experimentId, "v1", 200.0, true);
            tracker.RecordInference(experimentId, "v1", 300.0, true);
            tracker.RecordInference(experimentId, "v1", 150.0, false);

            // Assert
            var metrics = tracker.GetMetrics(experimentId, "v1");
            Assert.Equal(4, metrics.RequestCount);
            Assert.Equal(3, metrics.SuccessCount);
            Assert.Equal(1, metrics.ErrorCount);
            Assert.Equal(187.5, metrics.AverageLatencyMs); // (100 + 200 + 300 + 150) / 4
        }

        [Fact]
        public void CalculatePercentiles_CorrectCalculations()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            // Act - Record 10 inferences with latencies 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
            for (int i = 1; i <= 10; i++)
            {
                tracker.RecordInference(experimentId, "v1", i * 10.0, true);
            }

            // Assert
            var metrics = tracker.GetMetrics(experimentId, "v1");
            Assert.Equal(10, metrics.RequestCount);
            Assert.Equal(50.0, metrics.P50LatencyMs);  // Median
            Assert.Equal(95.0, metrics.P95LatencyMs);  // 95th percentile
            Assert.Equal(100.0, metrics.P99LatencyMs); // 99th percentile
        }

        [Fact]
        public void GetMetrics_NonExistentExperiment_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() => tracker.GetMetrics("non_existent", "v1"));
        }

        [Fact]
        public void GetMetrics_NonExistentVersion_ThrowsException()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() => tracker.GetMetrics(experimentId, "v2"));
        }

        [Fact]
        public void GetAllMetrics_ReturnsAllVersions()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 0.5f },
                { "v2", 0.3f },
                { "v3", 0.2f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            // Act
            var allMetrics = tracker.GetAllMetrics(experimentId);

            // Assert
            Assert.Equal(3, allMetrics.Count);
            Assert.Contains("v1", allMetrics.Keys);
            Assert.Contains("v2", allMetrics.Keys);
            Assert.Contains("v3", allMetrics.Keys);
        }

        [Fact]
        public void CompareVersions_ReturnsComparisons()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 0.5f },
                { "v2", 0.5f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            
            // Record inferences for v1 (faster, lower success rate)
            tracker.RecordInference(experimentId, "v1", 100.0, true);
            tracker.RecordInference(experimentId, "v1", 120.0, true);
            tracker.RecordInference(experimentId, "v1", 80.0, false);
            
            // Record inferences for v2 (slower, higher success rate)
            tracker.RecordInference(experimentId, "v2", 150.0, true);
            tracker.RecordInference(experimentId, "v2", 180.0, true);
            tracker.RecordInference(experimentId, "v2", 200.0, true);

            // Act
            var comparisons = tracker.CompareVersions(experimentId);

            // Assert
            Assert.NotEmpty(comparisons);
            Assert.True(comparisons.ContainsKey("v1_vs_v2_latency_diff_ms"));
            Assert.True(comparisons.ContainsKey("v1_vs_v2_success_rate_diff_pct"));
            Assert.True(comparisons.ContainsKey("v1_vs_v2_error_rate_diff_pct"));
            Assert.True(comparisons.ContainsKey("v1_vs_v2_t_statistic"));
            
            // v2 should be slower than v1 (positive latency diff)
            Assert.True(comparisons["v1_vs_v2_latency_diff_ms"] > 0);
            
            // v2 should have higher success rate than v1 (positive success rate diff)
            Assert.True(comparisons["v1_vs_v2_success_rate_diff_pct"] > 0);
        }

        [Fact]
        public void SuccessRate_CalculatedCorrectly()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            // Act
            tracker.RecordInference(experimentId, "v1", 100.0, true);
            tracker.RecordInference(experimentId, "v1", 100.0, true);
            tracker.RecordInference(experimentId, "v1", 100.0, false);

            // Assert
            var metrics = tracker.GetMetrics(experimentId, "v1");
            Assert.Equal(66.66666666666667, metrics.SuccessRate, 1); // 2/3 * 100
            Assert.Equal(33.33333333333333, metrics.ErrorRate, 1); // 1/3 * 100
        }

        [Fact]
        public void PerformanceTest_Record10000Inferences_CompletesWithinTime()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "perf_test_exp";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            for (int i = 0; i < 10000; i++)
            {
                tracker.RecordInference(experimentId, "v1", 50.0 + (i % 100), true);
            }

            stopwatch.Stop();

            // Assert
            Assert.True(stopwatch.ElapsedMilliseconds < 1000, 
                $"Recording 10,000 inferences took {stopwatch.ElapsedMilliseconds}ms, expected < 1000ms");
            
            var metrics = tracker.GetMetrics(experimentId, "v1");
            Assert.Equal(10000, metrics.RequestCount);
        }

        [Fact]
        public void ConcurrentTracking_100ThreadsRecording_AllMetricsRecorded()
        {
            // Arrange
            var tracker = new ExperimentTracker();
            var experimentId = "concurrent_test_exp";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            tracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            
            var threads = new Thread[100];
            var recordsPerThread = 100;
            var barrier = new Barrier(100);

            // Act
            for (int i = 0; i < threads.Length; i++)
            {
                threads[i] = new Thread(() =>
                {
                    barrier.SignalAndWait(); // Synchronize start
                    for (int j = 0; j < recordsPerThread; j++)
                    {
                        tracker.RecordInference(experimentId, "v1", 100.0, true);
                    }
                });
                threads[i].Start();
            }

            foreach (var thread in threads)
            {
                thread.Join();
            }

            // Assert
            var metrics = tracker.GetMetrics(experimentId, "v1");
            Assert.Equal(threads.Length * recordsPerThread, metrics.RequestCount);
            Assert.Equal(threads.Length * recordsPerThread, metrics.SuccessCount);
            Assert.Equal(0, metrics.ErrorCount);
        }
    }

    /// <summary>
    /// Unit tests for the InferenceTracker class.
    /// </summary>
    public class InferenceTrackerTests
    {
        [Fact]
        public void RecordSuccess_RecordsCorrectly()
        {
            // Arrange
            var experimentTracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            experimentTracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            
            using var tracker = new InferenceTracker(experimentTracker, experimentId, "v1");

            // Act
            tracker.RecordSuccess(100.0);

            // Assert
            var metrics = experimentTracker.GetMetrics(experimentId, "v1");
            Assert.Equal(1, metrics.RequestCount);
            Assert.Equal(1, metrics.SuccessCount);
            Assert.Equal(0, metrics.ErrorCount);
        }

        [Fact]
        public void RecordError_RecordsCorrectly()
        {
            // Arrange
            var experimentTracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            experimentTracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            
            using var tracker = new InferenceTracker(experimentTracker, experimentId, "v1");

            // Act
            tracker.RecordError(150.0, "timeout");

            // Assert
            var metrics = experimentTracker.GetMetrics(experimentId, "v1");
            Assert.Equal(1, metrics.RequestCount);
            Assert.Equal(0, metrics.SuccessCount);
            Assert.Equal(1, metrics.ErrorCount);
            Assert.True(metrics.CustomMetrics.ContainsKey("error_type_timeout"));
        }

        [Fact]
        public void AddCustomMetric_AddsMetricCorrectly()
        {
            // Arrange
            var experimentTracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            experimentTracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            
            using var tracker = new InferenceTracker(experimentTracker, experimentId, "v1");
            tracker.AddCustomMetric("cpu_usage", 75.5);

            // Act
            tracker.RecordSuccess(100.0);

            // Assert
            var metrics = experimentTracker.GetMetrics(experimentId, "v1");
            Assert.Equal(75.5, metrics.CustomMetrics["cpu_usage"]);
        }

        [Fact]
        public void RecordTwice_ThrowsException()
        {
            // Arrange
            var experimentTracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            experimentTracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            
            using var tracker = new InferenceTracker(experimentTracker, experimentId, "v1");
            tracker.RecordSuccess(100.0);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => tracker.RecordSuccess(100.0));
        }

        [Fact]
        public void AddCustomMetricAfterRecording_ThrowsException()
        {
            // Arrange
            var experimentTracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            experimentTracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            
            using var tracker = new InferenceTracker(experimentTracker, experimentId, "v1");
            tracker.RecordSuccess(100.0);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => tracker.AddCustomMetric("test", 1.0));
        }

        [Fact]
        public void DisposeWithoutRecording_AutoRecordsError()
        {
            // Arrange
            var experimentTracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            experimentTracker.StartExperiment(experimentId, "TestModel", versionTraffic);

            // Act
            using (var tracker = new InferenceTracker(experimentTracker, experimentId, "v1"))
            {
                // Dispose without recording
            }

            // Assert
            var metrics = experimentTracker.GetMetrics(experimentId, "v1");
            Assert.Equal(1, metrics.RequestCount);
            Assert.Equal(0, metrics.SuccessCount);
            Assert.Equal(1, metrics.ErrorCount);
            Assert.True(metrics.CustomMetrics.ContainsKey("error_type_disposal_timeout"));
        }

        [Fact]
        public void NegativeLatency_ThrowsException()
        {
            // Arrange
            var experimentTracker = new ExperimentTracker();
            var experimentId = "test_exp_1";
            var versionTraffic = new Dictionary<string, float>
            {
                { "v1", 1.0f }
            };
            experimentTracker.StartExperiment(experimentId, "TestModel", versionTraffic);
            
            using var tracker = new InferenceTracker(experimentTracker, experimentId, "v1");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => tracker.RecordSuccess(-10.0));
        }

        [Fact]
        public void NullExperimentId_ThrowsException()
        {
            // Arrange
            var experimentTracker = new ExperimentTracker();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                new InferenceTracker(experimentTracker, null, "v1"));
        }

        [Fact]
        public void NullVersion_ThrowsException()
        {
            // Arrange
            var experimentTracker = new ExperimentTracker();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                new InferenceTracker(experimentTracker, "test_exp_1", null));
        }
    }
}
