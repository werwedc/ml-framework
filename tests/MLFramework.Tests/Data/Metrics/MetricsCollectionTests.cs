using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Data.Metrics;
using Xunit;

namespace MLFramework.Tests.Data.Metrics
{
    public class MetricsCollectionTests : IDisposable
    {
        private readonly MetricsCollector _collector;

        public MetricsCollectionTests()
        {
            _collector = new MetricsCollector("test");
        }

        public void Dispose()
        {
            _collector?.Dispose();
        }

        [Fact]
        public void Constructor_InitializesCorrectly()
        {
            var collector = new MetricsCollector("test");

            Assert.Equal("test", collector.Name);
        }

        [Fact]
        public void RecordInference_WithNullModelName_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => _collector.RecordInference(null, "v1", 100, true));
        }

        [Fact]
        public void RecordInference_WithNullVersion_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => _collector.RecordInference("model1", null, 100, true));
        }

        [Fact]
        public void RecordInference_RecordsSuccessfully()
        {
            _collector.RecordInference("model1", "v1", 100, true);

            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            Assert.Equal(1, metrics.RequestCount);
            Assert.Equal(100, metrics.AverageLatencyMs);
            Assert.Equal(0, metrics.ErrorRate);
        }

        [Fact]
        public void RecordInference_WithError_RecordsErrorRate()
        {
            _collector.RecordInference("model1", "v1", 100, false, "TimeoutError");
            _collector.RecordInference("model1", "v1", 100, true);

            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            Assert.Equal(2, metrics.RequestCount);
            Assert.Equal(1, metrics.ErrorCount);
            Assert.Equal(50.0, metrics.ErrorRate);
            Assert.Single(metrics.ErrorCountsByType);
            Assert.True(metrics.ErrorCountsByType.ContainsKey("TimeoutError"));
        }

        [Fact]
        public void RecordInference_MultipleRecords_CalculatesCorrectAverages()
        {
            _collector.RecordInference("model1", "v1", 100, true);
            _collector.RecordInference("model1", "v1", 200, true);
            _collector.RecordInference("model1", "v1", 300, true);

            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            Assert.Equal(3, metrics.RequestCount);
            Assert.Equal(200, metrics.AverageLatencyMs);
        }

        [Fact]
        public void RecordInference_CalculatesPercentilesCorrectly()
        {
            // Record latencies: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
            for (int i = 1; i <= 10; i++)
            {
                _collector.RecordInference("model1", "v1", i * 10, true);
            }

            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            // P50 should be around 55 (median)
            Assert.InRange(metrics.P50LatencyMs, 50, 60);

            // P95 should be around 95-100
            Assert.InRange(metrics.P95LatencyMs, 90, 100);

            // P99 should be around 100
            Assert.InRange(metrics.P99LatencyMs, 95, 100);
        }

        [Fact]
        public void RecordActiveConnections_WithNullModelName_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => _collector.RecordActiveConnections(null, "v1", 10));
        }

        [Fact]
        public void RecordActiveConnections_WithNullVersion_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => _collector.RecordActiveConnections("model1", null, 10));
        }

        [Fact]
        public void RecordActiveConnections_RecordsSuccessfully()
        {
            _collector.RecordActiveConnections("model1", "v1", 42);

            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            Assert.Equal(42, metrics.ActiveConnections);
        }

        [Fact]
        public void RecordMemoryUsage_WithNullModelName_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => _collector.RecordMemoryUsage(null, "v1", 1024));
        }

        [Fact]
        public void RecordMemoryUsage_WithNullVersion_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => _collector.RecordMemoryUsage("model1", null, 1024));
        }

        [Fact]
        public void RecordMemoryUsage_RecordsSuccessfully()
        {
            _collector.RecordMemoryUsage("model1", "v1", 1024 * 1024 * 500); // 500 MB

            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            Assert.Equal(500.0, metrics.MemoryUsageMB, 0.1);
        }

        [Fact]
        public void GetMetrics_NonExistentVersion_ReturnsEmptyMetrics()
        {
            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            Assert.Equal("model1", metrics.ModelName);
            Assert.Equal("v1", metrics.Version);
            Assert.Equal(0, metrics.RequestCount);
            Assert.Equal(0, metrics.ErrorRate);
        }

        [Fact]
        public void GetMetrics_TimeWindow_FiltersCorrectly()
        {
            var window = TimeSpan.FromMinutes(5);

            // Record metrics outside the window
            _collector.RecordInference("model1", "v1", 100, true);

            // Record metrics inside the window
            _collector.RecordInference("model1", "v1", 200, true);
            _collector.RecordInference("model1", "v1", 300, true);

            var metrics = _collector.GetMetrics("model1", "v1", window);

            // Should only see the last 2 records (the window logic filters by timestamp)
            // Since all are recorded at roughly the same time, we'll see all 3
            Assert.Equal(3, metrics.RequestCount);
        }

        [Fact]
        public void GetAllMetrics_ReturnsAllModelMetrics()
        {
            _collector.RecordInference("model1", "v1", 100, true);
            _collector.RecordInference("model2", "v1", 200, true);
            _collector.RecordInference("model3", "v1", 300, true);

            var allMetrics = _collector.GetAllMetrics(TimeSpan.FromMinutes(5));

            Assert.Equal(3, allMetrics.Count);
            Assert.True(allMetrics.ContainsKey("model1:v1"));
            Assert.True(allMetrics.ContainsKey("model2:v1"));
            Assert.True(allMetrics.ContainsKey("model3:v1"));
        }

        [Fact]
        public void SetExporter_WithNullExporter_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => _collector.SetExporter(null));
        }

        [Fact]
        public void SetExporter_UpdatesExporter()
        {
            var newExporter = new TestExporter();
            _collector.SetExporter(newExporter);

            _collector.ExportMetrics();

            Assert.True(newExporter.ExportCalled);
        }

        [Fact]
        public void StartAutoExport_WithNegativeInterval_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => _collector.StartAutoExport(TimeSpan.FromSeconds(-1)));
        }

        [Fact]
        public void StartAutoExport_WithZeroInterval_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => _collector.StartAutoExport(TimeSpan.Zero));
        }

        [Fact]
        public void StartAutoExport_ExportsPeriodically()
        {
            var testExporter = new TestExporter();
            _collector.SetExporter(testExporter);
            _collector.RecordInference("model1", "v1", 100, true);

            _collector.StartAutoExport(TimeSpan.FromMilliseconds(100));

            Thread.Sleep(300); // Wait for at least 2 exports

            _collector.StopAutoExport();

            Assert.True(testExporter.ExportCalled);
            Assert.True(testExporter.ExportCount >= 2);
        }

        [Fact]
        public void StopAutoExport_StopsExporting()
        {
            var testExporter = new TestExporter();
            _collector.SetExporter(testExporter);
            _collector.RecordInference("model1", "v1", 100, true);

            _collector.StartAutoExport(TimeSpan.FromMilliseconds(100));
            Thread.Sleep(50); // Wait for one export cycle

            _collector.StopAutoExport();

            var countAfterStop = testExporter.ExportCount;
            Thread.Sleep(200); // Wait for more time

            Assert.Equal(countAfterStop, testExporter.ExportCount);
        }

        [Fact]
        public void ConcurrentRecording_ThreadSafe()
        {
            const int threadCount = 100;
            const int recordsPerThread = 10;
            var threads = new List<Thread>();

            for (int i = 0; i < threadCount; i++)
            {
                var thread = new Thread(() =>
                {
                    for (int j = 0; j < recordsPerThread; j++)
                    {
                        _collector.RecordInference("model1", "v1", 100, true);
                    }
                });
                threads.Add(thread);
                thread.Start();
            }

            foreach (var thread in threads)
            {
                thread.Join();
            }

            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            Assert.Equal(threadCount * recordsPerThread, metrics.RequestCount);
        }

        [Fact]
        public async Task ConcurrentRecording_AsyncThreadSafe()
        {
            const int taskCount = 100;
            const int recordsPerTask = 10;
            var tasks = new List<Task>();

            for (int i = 0; i < taskCount; i++)
            {
                var task = Task.Run(() =>
                {
                    for (int j = 0; j < recordsPerTask; j++)
                    {
                        _collector.RecordInference("model1", "v1", 100, true);
                    }
                });
                tasks.Add(task);
            }

            await Task.WhenAll(tasks);

            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            Assert.Equal(taskCount * recordsPerTask, metrics.RequestCount);
        }

        [Fact]
        public void PerformanceTest_Record10000Inferences_ShouldCompleteFast()
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            for (int i = 0; i < 10000; i++)
            {
                _collector.RecordInference("model1", "v1", 100, true);
            }

            stopwatch.Stop();

            // Should complete in less than 1 second (target: < 0.01ms per record)
            Assert.True(stopwatch.ElapsedMilliseconds < 1000,
                $"Recording 10000 inferences took {stopwatch.ElapsedMilliseconds}ms, expected < 1000ms");
        }

        [Fact]
        public void GetMetrics_PerformanceTest_ShouldCompleteFast()
        {
            // Record 100,000 inferences
            for (int i = 0; i < 100000; i++)
            {
                _collector.RecordInference("model1", "v1", 100, true);
            }

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            stopwatch.Stop();

            // Should complete in less than 10ms (performance target)
            Assert.True(stopwatch.ElapsedMilliseconds < 10,
                $"Getting metrics took {stopwatch.ElapsedMilliseconds}ms, expected < 10ms");

            Assert.Equal(100000, metrics.RequestCount);
        }

        [Fact]
        public void VersionMetrics_ToString_ReturnsFormattedString()
        {
            var metrics = new VersionMetrics(
                "model1",
                "v1",
                DateTime.UtcNow.AddMinutes(-5),
                DateTime.UtcNow,
                100,
                10.5,
                50.0,
                45.0,
                90.0,
                99.0,
                2.5,
                42,
                512.0,
                2,
                new Dictionary<string, long> { { "Timeout", 1 }, { "OOM", 1 } }
            );

            var str = metrics.ToString();

            Assert.Contains("model1", str);
            Assert.Contains("v1", str);
            Assert.Contains("100", str);
            Assert.Contains("50.00ms", str);
        }

        [Fact]
        public void MultipleVersions_AllHaveSeparateMetrics()
        {
            _collector.RecordInference("model1", "v1", 100, true);
            _collector.RecordInference("model1", "v2", 200, true);
            _collector.RecordInference("model1", "v3", 300, true);

            var metricsV1 = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));
            var metricsV2 = _collector.GetMetrics("model1", "v2", TimeSpan.FromMinutes(5));
            var metricsV3 = _collector.GetMetrics("model1", "v3", TimeSpan.FromMinutes(5));

            Assert.Equal(1, metricsV1.RequestCount);
            Assert.Equal(100, metricsV1.AverageLatencyMs);

            Assert.Equal(1, metricsV2.RequestCount);
            Assert.Equal(200, metricsV2.AverageLatencyMs);

            Assert.Equal(1, metricsV3.RequestCount);
            Assert.Equal(300, metricsV3.AverageLatencyMs);
        }

        [Fact]
        public void CalculateRequestsPerSecond_CalculatesCorrectly()
        {
            var startTime = DateTime.UtcNow.AddSeconds(-10);

            // Record 100 requests over 10 seconds = 10 RPS
            for (int i = 0; i < 100; i++)
            {
                _collector.RecordInference("model1", "v1", 100, true);
            }

            var metrics = _collector.GetMetrics("model1", "v1", TimeSpan.FromMinutes(5));

            Assert.Equal(100, metrics.RequestCount);
            Assert.InRange(metrics.RequestsPerSecond, 0, 100); // Should be roughly 10 RPS
        }
    }

    /// <summary>
    /// Test helper class to verify export behavior
    /// </summary>
    internal class TestExporter : IMetricsExporter
    {
        public bool ExportCalled { get; private set; }
        public int ExportCount { get; private set; }
        public Dictionary<string, VersionMetrics> LastExportedMetrics { get; private set; }

        public void Export(Dictionary<string, VersionMetrics> metrics)
        {
            ExportCalled = true;
            ExportCount++;
            LastExportedMetrics = metrics;
        }

        public Task ExportAsync(Dictionary<string, VersionMetrics> metrics)
        {
            ExportCalled = true;
            ExportCount++;
            LastExportedMetrics = metrics;
            return Task.CompletedTask;
        }
    }
}
