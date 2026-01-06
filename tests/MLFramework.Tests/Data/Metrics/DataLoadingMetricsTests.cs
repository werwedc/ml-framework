using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Data.Metrics;
using Xunit;

namespace MLFramework.Tests.Data.Metrics
{
    public class DataLoadingMetricsTests
    {
        [Fact]
        public void Constructor_InitializesCorrectly()
        {
            var metrics = new DataLoadingMetrics();

            Assert.True(metrics.Enabled);
            Assert.Empty(metrics.GetMetricsSummary());
        }

        [Fact]
        public void Constructor_WithDisabledParameter_CreatesDisabledMetrics()
        {
            var metrics = new DataLoadingMetrics(enabled: false);

            Assert.False(metrics.Enabled);
        }

        [Fact]
        public void RecordTiming_RecordsCorrectly()
        {
            var metrics = new DataLoadingMetrics();

            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(100));

            var summary = metrics.GetMetricsSummary();
            Assert.True(summary.ContainsKey("TestMetric"));
            Assert.Equal(1, summary["TestMetric"].Count);
            Assert.Equal(100, summary["TestMetric"].Average);
        }

        [Fact]
        public void RecordTiming_WhenDisabled_DoesNotRecord()
        {
            var metrics = new DataLoadingMetrics(enabled: false);

            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(100));

            var summary = metrics.GetMetricsSummary();
            Assert.Empty(summary);
        }

        [Fact]
        public void RecordTiming_WithWorkerId_RecordsCorrectly()
        {
            var metrics = new DataLoadingMetrics();

            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(100), workerId: 2);

            var summary = metrics.GetMetricsSummary();
            Assert.True(summary.ContainsKey("TestMetric"));
            Assert.Equal(1, summary["TestMetric"].Count);
        }

        [Fact]
        public void RecordCounter_RecordsCorrectly()
        {
            var metrics = new DataLoadingMetrics();

            metrics.RecordCounter("CounterMetric", 42);

            var summary = metrics.GetMetricsSummary();
            Assert.True(summary.ContainsKey("CounterMetric"));
            Assert.Equal(1, summary["CounterMetric"].Count);
            Assert.Equal(42, summary["CounterMetric"].Average);
        }

        [Fact]
        public void RecordCounter_WhenDisabled_DoesNotRecord()
        {
            var metrics = new DataLoadingMetrics(enabled: false);

            metrics.RecordCounter("CounterMetric", 42);

            var summary = metrics.GetMetricsSummary();
            Assert.Empty(summary);
        }

        [Fact]
        public void RecordMultipleTimings_CalculatesCorrectStatistics()
        {
            var metrics = new DataLoadingMetrics();

            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(100));
            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(200));
            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(300));

            var summary = metrics.GetMetricsSummary();
            Assert.Equal(3, summary["TestMetric"].Count);
            Assert.Equal(200, summary["TestMetric"].Average);
            Assert.Equal(100, summary["TestMetric"].Min);
            Assert.Equal(300, summary["TestMetric"].Max);
            Assert.Equal(600, summary["TestMetric"].Total);
        }

        [Fact]
        public void GetMetricsSummary_ReturnsAllMetrics()
        {
            var metrics = new DataLoadingMetrics();

            metrics.RecordTiming("Metric1", TimeSpan.FromMilliseconds(100));
            metrics.RecordTiming("Metric2", TimeSpan.FromMilliseconds(200));
            metrics.RecordTiming("Metric3", TimeSpan.FromMilliseconds(300));

            var summary = metrics.GetMetricsSummary();
            Assert.Equal(3, summary.Count);
            Assert.True(summary.ContainsKey("Metric1"));
            Assert.True(summary.ContainsKey("Metric2"));
            Assert.True(summary.ContainsKey("Metric3"));
        }

        [Fact]
        public void StartEpoch_StartsTimer()
        {
            var metrics = new DataLoadingMetrics();

            metrics.StartEpoch();
            metrics.EndEpoch();

            var summary = metrics.GetMetricsSummary();
            Assert.True(summary.ContainsKey("EpochTime"));
            Assert.Equal(1, summary["EpochTime"].Count);
        }

        [Fact]
        public void StartEndEpoch_MultipleTimes_RecordsEachEpoch()
        {
            var metrics = new DataLoadingMetrics();

            metrics.StartEpoch();
            Thread.Sleep(10);
            metrics.EndEpoch();

            metrics.StartEpoch();
            Thread.Sleep(20);
            metrics.EndEpoch();

            var summary = metrics.GetMetricsSummary();
            Assert.Equal(2, summary["EpochTime"].Count);
            Assert.True(summary["EpochTime"].Total > 30);
        }

        [Fact]
        public void EndEpoch_WithoutStart_DoesNotThrow()
        {
            var metrics = new DataLoadingMetrics();

            var exception = Record.Exception(() => metrics.EndEpoch());

            Assert.Null(exception);
        }

        [Fact]
        public void Reset_ClearsAllMetrics()
        {
            var metrics = new DataLoadingMetrics();

            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(100));
            metrics.RecordCounter("CounterMetric", 42);

            Assert.Equal(2, metrics.GetMetricsSummary().Count);

            metrics.Reset();

            Assert.Empty(metrics.GetMetricsSummary());
        }

        [Fact]
        public void SetEnabled_False_DisablesRecording()
        {
            var metrics = new DataLoadingMetrics();

            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(100));
            Assert.Equal(1, metrics.GetMetricsSummary().Count);

            metrics.SetEnabled(false);

            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(100));
            Assert.Equal(1, metrics.GetMetricsSummary().Count);
        }

        [Fact]
        public void SetEnabled_True_EnablesRecording()
        {
            var metrics = new DataLoadingMetrics(enabled: false);

            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(100));
            Assert.Empty(metrics.GetMetricsSummary());

            metrics.SetEnabled(true);

            metrics.RecordTiming("TestMetric", TimeSpan.FromMilliseconds(100));
            Assert.Equal(1, metrics.GetMetricsSummary().Count);
        }

        [Fact]
        public void ConcurrentRecording_ThreadSafe()
        {
            var metrics = new DataLoadingMetrics();
            var threads = new List<Thread>();
            const int threadCount = 10;
            const int recordsPerThread = 100;

            for (int i = 0; i < threadCount; i++)
            {
                var threadId = i;
                var thread = new Thread(() =>
                {
                    for (int j = 0; j < recordsPerThread; j++)
                    {
                        metrics.RecordTiming($"Thread{threadId}", TimeSpan.FromMilliseconds(10 + j));
                    }
                });
                threads.Add(thread);
                thread.Start();
            }

            foreach (var thread in threads)
            {
                thread.Join();
            }

            var summary = metrics.GetMetricsSummary();

            // Verify all threads recorded their metrics
            for (int i = 0; i < threadCount; i++)
            {
                Assert.True(summary.ContainsKey($"Thread{i}"));
                Assert.Equal(recordsPerThread, summary[$"Thread{i}"].Count);
            }
        }

        [Fact]
        public async Task AsyncRecording_ThreadSafe()
        {
            var metrics = new DataLoadingMetrics();
            const int taskCount = 10;
            const int recordsPerTask = 100;

            var tasks = new List<Task>();
            for (int i = 0; i < taskCount; i++)
            {
                var taskId = i;
                var task = Task.Run(() =>
                {
                    for (int j = 0; j < recordsPerTask; j++)
                    {
                        metrics.RecordTiming($"Task{taskId}", TimeSpan.FromMilliseconds(10 + j));
                    }
                });
                tasks.Add(task);
            }

            await Task.WhenAll(tasks);

            var summary = metrics.GetMetricsSummary();

            // Verify all tasks recorded their metrics
            for (int i = 0; i < taskCount; i++)
            {
                Assert.True(summary.ContainsKey($"Task{i}"));
                Assert.Equal(recordsPerTask, summary[$"Task{i}"].Count);
            }
        }

        [Fact]
        public void RecordCounter_WithLargeValues_HandlesCorrectly()
        {
            var metrics = new DataLoadingMetrics();

            metrics.RecordCounter("LargeValue", double.MaxValue);
            metrics.RecordCounter("LargeValue", double.MinValue);

            var summary = metrics.GetMetricsSummary();
            Assert.Equal(2, summary["LargeValue"].Count);
        }

        [Fact]
        public void RecordTiming_WithZeroDuration_HandlesCorrectly()
        {
            var metrics = new DataLoadingMetrics();

            metrics.RecordTiming("Zero", TimeSpan.Zero);

            var summary = metrics.GetMetricsSummary();
            Assert.Equal(1, summary["Zero"].Count);
            Assert.Equal(0, summary["Zero"].Average);
        }
    }
}
