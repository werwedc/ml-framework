using System;
using System.Threading.Tasks;
using MLFramework.Data.Metrics;
using Xunit;

namespace MLFramework.Tests.Data.Metrics
{
    public class ProfilingScopeTests
    {
        [Fact]
        public void ProfilingScope_DisposeRecordsTiming()
        {
            var metrics = new DataLoadingMetrics();

            using (metrics.Profile("TestMetric"))
            {
                Thread.Sleep(50);
            }

            var summary = metrics.GetMetricsSummary();
            Assert.True(summary.ContainsKey("TestMetric"));
            Assert.Equal(1, summary["TestMetric"].Count);
            Assert.True(summary["TestMetric"].Total >= 45); // Allow some tolerance
        }

        [Fact]
        public void ProfilingScope_WithWorkerId_RecordsCorrectly()
        {
            var metrics = new DataLoadingMetrics();

            using (metrics.Profile("TestMetric", workerId: 5))
            {
                Thread.Sleep(10);
            }

            var summary = metrics.GetMetricsSummary();
            Assert.True(summary.ContainsKey("TestMetric"));
            Assert.Equal(1, summary["TestMetric"].Count);
        }

        [Fact]
        public void ProfilingScope_WhenDisabled_DoesNotRecord()
        {
            var metrics = new DataLoadingMetrics(enabled: false);

            using (metrics.Profile("TestMetric"))
            {
                Thread.Sleep(50);
            }

            var summary = metrics.GetMetricsSummary();
            Assert.Empty(summary);
        }

        [Fact]
        public void ProfilingScope_NestedScopes_RecordsAll()
        {
            var metrics = new DataLoadingMetrics();

            using (metrics.Profile("Outer"))
            {
                Thread.Sleep(20);
                using (metrics.Profile("Inner"))
                {
                    Thread.Sleep(10);
                }
                Thread.Sleep(20);
            }

            var summary = metrics.GetMetricsSummary();
            Assert.True(summary.ContainsKey("Outer"));
            Assert.True(summary.ContainsKey("Inner"));

            // Inner should be ~10ms, Outer should be ~50ms
            Assert.True(summary["Inner"].Total >= 5 && summary["Inner"].Total <= 20);
            Assert.True(summary["Outer"].Total >= 40 && summary["Outer"].Total <= 80);
        }

        [Fact]
        public void ProfilingScope_MultipleSequentialScopes_RecordsAll()
        {
            var metrics = new DataLoadingMetrics();

            using (metrics.Profile("Scope1"))
            {
                Thread.Sleep(10);
            }

            using (metrics.Profile("Scope2"))
            {
                Thread.Sleep(20);
            }

            using (metrics.Profile("Scope3"))
            {
                Thread.Sleep(30);
            }

            var summary = metrics.GetMetricsSummary();
            Assert.Equal(3, summary.Count);
            Assert.Equal(1, summary["Scope1"].Count);
            Assert.Equal(1, summary["Scope2"].Count);
            Assert.Equal(1, summary["Scope3"].Count);
        }

        [Fact]
        public void ProfilingScope_WithMultipleCallsToSameMetric_Aggregates()
        {
            var metrics = new DataLoadingMetrics();

            for (int i = 0; i < 5; i++)
            {
                using (metrics.Profile("RepeatedMetric"))
                {
                    Thread.Sleep(10);
                }
            }

            var summary = metrics.GetMetricsSummary();
            Assert.Equal(5, summary["RepeatedMetric"].Count);
            Assert.True(summary["RepeatedMetric"].Total >= 45); // Allow some tolerance
        }

        [Fact]
        public void ProfilingScope_ImmediateDispose_RecordsZeroTime()
        {
            var metrics = new DataLoadingMetrics();

            using (metrics.Profile("Immediate"))
            {
                // No delay
            }

            var summary = metrics.GetMetricsSummary();
            Assert.True(summary.ContainsKey("Immediate"));
            Assert.True(summary["Immediate"].Average < 10); // Should be very small
        }

        [Fact]
        public void ProfilingScope_DoubleDispose_DoesNotRecordTwice()
        {
            var metrics = new DataLoadingMetrics();
            ProfilingScope? scope = metrics.Profile("TestMetric") as ProfilingScope;

            Thread.Sleep(10);
            scope?.Dispose();

            var summary = metrics.GetMetricsSummary();
            Assert.Equal(1, summary["TestMetric"].Count);

            // Second dispose should not affect
            scope?.Dispose();
            Assert.Equal(1, summary["TestMetric"].Count);
        }

        [Fact]
        public void ProfilingScope_ConcurrentScopes_ThreadSafe()
        {
            var metrics = new DataLoadingMetrics();
            const int threadCount = 10;
            const int scopesPerThread = 10;

            var threads = new System.Collections.Generic.List<System.Threading.Thread>();

            for (int i = 0; i < threadCount; i++)
            {
                var threadId = i;
                var thread = new System.Threading.Thread(() =>
                {
                    for (int j = 0; j < scopesPerThread; j++)
                    {
                        using (metrics.Profile($"Thread{threadId}"))
                        {
                            System.Threading.Thread.Sleep(5);
                        }
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
                Assert.Equal(scopesPerThread, summary[$"Thread{i}"].Count);
            }
        }

        [Fact]
        public async Task ProfilingScope_AsyncScopes_ThreadSafe()
        {
            var metrics = new DataLoadingMetrics();
            const int taskCount = 10;
            const int scopesPerTask = 10;

            var tasks = new System.Collections.Generic.List<Task>();

            for (int i = 0; i < taskCount; i++)
            {
                var taskId = i;
                var task = Task.Run(async () =>
                {
                    for (int j = 0; j < scopesPerTask; j++)
                    {
                        using (metrics.Profile($"Task{taskId}"))
                        {
                            await Task.Delay(5);
                        }
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
                Assert.Equal(scopesPerTask, summary[$"Task{i}"].Count);
            }
        }

        [Fact]
        public void ProfilingScope_ProfileExtensionMethod_ReturnsIDisposable()
        {
            var metrics = new DataLoadingMetrics();

            IDisposable? scope = metrics.Profile("TestMetric");
            Assert.NotNull(scope);

            scope?.Dispose();

            var summary = metrics.GetMetricsSummary();
            Assert.Equal(1, summary["TestMetric"].Count);
        }

        [Fact]
        public void ProfilingScope_WithException_DoesRecord()
        {
            var metrics = new DataLoadingMetrics();

            try
            {
                using (metrics.Profile("ExceptionScope"))
                {
                    Thread.Sleep(10);
                    throw new InvalidOperationException("Test exception");
                }
            }
            catch (InvalidOperationException)
            {
                // Expected
            }

            var summary = metrics.GetMetricsSummary();
            Assert.True(summary.ContainsKey("ExceptionScope"));
            Assert.Equal(1, summary["ExceptionScope"].Count);
        }
    }
}
