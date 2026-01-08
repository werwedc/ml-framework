using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using MLFramework.Serving;

namespace MLFramework.Tests.Serving
{
    /// <summary>
    /// Unit tests for ReferenceTracker functionality.
    /// </summary>
    public class ReferenceTrackerTests
    {
        private readonly ReferenceTracker _tracker;

        public ReferenceTrackerTests()
        {
            _tracker = new ReferenceTracker();
        }

        [Fact]
        public void AcquireReference_IncreasesCount()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";

            // Act
            _tracker.AcquireReference(modelName, version, requestId);

            // Assert
            Assert.Equal(1, _tracker.GetReferenceCount(modelName, version));
        }

        [Fact]
        public void ReleaseReference_DecreasesCount()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";
            _tracker.AcquireReference(modelName, version, requestId);

            // Act
            _tracker.ReleaseReference(modelName, version, requestId);

            // Assert
            Assert.Equal(0, _tracker.GetReferenceCount(modelName, version));
        }

        [Fact]
        public void MultipleConcurrentRequests_AccurateCount()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const int requestCount = 10;

            // Act
            Parallel.For(0, requestCount, i =>
            {
                _tracker.AcquireReference(modelName, version, $"req-{i:D3}");
            });

            // Assert
            Assert.Equal(requestCount, _tracker.GetReferenceCount(modelName, version));
        }

        [Fact]
        public void HasReferences_ReturnsCorrectValue()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";

            // Act & Assert
            Assert.False(_tracker.HasReferences(modelName, version));

            _tracker.AcquireReference(modelName, version, requestId);
            Assert.True(_tracker.HasReferences(modelName, version));

            _tracker.ReleaseReference(modelName, version, requestId);
            Assert.False(_tracker.HasReferences(modelName, version));
        }

        [Fact]
        public async Task WaitForZeroReferences_ReturnsImmediately_WhenCountIsZero()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";

            // Act
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            await _tracker.WaitForZeroReferencesAsync(modelName, version, TimeSpan.FromSeconds(1));
            stopwatch.Stop();

            // Assert
            Assert.True(stopwatch.ElapsedMilliseconds < 100, "Should return immediately");
        }

        [Fact]
        public async Task WaitForZeroReferences_BlocksUntilZero()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";

            _tracker.AcquireReference(modelName, version, requestId);

            var waitForZeroTask = _tracker.WaitForZeroReferencesAsync(modelName, version, TimeSpan.FromSeconds(5));

            // Give the task a moment to start waiting
            await Task.Delay(50);

            Assert.False(waitForZeroTask.IsCompleted);

            // Act
            _tracker.ReleaseReference(modelName, version, requestId);

            // Assert
            await waitForZeroTask;
        }

        [Fact]
        public async Task WaitForZeroReferences_TimesOut_WhenReferencesRemain()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";

            _tracker.AcquireReference(modelName, version, requestId);

            // Act & Assert
            await Assert.ThrowsAsync<TimeoutException>(async () =>
            {
                await _tracker.WaitForZeroReferencesAsync(modelName, version, TimeSpan.FromMilliseconds(100));
            });
        }

        [Fact]
        public void AcquireWithNullParameters_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _tracker.AcquireReference(null, "1.0.0", "req-001"));

            Assert.Throws<ArgumentException>(() =>
                _tracker.AcquireReference("model", null, "req-001"));

            Assert.Throws<ArgumentException>(() =>
                _tracker.AcquireReference("model", "1.0.0", null));

            Assert.Throws<ArgumentException>(() =>
                _tracker.AcquireReference("", "1.0.0", "req-001"));

            Assert.Throws<ArgumentException>(() =>
                _tracker.AcquireReference("model", "", "req-001"));

            Assert.Throws<ArgumentException>(() =>
                _tracker.AcquireReference("model", "1.0.0", ""));
        }

        [Fact]
        public void ReleaseNeverAcquiredReference_ThrowsInvalidOperationException()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                _tracker.ReleaseReference(modelName, version, requestId));
        }

        [Fact]
        public void ReleaseTooManyTimes_ThrowsInvalidOperationException()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";

            _tracker.AcquireReference(modelName, version, requestId);
            _tracker.ReleaseReference(modelName, version, requestId);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                _tracker.ReleaseReference(modelName, version, requestId));
        }

        [Fact]
        public void RequestTracker_AutoReleasesOnDispose()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";

            // Act
            using (var requestTracker = new RequestTracker(_tracker, modelName, version, requestId))
            {
                Assert.Equal(1, _tracker.GetReferenceCount(modelName, version));
            }

            // Assert
            Assert.Equal(0, _tracker.GetReferenceCount(modelName, version));
        }

        [Fact]
        public void RequestTracker_ManualRelease()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";

            // Act
            var requestTracker = new RequestTracker(_tracker, modelName, version, requestId);
            Assert.Equal(1, _tracker.GetReferenceCount(modelName, version));

            requestTracker.Release();
            Assert.Equal(0, _tracker.GetReferenceCount(modelName, version));

            // Second release should not throw or change count
            requestTracker.Release();
            Assert.Equal(0, _tracker.GetReferenceCount(modelName, version));
        }

        [Fact]
        public void RequestTracker_NullTracker_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new RequestTracker(null, "model", "1.0.0", "req-001"));
        }

        [Fact]
        public void RequestTracker_NullParameters_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new RequestTracker(_tracker, null, "1.0.0", "req-001"));

            Assert.Throws<ArgumentNullException>(() =>
                new RequestTracker(_tracker, "model", null, "req-001"));

            Assert.Throws<ArgumentNullException>(() =>
                new RequestTracker(_tracker, "model", "1.0.0", null));
        }

        [Fact]
        public void MultipleModels_SeparateTracking()
        {
            // Arrange
            const string model1 = "model1";
            const string model2 = "model2";
            const string version = "1.0.0";

            // Act
            _tracker.AcquireReference(model1, version, "req-001");
            _tracker.AcquireReference(model2, version, "req-002");

            // Assert
            Assert.Equal(1, _tracker.GetReferenceCount(model1, version));
            Assert.Equal(1, _tracker.GetReferenceCount(model2, version));

            _tracker.ReleaseReference(model1, version, "req-001");

            Assert.Equal(0, _tracker.GetReferenceCount(model1, version));
            Assert.Equal(1, _tracker.GetReferenceCount(model2, version));
        }

        [Fact]
        public void GetAllReferenceCounts_ReturnsAllModels()
        {
            // Arrange
            _tracker.AcquireReference("model1", "1.0.0", "req-001");
            _tracker.AcquireReference("model2", "1.0.0", "req-002");
            _tracker.AcquireReference("model2", "1.0.0", "req-003");

            // Act
            var allCounts = _tracker.GetAllReferenceCounts();

            // Assert
            Assert.Equal(2, allCounts.Count);
            Assert.Equal(1, allCounts["model1:1.0.0"]);
            Assert.Equal(2, allCounts["model2:1.0.0"]);
        }

        [Fact]
        public async Task HighConcurrency_1000Threads_UpdatesCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const int threadCount = 1000;

            // Act - Acquire 1000 references concurrently
            var acquireTasks = Enumerable.Range(0, threadCount).Select(i =>
                Task.Run(() => _tracker.AcquireReference(modelName, version, $"req-{i:D4}"))
            );

            await Task.WhenAll(acquireTasks);

            Assert.Equal(threadCount, _tracker.GetReferenceCount(modelName, version));

            // Release 1000 references concurrently
            var releaseTasks = Enumerable.Range(0, threadCount).Select(i =>
                Task.Run(() => _tracker.ReleaseReference(modelName, version, $"req-{i:D4}"))
            );

            await Task.WhenAll(releaseTasks);

            // Assert
            Assert.Equal(0, _tracker.GetReferenceCount(modelName, version));
        }

        [Fact]
        public async Task MultipleWaiters_AllSignaledWhenZero()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const int requestCount = 3;

            // Create multiple references
            for (int i = 0; i < requestCount; i++)
            {
                _tracker.AcquireReference(modelName, version, $"req-{i}");
            }

            // Create multiple waiting tasks
            var waitTasks = Enumerable.Range(0, 5).Select(i =>
                _tracker.WaitForZeroReferencesAsync(modelName, version, TimeSpan.FromSeconds(5))
            ).ToArray();

            // Give tasks time to start waiting
            await Task.Delay(50);

            // Release all references
            for (int i = 0; i < requestCount; i++)
            {
                _tracker.ReleaseReference(modelName, version, $"req-{i}");
            }

            // Act & Assert - All waiters should complete
            await Task.WhenAll(waitTasks);
        }

        [Fact]
        public async Task WaitForZeroReferences_RespectsCancellationToken()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";

            _tracker.AcquireReference(modelName, version, requestId);

            var cts = new CancellationTokenSource();

            var waitTask = _tracker.WaitForZeroReferencesAsync(modelName, version, TimeSpan.FromSeconds(10), cts.Token);

            // Act
            cts.Cancel();

            // Assert
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waitTask);
        }

        [Fact]
        public void ClearAll_RemovesAllReferences()
        {
            // Arrange
            _tracker.AcquireReference("model1", "1.0.0", "req-001");
            _tracker.AcquireReference("model2", "1.0.0", "req-002");

            // Act
            _tracker.ClearAll();

            // Assert
            Assert.Equal(0, _tracker.GetReferenceCount("model1", "1.0.0"));
            Assert.Equal(0, _tracker.GetReferenceCount("model2", "1.0.0"));
            Assert.Empty(_tracker.GetAllReferenceCounts());
        }

        [Fact]
        public void ReferenceLeakDetectionEnabled_TracksRequestIds()
        {
            // Arrange
            var trackerWithLeakDetection = new ReferenceTracker(enableReferenceLeakDetection: true);
            const string modelName = "test-model";
            const string version = "1.0.0";
            const string requestId = "req-001";

            // Act
            trackerWithLeakDetection.AcquireReference(modelName, version, requestId);

            // Assert - Count should be incremented
            Assert.Equal(1, trackerWithLeakDetection.GetReferenceCount(modelName, version));

            // Cleanup
            trackerWithLeakDetection.ReleaseReference(modelName, version, requestId);
        }

        [Fact]
        public void Performance_AcquireAndRelease_UnderPerformanceTarget()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            const int iterations = 10000;
            const double targetTimePerOperation = 0.01; // 0.01ms = 10 microseconds

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            for (int i = 0; i < iterations; i++)
            {
                _tracker.AcquireReference(modelName, version, $"req-{i:D5}");
            }

            var acquireTime = stopwatch.Elapsed.TotalMilliseconds;

            stopwatch.Restart();

            for (int i = 0; i < iterations; i++)
            {
                _tracker.ReleaseReference(modelName, version, $"req-{i:D5}");
            }

            var releaseTime = stopwatch.Elapsed.TotalMilliseconds;

            // Assert
            var avgAcquireTime = acquireTime / iterations;
            var avgReleaseTime = releaseTime / iterations;

            Assert.True(avgAcquireTime < targetTimePerOperation,
                $"Average acquire time {avgAcquireTime}ms exceeds target {targetTimePerOperation}ms");

            Assert.True(avgReleaseTime < targetTimePerOperation,
                $"Average release time {avgReleaseTime}ms exceeds target {targetTimePerOperation}ms");
        }
    }
}
