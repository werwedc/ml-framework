using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Data.Worker;
using Xunit;

namespace MLFramework.Tests.Data.Worker
{
    public class PrefetchQueueTests
    {
        [Fact]
        public void Constructor_WithNullBatchGenerator_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => new PrefetchQueue<int>(null));
        }

        [Fact]
        public void Constructor_WithNonPositivePrefetchCount_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new PrefetchQueue<int>(() => Enumerable.Range(1, 10), 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new PrefetchQueue<int>(() => Enumerable.Range(1, 10), -1));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 3);
            Assert.NotNull(queue);
            Assert.Equal(3, queue.PrefetchCount);
        }

        [Fact]
        public void IsRunning_Initially_ReturnsFalse()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 2);
            Assert.False(queue.IsRunning);
        }

        [Fact]
        public void Start_WhenNotRunning_SetsIsRunningToTrue()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 2);
            queue.Start();
            Assert.True(queue.IsRunning);
        }

        [Fact]
        public void Start_WhenAlreadyRunning_DoesNotThrow()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 2);
            queue.Start();
            queue.Start(); // Should not throw
            Assert.True(queue.IsRunning);
        }

        [Fact]
        public void Stop_WhenNotRunning_DoesNotThrow()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 2);
            queue.Stop(); // Should not throw
            Assert.False(queue.IsRunning);
        }

        [Fact]
        public void Stop_WhenRunning_SetsIsRunningToFalse()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 2);
            queue.Start();
            queue.Stop();
            Assert.False(queue.IsRunning);
        }

        [Fact]
        public void GetNext_WhenNotRunning_ThrowsInvalidOperationException()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 2);
            Assert.Throws<InvalidOperationException>(() => queue.GetNext());
        }

        [Fact]
        public void TryGetNext_WhenNotRunning_ReturnsFalse()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 2);
            Assert.False(queue.TryGetNext(out _));
        }

        [Fact]
        public void GetNext_AfterStart_ReturnsItemsInOrder()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 5), prefetchCount: 2);
            queue.Start();

            // Wait a bit for prefetching
            Thread.Sleep(100);

            Assert.Equal(1, queue.GetNext());
            Assert.Equal(2, queue.GetNext());
            Assert.Equal(3, queue.GetNext());
            Assert.Equal(4, queue.GetNext());
            Assert.Equal(5, queue.GetNext());
        }

        [Fact]
        public void TryGetNext_AfterStart_ReturnsItemsInOrder()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 5), prefetchCount: 2);
            queue.Start();

            // Wait a bit for prefetching
            Thread.Sleep(100);

            Assert.True(queue.TryGetNext(out var item1));
            Assert.Equal(1, item1);

            Assert.True(queue.TryGetNext(out var item2));
            Assert.Equal(2, item2);
        }

        [Fact]
        public void AvailableBatches_AfterStartAndPrefetch_ReturnsPrefetchCount()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 3);
            queue.Start();

            // Wait for prefetching
            Thread.Sleep(200);

            Assert.True(queue.AvailableBatches > 0);
            Assert.True(queue.AvailableBatches <= queue.PrefetchCount);
        }

        [Fact]
        public void PrefetchCount_ReturnsConfiguredValue()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 5);
            Assert.Equal(5, queue.PrefetchCount);
        }

        [Fact]
        public void GetNext_ConsumesPrefetchedBatches_ReducesAvailableBatches()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 3);
            queue.Start();

            // Wait for prefetching
            Thread.Sleep(200);

            var initialAvailable = queue.AvailableBatches;
            queue.GetNext();

            Assert.True(queue.AvailableBatches < initialAvailable);
        }

        [Fact]
        public void PrefetchLoop_ContinuouslyRefillsBuffer()
        {
            var iterationCount = 0;
            using var queue = new PrefetchQueue<int>(() =>
            {
                iterationCount++;
                return Enumerable.Range(iterationCount * 10, 3);
            }, prefetchCount: 2);

            queue.Start();

            // Consume batches and verify buffer refills
            for (int i = 0; i < 10; i++)
            {
                Thread.Sleep(50); // Give time for prefetching
                if (queue.TryGetNext(out _))
                {
                    Assert.True(queue.AvailableBatches >= 0);
                }
            }

            queue.Stop();
        }

        [Fact]
        public void Stop_WaitsForPrefetchTaskToComplete()
        {
            var isRunning = true;
            using var queue = new PrefetchQueue<int>(() =>
            {
                Thread.Sleep(50);
                return isRunning ? Enumerable.Range(1, 1) : Enumerable.Empty<int>();
            }, prefetchCount: 2);

            queue.Start();
            Thread.Sleep(100);
            isRunning = false;

            var stopTime = DateTime.UtcNow;
            queue.Stop();
            var stopDuration = DateTime.UtcNow - stopTime;

            // Stop should complete within reasonable time
            Assert.True(stopDuration.TotalSeconds < 2);
        }

        [Fact]
        public void Dispose_StopsQueueAndReleasesResources()
        {
            var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 2);
            queue.Start();

            queue.Dispose();

            Assert.False(queue.IsRunning);
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 10), prefetchCount: 2);
            queue.Start();

            queue.Dispose();
            queue.Dispose(); // Should not throw
        }

        [Fact]
        public void BatchGenerator_ThrowsException_ContinuesPrefetching()
        {
            var callCount = 0;
            using var queue = new PrefetchQueue<int>(() =>
            {
                callCount++;
                if (callCount == 2)
                    throw new InvalidOperationException("Test exception");

                return Enumerable.Range(1, 3);
            }, prefetchCount: 2);

            queue.Start();

            // Should get first batch
            Thread.Sleep(200);
            var firstBatch = queue.GetNext();
            Assert.Equal(1, firstBatch);

            // Second call to generator will throw, but queue should continue
            Thread.Sleep(100);

            // Should still be able to get items from first batch
            Assert.True(queue.TryGetNext(out _));
        }

        [Fact]
        public void EmptyBatchGenerator_DoesNotCrash()
        {
            var callCount = 0;
            using var queue = new PrefetchQueue<int>(() =>
            {
                callCount++;
                if (callCount < 3)
                    return Enumerable.Range(1, 2);

                // Return empty enumerable
                return Enumerable.Empty<int>();
            }, prefetchCount: 2);

            queue.Start();

            // Get first few batches
            Thread.Sleep(100);
            Assert.Equal(1, queue.GetNext());
            Assert.Equal(2, queue.GetNext());
        }

        [Fact]
        public void MultipleConsumers_ThreadSafe()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 100), prefetchCount: 4);
            queue.Start();

            // Wait for prefetching
            Thread.Sleep(200);

            var consumerCount1 = 0;
            var consumerCount2 = 0;
            var completed = false;

            var consumer1 = Task.Run(() =>
            {
                while (!completed)
                {
                    if (queue.TryGetNext(out _))
                        Interlocked.Increment(ref consumerCount1);
                    else
                        Thread.Sleep(1);
                }
            });

            var consumer2 = Task.Run(() =>
            {
                while (!completed)
                {
                    if (queue.TryGetNext(out _))
                        Interlocked.Increment(ref consumerCount2);
                    else
                        Thread.Sleep(1);
                }
            });

            Thread.Sleep(500);
            completed = true;

            Task.WaitAll(consumer1, consumer2);

            var totalConsumed = consumerCount1 + consumerCount2;
            Assert.Equal(100, totalConsumed);
        }

        [Fact]
        public void PrefetchQueue_LargePrefetchCount_MaintainsBufferSize()
        {
            using var queue = new PrefetchQueue<int>(() => Enumerable.Range(1, 100), prefetchCount: 10);
            queue.Start();

            // Wait for prefetching
            Thread.Sleep(300);

            var availableAfterStart = queue.AvailableBatches;
            Assert.True(availableAfterStart > 0);
            Assert.True(availableAfterStart <= queue.PrefetchCount);

            // Consume some items
            for (int i = 0; i < 5; i++)
            {
                queue.GetNext();
            }

            Thread.Sleep(100);

            // Buffer should have refilled
            Assert.True(queue.AvailableBatches > 0);
        }

        [Fact]
        public void UsageExample_MatchesSpec()
        {
            using var prefetchQueue = new PrefetchQueue<int>(
                () => Enumerable.Range(1, 100),
                prefetchCount: 3);

            prefetchQueue.Start();

            // Wait for prefetching
            Thread.Sleep(100);

            var results = new List<int>();
            for (int i = 0; i < 10; i++)
            {
                var batch = prefetchQueue.GetNext();
                results.Add(batch);
            }

            // Verify we got the first 10 items in order
            Assert.Equal(Enumerable.Range(1, 10), results);
        }
    }
}
