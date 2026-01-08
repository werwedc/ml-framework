using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace MLFramework.Tests.Data
{
    /// <summary>
    /// Comprehensive unit tests for SharedQueue implementation.
    /// Tests cover constructor, properties, enqueue/dequeue operations,
    /// blocking behavior, cancellation, statistics, and concurrent access.
    /// </summary>
    public class SharedQueueTests
    {
        private readonly ITestOutputHelper _output;

        public SharedQueueTests(ITestOutputHelper output)
        {
            _output = output;
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_ValidCapacity_CreatesQueue()
        {
            // Act
            using var queue = new SharedQueue<int>(10);

            // Assert
            Assert.Equal(10, queue.Capacity);
            Assert.Equal(0, queue.Count);
            Assert.False(queue.IsCompleted);
        }

        [Fact]
        public void Constructor_CapacityOne_CreatesQueue()
        {
            // Act
            using var queue = new SharedQueue<int>(1);

            // Assert
            Assert.Equal(1, queue.Capacity);
        }

        [Fact]
        public void Constructor_LargeCapacity_CreatesQueue()
        {
            // Act
            using var queue = new SharedQueue<int>(10000);

            // Assert
            Assert.Equal(10000, queue.Capacity);
        }

        [Fact]
        public void Constructor_WithCancellationToken_UsesToken()
        {
            // Arrange
            var cts = new CancellationTokenSource();

            // Act
            using var queue = new SharedQueue<int>(10, cts.Token);

            // Assert - No exception thrown, token accepted
            Assert.Equal(10, queue.Capacity);
        }

        [Fact]
        public void Constructor_WithoutCancellationToken_NoToken()
        {
            // Act & Assert
            using var queue = new SharedQueue<int>(10);
            Assert.Equal(10, queue.Capacity);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        [InlineData(-100)]
        public void Constructor_ZeroOrNegativeCapacity_ThrowsArgumentOutOfRangeException(int capacity)
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new SharedQueue<int>(capacity));
        }

        #endregion

        #region Properties Tests

        [Fact]
        public void Count_EmptyQueue_ReturnsZero()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act & Assert
            Assert.Equal(0, queue.Count);
        }

        [Fact]
        public void Count_AfterEnqueue_Increments()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Assert
            Assert.Equal(2, queue.Count);
        }

        [Fact]
        public void Count_AfterDequeue_Decrements()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Act
            queue.Dequeue();

            // Assert
            Assert.Equal(1, queue.Count);
        }

        [Fact]
        public void Count_AfterCompleteAdding_Stable()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.CompleteAdding();

            // Act
            int count = queue.Count;

            // Assert
            Assert.Equal(2, count);
        }

        [Fact]
        public void IsCompleted_Initial_ReturnsFalse()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act & Assert
            Assert.False(queue.IsCompleted);
        }

        [Fact]
        public void IsCompleted_AfterCompleteAdding_ReturnsTrue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.CompleteAdding();

            // Act
            var isCompleted = queue.IsCompleted;

            // Assert
            Assert.True(isCompleted);
        }

        [Fact]
        public void IsCompleted_AfterAllDequeued_ReturnsTrue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.CompleteAdding();
            queue.Dequeue();

            // Act & Assert
            Assert.True(queue.IsCompleted);
        }

        [Fact]
        public void Capacity_ReturnsCorrectValue()
        {
            // Arrange & Act
            using var queue = new SharedQueue<int>(100);

            // Assert
            Assert.Equal(100, queue.Capacity);
        }

        #endregion

        #region Enqueue Tests

        [Fact]
        public void Enqueue_SingleItem_IncreasesCount()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            queue.Enqueue(42);

            // Assert
            Assert.Equal(1, queue.Count);
        }

        [Fact]
        public void Enqueue_MultipleItems_IncreasesCount()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            for (int i = 0; i < 5; i++)
            {
                queue.Enqueue(i);
            }

            // Assert
            Assert.Equal(5, queue.Count);
        }

        [Fact]
        public async Task Enqueue_FullQueue_BlocksUntilSpace()
        {
            // Arrange
            using var queue = new SharedQueue<int>(2);
            queue.Enqueue(1);
            queue.Enqueue(2);

            var enqueueTask = Task.Run(() =>
            {
                queue.Enqueue(3); // Should block
            });

            // Act
            var completedInTime = await Task.WhenAny(enqueueTask, Task.Delay(100));
            queue.Dequeue(); // Free up space
            await enqueueTask;

            // Assert
            Assert.Equal(TaskStatus.RanToCompletion, enqueueTask.Status);
            Assert.Equal(2, queue.Count);
        }

        [Fact]
        public void Enqueue_AfterCompleteAdding_ThrowsInvalidOperationException()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.CompleteAdding();

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => queue.Enqueue(1));
        }

        [Fact]
        public async Task ConcurrentEnqueue_MultipleThreads_ThreadSafe()
        {
            // Arrange
            using var queue = new SharedQueue<int>(1000);
            const int threadCount = 10;
            const int itemsPerThread = 100;

            // Act
            var tasks = Enumerable.Range(0, threadCount).Select(threadId =>
                Task.Run(() =>
                {
                    for (int i = 0; i < itemsPerThread; i++)
                    {
                        queue.Enqueue(threadId * itemsPerThread + i);
                    }
                })
            ).ToArray();

            await Task.WhenAll(tasks);

            // Assert
            Assert.Equal(threadCount * itemsPerThread, queue.Count);
        }

        [Fact]
        public async Task ConcurrentEnqueue_HighContention_RaceFree()
        {
            // Arrange
            using var queue = new SharedQueue<int>(1000);
            const int producerCount = 100;
            const int itemsPerProducer = 10;

            // Act
            var tasks = Enumerable.Range(0, producerCount).Select(producerId =>
                Task.Run(() =>
                {
                    for (int i = 0; i < itemsPerProducer; i++)
                    {
                        queue.Enqueue(producerId * itemsPerProducer + i);
                    }
                })
            ).ToArray();

            await Task.WhenAll(tasks);

            // Assert
            Assert.Equal(producerCount * itemsPerProducer, queue.Count);
        }

        #endregion

        #region Dequeue Tests

        [Fact]
        public void Dequeue_SingleItem_ReturnsCorrectItem()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(42);

            // Act
            var item = queue.Dequeue();

            // Assert
            Assert.Equal(42, item);
        }

        [Fact]
        public void Dequeue_MultipleItems_ReturnsInOrder()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);

            // Act
            var item1 = queue.Dequeue();
            var item2 = queue.Dequeue();
            var item3 = queue.Dequeue();

            // Assert - FIFO order
            Assert.Equal(1, item1);
            Assert.Equal(2, item2);
            Assert.Equal(3, item3);
        }

        [Fact]
        public async Task Dequeue_EmptyQueue_BlocksUntilItem()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            var dequeueTask = Task.Run(() => queue.Dequeue());

            // Act
            var completedInTime = await Task.WhenAny(dequeueTask, Task.Delay(100));
            queue.Enqueue(42); // Unblock dequeue
            var item = await dequeueTask;

            // Assert
            Assert.Equal(42, item);
            Assert.Equal(TaskStatus.RanToCompletion, dequeueTask.Status);
        }

        [Fact]
        public void Dequeue_AfterCompleteAdding_ReturnsThenThrows()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.CompleteAdding();

            // Act & Assert
            Assert.Equal(1, queue.Dequeue());
            Assert.Equal(2, queue.Dequeue());
            Assert.Throws<InvalidOperationException>(() => queue.Dequeue());
        }

        [Fact]
        public async Task ConcurrentDequeue_MultipleThreads_ThreadSafe()
        {
            // Arrange
            using var queue = new SharedQueue<int>(1000);
            const int itemCount = 1000;

            for (int i = 0; i < itemCount; i++)
            {
                queue.Enqueue(i);
            }

            // Act
            const int consumerCount = 10;
            var results = new List<int>();
            var lockObj = new object();

            var tasks = Enumerable.Range(0, consumerCount).Select(_ =>
                Task.Run(() =>
                {
                    for (int i = 0; i < itemCount / consumerCount; i++)
                    {
                        var item = queue.Dequeue();
                        lock (lockObj)
                        {
                            results.Add(item);
                        }
                    }
                })
            ).ToArray();

            await Task.WhenAll(tasks);

            // Assert
            Assert.Equal(itemCount, results.Count);
            Assert.All(results, item => Assert.InRange(item, 0, itemCount - 1));
        }

        [Fact]
        public async Task ConcurrentDequeue_HighContention_RaceFree()
        {
            // Arrange
            using var queue = new SharedQueue<int>(1000);
            const int itemCount = 1000;

            for (int i = 0; i < itemCount; i++)
            {
                queue.Enqueue(i);
            }

            // Act
            const int consumerCount = 100;
            var results = new List<int>();
            var lockObj = new object();

            var tasks = Enumerable.Range(0, consumerCount).Select(_ =>
                Task.Run(() =>
                {
                    int itemsToConsume = itemCount / consumerCount;
                    for (int i = 0; i < itemsToConsume; i++)
                    {
                        var item = queue.Dequeue();
                        lock (lockObj)
                        {
                            results.Add(item);
                        }
                    }
                })
            ).ToArray();

            await Task.WhenAll(tasks);

            // Assert
            Assert.Equal(1000, results.Count);
        }

        [Fact]
        public async Task ConcurrentEnqueueDequeue_MultipleProducersConsumers_ThreadSafe()
        {
            // Arrange
            using var queue = new SharedQueue<int>(1000);
            const int producerCount = 5;
            const int consumerCount = 5;
            const int itemsPerProducer = 100;

            var results = new List<int>();
            var lockObj = new object();

            // Act
            var producerTasks = Enumerable.Range(0, producerCount).Select(producerId =>
                Task.Run(() =>
                {
                    for (int i = 0; i < itemsPerProducer; i++)
                    {
                        queue.Enqueue(producerId * itemsPerProducer + i);
                    }
                })
            ).ToArray();

            var consumerTasks = Enumerable.Range(0, consumerCount).Select(_ =>
                Task.Run(() =>
                {
                    int itemsToConsume = (producerCount * itemsPerProducer) / consumerCount;
                    for (int i = 0; i < itemsToConsume; i++)
                    {
                        var item = queue.Dequeue();
                        lock (lockObj)
                        {
                            results.Add(item);
                        }
                    }
                })
            ).ToArray();

            await Task.WhenAll(producerTasks.Concat(consumerTasks));

            // Assert
            Assert.Equal(producerCount * itemsPerProducer, results.Count);
        }

        [Fact]
        public async Task ConcurrentEnqueueDequeue_Balanced_Workload()
        {
            // Arrange
            using var queue = new SharedQueue<int>(1000);
            const int count = 10;
            const int itemsPerEntity = 50;

            var results = new List<int>();
            var lockObj = new object();

            // Act
            var producerTasks = Enumerable.Range(0, count).Select(producerId =>
                Task.Run(() =>
                {
                    for (int i = 0; i < itemsPerEntity; i++)
                    {
                        queue.Enqueue(producerId * itemsPerEntity + i);
                    }
                })
            ).ToArray();

            var consumerTasks = Enumerable.Range(0, count).Select(_ =>
                Task.Run(() =>
                {
                    int itemsToConsume = (count * itemsPerEntity) / count;
                    for (int i = 0; i < itemsToConsume; i++)
                    {
                        var item = queue.Dequeue();
                        lock (lockObj)
                        {
                            results.Add(item);
                        }
                    }
                })
            ).ToArray();

            await Task.WhenAll(producerTasks.Concat(consumerTasks));

            // Assert
            Assert.Equal(count * itemsPerEntity, results.Count);
        }

        [Fact]
        public async Task ConcurrentEnqueueDequeue_Unbalanced_Workload()
        {
            // Arrange
            using var queue = new SharedQueue<int>(1000);
            const int producerCount = 3;
            const int consumerCount = 7;
            const int itemsPerProducer = 100;

            var results = new List<int>();
            var lockObj = new object();

            // Act
            var producerTasks = Enumerable.Range(0, producerCount).Select(producerId =>
                Task.Run(() =>
                {
                    for (int i = 0; i < itemsPerProducer; i++)
                    {
                        queue.Enqueue(producerId * itemsPerProducer + i);
                    }
                })
            ).ToArray();

            var consumerTasks = Enumerable.Range(0, consumerCount).Select(_ =>
                Task.Run(() =>
                {
                    int itemsToConsume = (producerCount * itemsPerProducer) / consumerCount;
                    for (int i = 0; i < itemsToConsume; i++)
                    {
                        var item = queue.Dequeue();
                        lock (lockObj)
                        {
                            results.Add(item);
                        }
                    }
                })
            ).ToArray();

            await Task.WhenAll(producerTasks.Concat(consumerTasks));

            // Assert
            Assert.Equal(producerCount * itemsPerProducer, results.Count);
        }

        #endregion

        #region TryEnqueue Tests

        [Fact]
        public void TryEnqueue_SingleItem_ReturnsTrue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            var success = queue.TryEnqueue(42, 100);

            // Assert
            Assert.True(success);
            Assert.Equal(1, queue.Count);
        }

        [Fact]
        public void TryEnqueue_NonFullQueue_ReturnsTrue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);

            // Act
            var success = queue.TryEnqueue(2, 100);

            // Assert
            Assert.True(success);
            Assert.Equal(2, queue.Count);
        }

        [Fact]
        public void TryEnqueue_FullQueue_Timeout_ReturnsFalse()
        {
            // Arrange
            using var queue = new SharedQueue<int>(2);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Act
            var success = queue.TryEnqueue(3, 10);

            // Assert
            Assert.False(success);
            Assert.Equal(2, queue.Count);
        }

        [Fact]
        public void TryEnqueue_FullQueue_ShortTimeout_ReturnsFalse()
        {
            // Arrange
            using var queue = new SharedQueue<int>(2);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Act
            var success = queue.TryEnqueue(3, 10);

            // Assert
            Assert.False(success);
        }

        [Fact]
        public void TryEnqueue_FullQueue_ZeroTimeout_ReturnsFalse()
        {
            // Arrange
            using var queue = new SharedQueue<int>(2);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Act
            var success = queue.TryEnqueue(3, 0);

            // Assert
            Assert.False(success);
        }

        #endregion

        #region TryDequeue Tests

        [Fact]
        public void TryDequeue_SingleItem_ReturnsTrue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(42);

            // Act
            var success = queue.TryDequeue(out var item, 100);

            // Assert
            Assert.True(success);
            Assert.Equal(42, item);
            Assert.Equal(0, queue.Count);
        }

        [Fact]
        public void TryDequeue_NonEmptyQueue_ReturnsTrue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Act
            var success = queue.TryDequeue(out var item, 100);

            // Assert
            Assert.True(success);
            Assert.Equal(1, item);
            Assert.Equal(1, queue.Count);
        }

        [Fact]
        public void TryDequeue_EmptyQueue_Timeout_ReturnsFalse()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            var success = queue.TryDequeue(out var item, 10);

            // Assert
            Assert.False(success);
            Assert.Equal(default, item);
        }

        [Fact]
        public void TryDequeue_EmptyQueue_ShortTimeout_ReturnsFalse()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            var success = queue.TryDequeue(out var item, 10);

            // Assert
            Assert.False(success);
        }

        [Fact]
        public void TryDequeue_EmptyQueue_ZeroTimeout_ReturnsFalse()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            var success = queue.TryDequeue(out var item, 0);

            // Assert
            Assert.False(success);
        }

        #endregion

        #region TryPeek Tests

        [Fact]
        public void TryPeek_SingleItem_ReturnsTrue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(42);

            // Act
            var success = queue.TryPeek(out var item);

            // Assert
            Assert.True(success);
            Assert.Equal(42, item);
            Assert.Equal(1, queue.Count); // Item not removed
        }

        [Fact]
        public void TryPeek_NonEmptyQueue_ReturnsTrue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);

            // Act
            var success = queue.TryPeek(out var item);

            // Assert
            Assert.True(success);
            Assert.Equal(1, item); // First item
            Assert.Equal(3, queue.Count); // All items still there
        }

        [Fact]
        public void TryPeek_DoesNotRemoveItem()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(42);

            // Act
            queue.TryPeek(out var firstPeek);
            queue.TryPeek(out var secondPeek);

            // Assert
            Assert.Equal(42, firstPeek);
            Assert.Equal(42, secondPeek);
            Assert.Equal(1, queue.Count);
        }

        [Fact]
        public void TryPeek_ReturnsNextItem()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);

            // Act
            queue.TryPeek(out var item);

            // Assert - FIFO order
            Assert.Equal(1, item);
        }

        [Fact]
        public void TryPeek_EmptyQueue_ReturnsFalse()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            var success = queue.TryPeek(out var item);

            // Assert
            Assert.False(success);
            Assert.Equal(default, item);
        }

        [Fact]
        public void TryPeek_CompletedEmptyQueue_ReturnsFalse()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.CompleteAdding();

            // Act
            var success = queue.TryPeek(out var item);

            // Assert
            Assert.False(success);
        }

        #endregion

        #region CompleteAdding Tests

        [Fact]
        public void CompleteAdding_PreventsFurtherEnqueue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);

            // Act
            queue.CompleteAdding();

            // Assert
            Assert.Throws<InvalidOperationException>(() => queue.Enqueue(2));
        }

        [Fact]
        public void CompleteAdding_AllowsDequeueOfRemainingItems()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Act
            queue.CompleteAdding();
            var item1 = queue.Dequeue();
            var item2 = queue.Dequeue();

            // Assert
            Assert.Equal(1, item1);
            Assert.Equal(2, item2);
        }

        [Fact]
        public void CompleteAdding_MultipleCalls_Idempotent()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);

            // Act
            queue.CompleteAdding();
            queue.CompleteAdding();
            queue.CompleteAdding();

            // Assert - No exception thrown
            Assert.Throws<InvalidOperationException>(() => queue.Enqueue(2));
        }

        [Fact]
        public void CompleteAdding_EmptyQueue_SetsIsCompleted()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            queue.CompleteAdding();

            // Assert
            Assert.True(queue.IsCompleted);
        }

        [Fact]
        public void CompleteAdding_NonEmptyQueue_IsCompletedFalseUntilEmpty()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Act
            queue.CompleteAdding();
            var isCompletedBefore = queue.IsCompleted;
            queue.Dequeue();
            queue.Dequeue();
            var isCompletedAfter = queue.IsCompleted;

            // Assert
            Assert.False(isCompletedBefore);
            Assert.True(isCompletedAfter);
        }

        [Fact]
        public void CompleteAdding_AfterCompleteAdding_ThrowsInvalidOperationException()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.CompleteAdding();

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => queue.Enqueue(1));
        }

        #endregion

        #region WaitForCompletion Tests

        [Fact]
        public void WaitForCompletion_EmptyCompletedQueue_ReturnsImmediately()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.CompleteAdding();

            // Act
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            queue.WaitForCompletion();
            stopwatch.Stop();

            // Assert
            Assert.True(stopwatch.ElapsedMilliseconds < 100);
        }

        [Fact]
        public void WaitForCompletion_NonEmptyCompletedQueue_WaitsForEmpty()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.CompleteAdding();

            // Act
            var waitForTask = Task.Run(() => queue.WaitForCompletion());
            var completedInTime = Task.WhenAny(waitForTask, Task.Delay(50)).Result;
            Assert.Equal(waitForTask, completedInTime); // Should still be waiting

            queue.Dequeue();
            queue.Dequeue();

            waitForTask.Wait();

            // Assert
            Assert.True(waitForTask.IsCompleted);
        }

        [Fact]
        public void WaitForCompletion_WithoutCompleteAdding_ReturnsAfterItems()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Act
            var waitForTask = Task.Run(() => queue.WaitForCompletion());
            var completedInTime = Task.WhenAny(waitForTask, Task.Delay(50)).Result;
            Assert.Equal(waitForTask, completedInTime); // Should still be waiting

            queue.CompleteAdding();
            queue.Dequeue();
            queue.Dequeue();

            waitForTask.Wait();

            // Assert
            Assert.True(waitForTask.IsCompleted);
        }

        #endregion

        #region Shutdown Tests

        [Fact]
        public void Shutdown_ImmediatelyStopsQueue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Act
            queue.Shutdown();

            // Assert
            Assert.True(queue.IsCompleted);
        }

        [Fact]
        public void Shutdown_AfterShutdown_CannotEnqueue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Shutdown();

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => queue.Enqueue(1));
        }

        [Fact]
        public void Shutdown_AfterShutdown_CannotDequeue()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);
            queue.Shutdown();

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => queue.Dequeue());
        }

        [Fact]
        public void Shutdown_MultipleCalls_Idempotent()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            queue.Shutdown();
            queue.Shutdown();
            queue.Shutdown();

            // Assert - No exception thrown
            Assert.True(queue.IsCompleted);
        }

        #endregion

        #region Statistics Tests

        [Fact]
        public void Statistics_InitiallyZero_ReturnsZeroValues()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10, enableStatistics: true);

            // Act
            var stats = queue.GetStatistics();

            // Assert
            Assert.Equal(0, stats.TotalEnqueued);
            Assert.Equal(0, stats.TotalDequeued);
            Assert.Equal(0, stats.MaxQueueSize);
            Assert.Equal(0, stats.ProducerWaitCount);
            Assert.Equal(0, stats.ConsumerWaitCount);
        }

        [Fact]
        public void Statistics_AfterEnqueue_IncrementsTotalEnqueued()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10, enableStatistics: true);

            // Act
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);

            var stats = queue.GetStatistics();

            // Assert
            Assert.Equal(3, stats.TotalEnqueued);
        }

        [Fact]
        public void Statistics_AfterDequeue_IncrementsTotalDequeued()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10, enableStatistics: true);
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);

            // Act
            queue.Dequeue();
            queue.Dequeue();

            var stats = queue.GetStatistics();

            // Assert
            Assert.Equal(2, stats.TotalDequeued);
        }

        [Fact]
        public void Statistics_EnqueueDequeue_BothIncremented()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10, enableStatistics: true);

            // Act
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Dequeue();

            var stats = queue.GetStatistics();

            // Assert
            Assert.Equal(2, stats.TotalEnqueued);
            Assert.Equal(1, stats.TotalDequeued);
        }

        [Fact]
        public void Statistics_MaxQueueSize_TrackedCorrectly()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10, enableStatistics: true);

            // Act
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);
            queue.Dequeue();
            queue.Enqueue(4);
            queue.Enqueue(5);

            var stats = queue.GetStatistics();

            // Assert - Max should be 3 (when we had 3 items)
            Assert.Equal(3, stats.MaxQueueSize);
        }

        #endregion

        #region Batch Operations Tests

        [Fact]
        public void EnqueueBatch_MultipleItems_IncrementsCountCorrectly()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10, enableStatistics: true);

            // Act
            queue.EnqueueBatch(new[] { 1, 2, 3, 4, 5 });

            // Assert
            Assert.Equal(5, queue.Count);
            Assert.Equal(5, queue.GetStatistics().TotalEnqueued);
        }

        [Fact]
        public async Task EnqueueBatch_FullQueue_BlocksUntilSpace()
        {
            // Arrange
            using var queue = new SharedQueue<int>(5);

            // Act
            queue.EnqueueBatch(new[] { 1, 2, 3, 4, 5 });

            var enqueueTask = Task.Run(() =>
            {
                queue.EnqueueBatch(new[] { 6, 7 }); // Should block
            });

            var completedInTime = await Task.WhenAny(enqueueTask, Task.Delay(100));
            Assert.Equal(enqueueTask, completedInTime); // Should be waiting

            queue.Dequeue();
            await enqueueTask;

            // Assert
            Assert.Equal(5, queue.Count);
        }

        [Fact]
        public void DequeueBatch_MultipleItems_ReturnsCorrectItems()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10, enableStatistics: true);
            queue.EnqueueBatch(new[] { 1, 2, 3, 4, 5 });

            // Act
            var items = queue.DequeueBatch(3, 1000);

            // Assert
            Assert.Equal(3, items.Length);
            Assert.Equal(1, items[0]);
            Assert.Equal(2, items[1]);
            Assert.Equal(3, items[2]);
        }

        [Fact]
        public void DequeueBatch_RequestedMoreThanAvailable_ReturnsAvailable()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10, enableStatistics: true);
            queue.EnqueueBatch(new[] { 1, 2 });

            // Act
            var items = queue.DequeueBatch(5, 100);

            // Assert
            Assert.Equal(2, items.Length);
            Assert.Equal(1, items[0]);
            Assert.Equal(2, items[1]);
        }

        [Fact]
        public void DequeueBatch_EmptyQueue_Timeout_ReturnsEmptyArray()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10, enableStatistics: true);

            // Act
            var items = queue.DequeueBatch(5, 10);

            // Assert
            Assert.Equal(0, items.Length);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Enqueue_NullItem_Succeeds()
        {
            // Arrange
            using var queue = new SharedQueue<string?>(10);

            // Act
            queue.Enqueue(null);

            // Assert
            Assert.Equal(1, queue.Count);
        }

        [Fact]
        public void Dequeue_NullItem_ReturnsNull()
        {
            // Arrange
            using var queue = new SharedQueue<string?>(10);
            queue.Enqueue(null);

            // Act
            var item = queue.Dequeue();

            // Assert
            Assert.Null(item);
            Assert.Equal(0, queue.Count);
        }

        [Fact]
        public async Task FullQueue_Enqueue_Blocks()
        {
            // Arrange
            using var queue = new SharedQueue<int>(2);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Act
            var enqueueTask = Task.Run(() => queue.Enqueue(3));
            var completedInTime = await Task.WhenAny(enqueueTask, Task.Delay(50));

            // Assert
            Assert.Equal(enqueueTask, completedInTime); // Should be blocking
            enqueueTask.IsCompleted.Should().BeFalse();
        }

        [Fact]
        public async Task EmptyQueue_Dequeue_Blocks()
        {
            // Arrange
            using var queue = new SharedQueue<int>(10);

            // Act
            var dequeueTask = Task.Run(() => queue.Dequeue());
            var completedInTime = await Task.WhenAny(dequeueTask, Task.Delay(50));

            // Assert
            Assert.Equal(dequeueTask, completedInTime); // Should be blocking
            dequeueTask.IsCompleted.Should().BeFalse();
        }

        #endregion
    }
}
