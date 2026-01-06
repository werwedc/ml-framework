using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Data.Worker;
using Xunit;

namespace MLFramework.Tests.Data.Worker
{
    public class SharedMemoryQueueTests
    {
        [Fact]
        public void Constructor_WithNegativeMaxSize_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new SharedMemoryQueue<int>(-1));
        }

        [Fact]
        public void Constructor_WithZeroMaxSize_CreatesUnboundedQueue()
        {
            using var queue = new SharedMemoryQueue<int>(0);
            Assert.Equal(0, queue.MaxSize);
        }

        [Fact]
        public void Enqueue_WhenNotDisposed_AddsItem()
        {
            using var queue = new SharedMemoryQueue<int>();
            queue.Enqueue(1);
            Assert.Equal(1, queue.Count);
        }

        [Fact]
        public void Enqueue_AfterDisposal_ThrowsObjectDisposedException()
        {
            var queue = new SharedMemoryQueue<int>();
            queue.Dispose();
            Assert.Throws<ObjectDisposedException>(() => queue.Enqueue(1));
        }

        [Fact]
        public void Dequeue_ReturnsItemsInFIFOOrder()
        {
            using var queue = new SharedMemoryQueue<int>();
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);

            Assert.Equal(1, queue.Dequeue());
            Assert.Equal(2, queue.Dequeue());
            Assert.Equal(3, queue.Dequeue());
        }

        [Fact]
        public void Dequeue_AfterDisposal_ThrowsObjectDisposedException()
        {
            using var queue = new SharedMemoryQueue<int>();
            queue.Enqueue(1);
            queue.Dispose();
            Assert.Throws<ObjectDisposedException>(() => queue.Dequeue());
        }

        [Fact]
        public void Dequeue_WhenEmpty_ThrowsInvalidOperationException()
        {
            using var queue = new SharedMemoryQueue<int>();
            Assert.Throws<InvalidOperationException>(() => queue.Dequeue());
        }

        [Fact]
        public void TryDequeue_WhenEmpty_ReturnsFalse()
        {
            using var queue = new SharedMemoryQueue<int>();
            Assert.False(queue.TryDequeue(out _));
        }

        [Fact]
        public void TryDequeue_AfterDisposal_ReturnsFalse()
        {
            var queue = new SharedMemoryQueue<int>();
            queue.Enqueue(1);
            queue.Dispose();
            Assert.False(queue.TryDequeue(out _));
        }

        [Fact]
        public void TryDequeue_WhenNotEmpty_ReturnsTrueAndItem()
        {
            using var queue = new SharedMemoryQueue<int>();
            queue.Enqueue(42);

            Assert.True(queue.TryDequeue(out var item));
            Assert.Equal(42, item);
        }

        [Fact]
        public void Count_ReturnsCorrectNumberOfItems()
        {
            using var queue = new SharedMemoryQueue<int>();
            Assert.Equal(0, queue.Count);

            queue.Enqueue(1);
            Assert.Equal(1, queue.Count);

            queue.Enqueue(2);
            Assert.Equal(2, queue.Count);

            queue.Dequeue();
            Assert.Equal(1, queue.Count);
        }

        [Fact]
        public void IsEmpty_WhenEmpty_ReturnsTrue()
        {
            using var queue = new SharedMemoryQueue<int>();
            Assert.True(queue.IsEmpty);
        }

        [Fact]
        public void IsEmpty_WhenNotEmpty_ReturnsFalse()
        {
            using var queue = new SharedMemoryQueue<int>();
            queue.Enqueue(1);
            Assert.False(queue.IsEmpty);
        }

        [Fact]
        public void Clear_RemovesAllItems()
        {
            using var queue = new SharedMemoryQueue<int>();
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);

            queue.Clear();

            Assert.Equal(0, queue.Count);
            Assert.True(queue.IsEmpty);
        }

        [Fact]
        public void Clear_AfterDisposal_ThrowsObjectDisposedException()
        {
            var queue = new SharedMemoryQueue<int>();
            queue.Enqueue(1);
            queue.Dispose();
            Assert.Throws<ObjectDisposedException>(() => queue.Clear());
        }

        [Fact]
        public void BoundedQueue_EnqueueBeyondMaxSize_Blocks()
        {
            using var queue = new SharedMemoryQueue<int>(maxSize: 2);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // This should block until we dequeue
            var enqueueTask = Task.Run(() => queue.Enqueue(3));

            // Enqueue should not complete quickly
            Assert.False(enqueueTask.Wait(100));

            // Dequeue to allow the enqueue to complete
            queue.Dequeue();

            // Now the enqueue should complete
            Assert.True(enqueueTask.Wait(1000));
            Assert.Equal(2, queue.Count);
        }

        [Fact]
        public void BoundedQueue_TryDequeueReturnsFalseWhenEmpty()
        {
            using var queue = new SharedMemoryQueue<int>(maxSize: 2);
            Assert.False(queue.TryDequeue(out _));
        }

        [Fact]
        public void ConcurrentEnqueueAndDequeue_ThreadSafe()
        {
            using var queue = new SharedMemoryQueue<int>(maxSize: 100);
            const int itemCount = 1000;
            var producerCompleted = new ManualResetEvent(false);
            var consumerCount = 0;

            // Producer
            var producerTask = Task.Run(() =>
            {
                for (int i = 0; i < itemCount; i++)
                {
                    queue.Enqueue(i);
                }
                producerCompleted.Set();
            });

            // Consumer
            var consumerTask = Task.Run(() =>
            {
                while (!producerCompleted.WaitOne(0) || !queue.IsEmpty)
                {
                    if (queue.TryDequeue(out _))
                    {
                        Interlocked.Increment(ref consumerCount);
                    }
                }
            });

            Task.WaitAll(producerTask, consumerTask);

            Assert.Equal(itemCount, consumerCount);
        }

        [Fact]
        public void MultipleProducersAndConsumers_ThreadSafe()
        {
            using var queue = new SharedMemoryQueue<int>(maxSize: 100);
            const int itemsPerProducer = 100;
            const int producerCount = 4;
            const int consumerCount = 4;
            var producersComplete = new CountdownEvent(producerCount);
            var consumerCount = 0;

            // Producers
            var producerTasks = new Task[producerCount];
            for (int p = 0; p < producerCount; p++)
            {
                producerTasks[p] = Task.Run(() =>
                {
                    for (int i = 0; i < itemsPerProducer; i++)
                    {
                        queue.Enqueue(i);
                    }
                    producersComplete.Signal();
                });
            }

            // Consumers
            var consumerTasks = new Task[consumerCount];
            for (int c = 0; c < consumerCount; c++)
            {
                consumerTasks[c] = Task.Run(() =>
                {
                    while (!producersComplete.IsSet || !queue.IsEmpty)
                    {
                        if (queue.TryDequeue(out _))
                        {
                            Interlocked.Increment(ref consumerCount);
                        }
                    }
                });
            }

            Task.WaitAll(producerTasks.Concat(consumerTasks).ToArray());

            Assert.Equal(itemsPerProducer * producerCount, consumerCount);
        }

        [Fact]
        public void Dispose_UnblocksWaitingOperations()
        {
            var queue = new SharedMemoryQueue<int>(maxSize: 2);
            queue.Enqueue(1);
            queue.Enqueue(2);

            // Start a task that will block on enqueue
            var blockedTask = Task.Run(() => queue.Enqueue(3));
            Assert.False(blockedTask.Wait(100));

            // Dispose should unblock the task
            queue.Dispose();
            Assert.True(blockedTask.Wait(1000));
        }

        [Fact]
        public void UnboundedQueue_NeverBlocksOnEnqueue()
        {
            using var queue = new SharedMemoryQueue<int>(maxSize: 0);

            // Should be able to enqueue many items without blocking
            for (int i = 0; i < 1000; i++)
            {
                queue.Enqueue(i);
            }

            Assert.Equal(1000, queue.Count);
        }

        [Fact]
        public void MultipleDisposeCalls_DoesNotThrow()
        {
            var queue = new SharedMemoryQueue<int>();
            queue.Dispose();
            queue.Dispose(); // Should not throw
        }
    }

    // Extension to expose MaxSize for testing
    internal static class SharedMemoryQueueExtensions
    {
        public static int MaxSize(this SharedMemoryQueue<int> queue)
        {
            var field = queue.GetType().GetField("_maxSize", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return (int)field?.GetValue(queue);
        }
    }
}
