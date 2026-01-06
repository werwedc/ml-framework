using System;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Data.Worker;
using Xunit;

namespace MLFramework.Tests.Data.Worker
{
    /// <summary>
    /// Unit tests for WorkerPool functionality.
    /// </summary>
    public class WorkerPoolTests : IDisposable
    {
        private WorkerPool _pool;

        public void Dispose()
        {
            _pool?.Dispose();
        }

        [Fact]
        public void Constructor_WithValidNumWorkers_CreatesPool()
        {
            // Arrange & Act
            using (var pool = new WorkerPool(numWorkers: 4))
            {
                // Assert
                Assert.Equal(4, pool.NumWorkers);
                Assert.False(pool.IsRunning);
            }
        }

        [Fact]
        public void Constructor_WithZeroWorkers_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new WorkerPool(0));
        }

        [Fact]
        public void Constructor_WithNegativeWorkers_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new WorkerPool(-1));
        }

        [Fact]
        public void Start_WhenCalled_StartsWorkersAndSetsIsRunning()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                // Act
                pool.Start();

                // Assert
                Assert.True(pool.IsRunning);
            }
        }

        [Fact]
        public void Start_WhenAlreadyRunning_DoesNothing()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                pool.Start();
                var firstRunningState = pool.IsRunning;

                // Act
                pool.Start();

                // Assert
                Assert.Equal(firstRunningState, pool.IsRunning);
            }
        }

        [Fact]
        public void SubmitTask_WhenNotRunning_ThrowsInvalidOperationException()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                // Act & Assert
                Assert.Throws<InvalidOperationException>(() =>
                    pool.SubmitTask(() => 42));
            }
        }

        [Fact]
        public void SubmitTask_WithNullTask_ThrowsArgumentNullException()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                pool.Start();

                // Act & Assert
                Assert.Throws<ArgumentNullException>(() =>
                    pool.SubmitTask<int>(null));
            }
        }

        [Fact]
        public void GetResult_WhenNotRunning_ThrowsInvalidOperationException()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                // Act & Assert
                Assert.Throws<InvalidOperationException>(() =>
                    pool.GetResult<int>());
            }
        }

        [Fact]
        public void SubmitTaskAndGetResult_ExecutesTaskAndReturnsCorrectResult()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                pool.Start();
                int expected = 42;

                // Act
                pool.SubmitTask(() => expected);
                int actual = pool.GetResult<int>();

                // Assert
                Assert.Equal(expected, actual);
            }
        }

        [Fact]
        public void TryGetResult_WhenNotRunning_ReturnsFalse()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                // Act
                bool result = pool.TryGetResult<int>(out var value);

                // Assert
                Assert.False(result);
                Assert.Equal(default, value);
            }
        }

        [Fact]
        public void TryGetResult_WithNoResults_ReturnsFalse()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                pool.Start();

                // Act
                bool result = pool.TryGetResult<int>(out var value);

                // Assert
                Assert.False(result);
                Assert.Equal(default, value);
            }
        }

        [Fact]
        public void TryGetResult_WithResult_ReturnsTrueAndValue()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                pool.Start();
                int expected = 100;
                pool.SubmitTask(() => expected);

                // Wait a bit for task to complete
                Thread.Sleep(100);

                // Act
                bool result = pool.TryGetResult<int>(out var value);

                // Assert
                Assert.True(result);
                Assert.Equal(expected, value);
            }
        }

        [Fact]
        public void GetResult_RetrievesResultsInOrder()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 4))
            {
                pool.Start();

                // Submit tasks in order
                pool.SubmitTask(() => 1);
                pool.SubmitTask(() => 2);
                pool.SubmitTask(() => 3);
                pool.SubmitTask(() => 4);

                // Act
                var result1 = pool.GetResult<int>();
                var result2 = pool.GetResult<int>();
                var result3 = pool.GetResult<int>();
                var result4 = pool.GetResult<int>();

                // Assert - Results should come in the order submitted
                Assert.Equal(1, result1);
                Assert.Equal(2, result2);
                Assert.Equal(3, result3);
                Assert.Equal(4, result4);
            }
        }

        [Fact]
        public void MultipleTasks_ExecuteInParallel()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 4))
            {
                pool.Start();
                var startTime = DateTime.Now;

                // Submit 8 tasks that each take 100ms
                for (int i = 0; i < 8; i++)
                {
                    int index = i;
                    pool.SubmitTask(() =>
                    {
                        Thread.Sleep(100);
                        return index;
                    });
                }

                // Retrieve all results
                for (int i = 0; i < 8; i++)
                {
                    pool.GetResult<int>();
                }

                var endTime = DateTime.Now;
                var duration = (endTime - startTime).TotalMilliseconds;

                // Assert - With 4 workers, 8 tasks of 100ms each should take less than 800ms
                // (ideally around 200-300ms if perfectly parallel)
                Assert.True(duration < 700,
                    $"Expected parallel execution to complete in < 700ms, but took {duration}ms");
            }
        }

        [Fact]
        public void WorkerException_DoesNotCrashPool()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                pool.Start();

                // Submit a task that throws an exception
                pool.SubmitTask<int>(() => throw new InvalidOperationException("Test exception"));

                // Submit a valid task after the failing one
                pool.SubmitTask(() => 42);

                // Act & Assert - Pool should still be running
                Assert.True(pool.IsRunning);

                // Try to get result - the good task should still complete
                // We skip the bad task's result (it may be null or have an error marker)
                bool gotResult = pool.TryGetResult<int>(out _);

                // Try again for the good task
                bool gotGoodResult = pool.TryGetResult<int>(out var result);

                // Assert - The pool is still running and can process valid tasks
                Assert.True(pool.IsRunning);
            }
        }

        [Fact]
        public void Stop_WhenRunning_CancelsAllWorkers()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                pool.Start();

                // Act
                pool.Stop();

                // Assert
                Assert.False(pool.IsRunning);
            }
        }

        [Fact]
        public void Stop_WhenNotRunning_DoesNothing()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                // Act - Should not throw
                pool.Stop();

                // Assert
                Assert.False(pool.IsRunning);
            }
        }

        [Fact]
        public void Dispose_CleansUpAllResources()
        {
            // Arrange
            var pool = new WorkerPool(numWorkers: 2);
            pool.Start();
            pool.SubmitTask(() => 42);
            pool.GetResult<int>();

            // Act
            pool.Dispose();

            // Assert - IsRunning should be false and all resources should be cleaned up
            Assert.False(pool.IsRunning);
        }

        [Fact]
        public void MultipleSubmitAndGetResultCalls_WorkCorrectly()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                pool.Start();
                int numTasks = 20;

                // Submit multiple tasks
                for (int i = 0; i < numTasks; i++)
                {
                    int index = i;
                    pool.SubmitTask(() => index * 2);
                }

                // Retrieve all results and verify
                int[] results = new int[numTasks];
                for (int i = 0; i < numTasks; i++)
                {
                    results[i] = pool.GetResult<int>();
                }

                // Assert - All tasks should complete
                Assert.All(results, r => Assert.True(r >= 0 && r < numTasks * 2));
            }
        }

        [Fact]
        public void DifferentTaskTypes_WorkCorrectly()
        {
            // Arrange
            using (var pool = new WorkerPool(numWorkers: 2))
            {
                pool.Start();

                // Submit different types of tasks
                pool.SubmitTask(() => 42);
                pool.SubmitTask(() => "Hello");
                pool.SubmitTask(() => 3.14);
                pool.SubmitTask(() => true);

                // Act
                var intResult = pool.GetResult<int>();
                var stringResult = pool.GetResult<string>();
                var doubleResult = pool.GetResult<double>();
                var boolResult = pool.GetResult<bool>();

                // Assert
                Assert.Equal(42, intResult);
                Assert.Equal("Hello", stringResult);
                Assert.Equal(3.14, doubleResult);
                Assert.True(boolResult);
            }
        }
    }
}
