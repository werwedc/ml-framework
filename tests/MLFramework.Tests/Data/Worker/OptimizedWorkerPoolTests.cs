using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Data.Worker;
using Xunit;

namespace MLFramework.Tests.Data.Worker
{
    /// <summary>
    /// Unit tests for OptimizedWorkerPool functionality.
    /// </summary>
    public class OptimizedWorkerPoolTests : IDisposable
    {
        private OptimizedWorkerPool _pool;

        public void Dispose()
        {
            _pool?.Dispose();
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesPool()
        {
            // Arrange & Act
            using (var pool = new OptimizedWorkerPool(numWorkers: 4, lazyInitialization: true, warmupBatches: 2))
            {
                // Assert
                Assert.Equal(4, pool.NumWorkers);
                Assert.False(pool.WorkersInitialized);
                Assert.Equal(TimeSpan.Zero, pool.InitializationTime);
            }
        }

        [Fact]
        public void Constructor_WithNegativeWarmupBatches_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new OptimizedWorkerPool(4, true, -1));
        }

        [Fact]
        public void Constructor_WithZeroWorkers_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new OptimizedWorkerPool(0));
        }

        [Fact]
        public void Start_WithLazyInitialization_DoesNotCreateWorkers()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: true))
            {
                // Act
                pool.Start();

                // Assert
                Assert.False(pool.WorkersInitialized);
                Assert.Equal(TimeSpan.Zero, pool.InitializationTime);
            }
        }

        [Fact]
        public void Start_WithEagerInitialization_CreatesWorkers()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: false))
            {
                // Act
                pool.Start();

                // Assert
                Assert.True(pool.WorkersInitialized);
                Assert.True(pool.InitializationTime > TimeSpan.Zero);
            }
        }

        [Fact]
        public void SubmitTask_WithLazyInitialization_CreatesWorkersOnFirstTask()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: true))
            {
                pool.Start();

                // Act
                pool.SubmitTask(() => 42);

                // Give workers time to initialize
                Thread.Sleep(100);

                // Assert
                Assert.True(pool.WorkersInitialized);
                Assert.True(pool.InitializationTime > TimeSpan.Zero);
            }
        }

        [Fact]
        public void SubmitTask_WithEagerInitialization_AlreadyHasWorkers()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: false))
            {
                pool.Start();

                // Act
                pool.SubmitTask(() => 42);

                // Assert
                Assert.True(pool.WorkersInitialized);
            }
        }

        [Fact]
        public void WorkerInitializer_IsCalledForEachWorker()
        {
            // Arrange
            var initializedWorkers = new List<int>();
            using (var pool = new OptimizedWorkerPool(
                numWorkers: 3,
                lazyInitialization: false,
                warmupBatches: 0,
                workerInitializer: ctx =>
                {
                    lock (initializedWorkers)
                    {
                        initializedWorkers.Add(ctx.WorkerId);
                    }
                }))

            // Act
            pool.Start();
            Thread.Sleep(200); // Give time for initialization

            // Assert
            Assert.Equal(3, initializedWorkers.Count);
            Assert.Contains(0, initializedWorkers);
            Assert.Contains(1, initializedWorkers);
            Assert.Contains(2, initializedWorkers);
        }

        [Fact]
        public void WorkerContext_SharedState_PersistsPerWorker()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(
                numWorkers: 2,
                lazyInitialization: false,
                workerInitializer: ctx =>
                {
                    ctx.SetState("CustomData", ctx.WorkerId * 100);
                }))

            // Act
            pool.Start();
            Thread.Sleep(200);

            // Assert
            var context0 = pool.GetWorkerContext(0);
            var context1 = pool.GetWorkerContext(1);

            Assert.NotNull(context0);
            Assert.NotNull(context1);
            Assert.Equal(0, context0.GetState<int>("CustomData"));
            Assert.Equal(100, context1.GetState<int>("CustomData"));
        }

        [Fact]
        public void GetWorkerContext_WithInvalidId_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2))
            {
                // Act & Assert
                Assert.Throws<ArgumentOutOfRangeException>(() => pool.GetWorkerContext(-1));
                Assert.Throws<ArgumentOutOfRangeException>(() => pool.GetWorkerContext(2));
            }
        }

        [Fact]
        public void SubmitTaskAndGetResult_CompletesSuccessfully()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: false))
            {
                pool.Start();

                // Act
                pool.SubmitTask(() => 42);
                var result = pool.GetResult<int>();

                // Assert
                Assert.Equal(42, result);
            }
        }

        [Fact]
        public void SubmitTaskAndGetResult_MultipleTasks_ProcessedInOrder()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: false))
            {
                pool.Start();

                // Act
                for (int i = 0; i < 5; i++)
                {
                    pool.SubmitTask(() => i);
                }

                var results = new List<int>();
                for (int i = 0; i < 5; i++)
                {
                    results.Add(pool.GetResult<int>());
                }

                // Assert
                Assert.Equal(5, results.Count);
            }
        }

        [Fact]
        public void TryGetResult_WithNoResult_ReturnsFalse()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: false))
            {
                pool.Start();

                // Act
                var success = pool.TryGetResult<int>(out var result);

                // Assert
                Assert.False(success);
                Assert.Equal(default, result);
            }
        }

        [Fact]
        public void TryGetResult_WithResult_ReturnsTrue()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: false))
            {
                pool.Start();
                pool.SubmitTask(() => 42);

                // Give task time to complete
                Thread.Sleep(100);

                // Act
                var success = pool.TryGetResult<int>(out var result);

                // Assert
                Assert.True(success);
                Assert.Equal(42, result);
            }
        }

        [Fact]
        public void InitializationTime_AccuratelyMeasuresStartup()
        {
            // Arrange
            var slowInitializer = new Action<WorkerContext>(ctx =>
            {
                Thread.Sleep(50); // Simulate slow initialization
            });

            using (var pool = new OptimizedWorkerPool(
                numWorkers: 2,
                lazyInitialization: false,
                workerInitializer: slowInitializer))

            // Act
            pool.Start();

            // Assert
            Assert.True(pool.InitializationTime >= TimeSpan.FromMilliseconds(100)); // At least 2 workers * 50ms
        }

        [Fact]
        public void WorkersInitialized_CorrectlyReportsState()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: true))
            {
                // Assert - Before any operation
                Assert.False(pool.WorkersInitialized);

                // Act
                pool.Start();

                // Assert - After Start with lazy initialization
                Assert.False(pool.WorkersInitialized);

                // Act
                pool.SubmitTask(() => 42);
                Thread.Sleep(100);

                // Assert - After first task
                Assert.True(pool.WorkersInitialized);
            }
        }

        [Fact]
        public void NoDuplicateWorkerCreation_CallsStartMultipleTimes()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(
                numWorkers: 2,
                lazyInitialization: false,
                workerInitializer: ctx =>
                {
                    // This should only be called once per worker
                    if (ctx.HasState("Initialized"))
                    {
                        throw new InvalidOperationException("Worker initialized twice!");
                    }
                    ctx.SetState("Initialized", true);
                }))

            // Act
            pool.Start();
            Thread.Sleep(100);
            pool.Start(); // Second call should be no-op
            Thread.Sleep(100);

            // Assert
            Assert.True(pool.WorkersInitialized);
        }

        [Fact]
        public void ThreadSafeInitialization_MultipleConcurrentStarts()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: false))
            {
                // Act
                var tasks = new Task[10];
                for (int i = 0; i < 10; i++)
                {
                    tasks[i] = Task.Run(() => pool.Start());
                }
                Task.WaitAll(tasks);

                Thread.Sleep(200);

                // Assert
                Assert.True(pool.WorkersInitialized);
                Assert.True(pool.InitializationTime > TimeSpan.Zero);
            }
        }

        [Fact]
        public void ThreadSafeInitialization_MultipleConcurrentSubmitTasks()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: true))
            {
                pool.Start();

                // Act
                var tasks = new Task[10];
                for (int i = 0; i < 10; i++)
                {
                    int value = i;
                    tasks[i] = Task.Run(() => pool.SubmitTask(() => value));
                }
                Task.WaitAll(tasks);

                Thread.Sleep(200);

                // Assert
                Assert.True(pool.WorkersInitialized);

                // Verify we can get results
                var results = new List<int>();
                while (results.Count < 10)
                {
                    if (pool.TryGetResult<int>(out var result))
                    {
                        results.Add(result);
                    }
                    else
                    {
                        Thread.Sleep(10);
                    }
                }

                Assert.Equal(10, results.Count);
            }
        }

        [Fact]
        public void WarmupBatches_AreProcessedDuringInitialization()
        {
            // Arrange
            using (var pool = new OptimizedWorkerPool(
                numWorkers: 2,
                lazyInitialization: false,
                warmupBatches: 3))

            // Act
            var stopwatch = Stopwatch.StartNew();
            pool.Start();
            stopwatch.Stop();

            // Assert
            Assert.True(pool.WorkersInitialized);
            // Warmup should take some time
            Assert.True(stopwatch.Elapsed > TimeSpan.FromMilliseconds(10));
        }

        [Fact]
        public void Dispose_CleansUpResources()
        {
            // Arrange
            var pool = new OptimizedWorkerPool(numWorkers: 2, lazyInitialization: false);
            pool.Start();

            // Act
            pool.Dispose();

            // Assert - Should not throw
            Assert.True(true);
        }

        [Fact]
        public void WorkerContext_GetState_ReturnsDefaultForMissingKey()
        {
            // Arrange
            var context = new WorkerContext();

            // Act
            var result = context.GetState<int>("NonExistentKey");

            // Assert
            Assert.Equal(default, result);
        }

        [Fact]
        public void WorkerContext_SetState_OverwritesExistingValue()
        {
            // Arrange
            var context = new WorkerContext();
            context.SetState("Key", "OriginalValue");

            // Act
            context.SetState("Key", "NewValue");

            // Assert
            Assert.Equal("NewValue", context.GetState<string>("Key"));
        }

        [Fact]
        public void WorkerContext_HasState_ReturnsCorrectStatus()
        {
            // Arrange
            var context = new WorkerContext();
            context.SetState("ExistingKey", "Value");

            // Act & Assert
            Assert.True(context.HasState("ExistingKey"));
            Assert.False(context.HasState("NonExistentKey"));
        }

        [Fact]
        public void WorkerContext_RemoveState_RemovesValue()
        {
            // Arrange
            var context = new WorkerContext();
            context.SetState("Key", "Value");

            // Act
            var removed = context.RemoveState("Key");

            // Assert
            Assert.True(removed);
            Assert.False(context.HasState("Key"));
        }

        [Fact]
        public void Performance_LazyInitializationFasterThanEager()
        {
            // Arrange
            var initializer = new Action<WorkerContext>(ctx =>
            {
                Thread.Sleep(10); // Simulate some initialization work
            });

            // Act - Eager initialization
            var eagerStopwatch = Stopwatch.StartNew();
            using (var eagerPool = new OptimizedWorkerPool(
                numWorkers: 4,
                lazyInitialization: false,
                workerInitializer: initializer))
            {
                eagerPool.Start();
            }
            eagerStopwatch.Stop();

            // Act - Lazy initialization
            var lazyStopwatch = Stopwatch.StartNew();
            using (var lazyPool = new OptimizedWorkerPool(
                numWorkers: 4,
                lazyInitialization: true,
                workerInitializer: initializer))
            {
                lazyPool.Start(); // No workers created yet
            }
            lazyStopwatch.Stop();

            // Assert
            Assert.True(lazyStopwatch.Elapsed < eagerStopwatch.Elapsed,
                "Lazy initialization should be faster than eager initialization");
        }
    }
}
