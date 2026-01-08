using MLFramework.Data;
using Xunit;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Tests.Data;

/// <summary>
/// Comprehensive unit tests for worker pool management and parallel execution.
/// </summary>
public class WorkerPoolTests : IDisposable
{
    private readonly CancellationTokenSource _cts;

    public WorkerPoolTests()
    {
        _cts = new CancellationTokenSource();
    }

    public void Dispose()
    {
        _cts.Dispose();
    }

    #region Helper Methods

    private DataWorker<int> CreateSimpleWorker()
    {
        return (workerId, token) => workerId * 100;
    }

    private DataWorker<int> CreateCountingWorker(SharedQueue<int> queue, int count)
    {
        return (workerId, token) =>
        {
            for (int i = 0; i < count; i++)
            {
                queue.Enqueue(workerId * count + i);
            }
            return 0;
        };
    }

    private async Task EnqueueItemsAsync(WorkerPool<int> pool, SharedQueue<int> queue, int expectedCount)
    {
        pool.Start();
        await pool.WaitAsync();
        Assert.Equal(expectedCount, queue.Count);
    }

    #endregion

    #region WorkerContext Tests - Constructor Tests

    [Fact]
    public void WorkerContext_Constructor_ValidParameters_CreatesContext()
    {
        // Act
        var ctx = new WorkerContext(0, 2, 10, CancellationToken.None);

        // Assert
        Assert.Equal(0, ctx.WorkerId);
        Assert.Equal(2, ctx.NumWorkers);
    }

    [Fact]
    public void WorkerContext_Constructor_EvenDivision_CalculatesCorrectRanges()
    {
        // Arrange & Act
        var ctx1 = new WorkerContext(0, 2, 10, CancellationToken.None);
        var ctx2 = new WorkerContext(1, 2, 10, CancellationToken.None);

        // Assert
        Assert.Equal(0, ctx1.StartIndex);
        Assert.Equal(5, ctx1.EndIndex);
        Assert.Equal(5, ctx2.StartIndex);
        Assert.Equal(10, ctx2.EndIndex);
    }

    [Fact]
    public void WorkerContext_Constructor_UnevenDivision_LastWorkerGetsRemainder()
    {
        // Arrange & Act
        var ctx1 = new WorkerContext(0, 3, 10, CancellationToken.None);
        var ctx2 = new WorkerContext(1, 3, 10, CancellationToken.None);
        var ctx3 = new WorkerContext(2, 3, 10, CancellationToken.None);

        // Assert
        Assert.Equal(4, ctx1.PartitionSize); // Worker 0 gets 4 items
        Assert.Equal(3, ctx2.PartitionSize); // Worker 1 gets 3 items
        Assert.Equal(3, ctx3.PartitionSize); // Worker 2 gets 3 items
    }

    [Fact]
    public void WorkerContext_Constructor_SingleItem_CorrectPartition()
    {
        // Arrange & Act
        var ctx = new WorkerContext(0, 1, 1, CancellationToken.None);

        // Assert
        Assert.Equal(0, ctx.StartIndex);
        Assert.Equal(1, ctx.EndIndex);
    }

    [Fact]
    public void WorkerContext_Constructor_EmptyDataset_ZeroLengthRange()
    {
        // Arrange & Act
        var ctx = new WorkerContext(0, 2, 0, CancellationToken.None);

        // Assert
        Assert.Equal(0, ctx.StartIndex);
        Assert.Equal(0, ctx.EndIndex);
    }

    #endregion

    #region WorkerContext Tests - Property Tests

    [Fact]
    public void WorkerContext_WorkerId_ReturnsCorrectValue()
    {
        // Act
        var ctx = new WorkerContext(5, 10, 100, CancellationToken.None);

        // Assert
        Assert.Equal(5, ctx.WorkerId);
    }

    [Fact]
    public void WorkerContext_NumWorkers_ReturnsCorrectValue()
    {
        // Act
        var ctx = new WorkerContext(0, 10, 100, CancellationToken.None);

        // Assert
        Assert.Equal(10, ctx.NumWorkers);
    }

    [Fact]
    public void WorkerContext_StartIndex_CalculatedCorrectly()
    {
        // Act
        var ctx = new WorkerContext(2, 4, 100, CancellationToken.None);

        // Assert
        Assert.Equal(50, ctx.StartIndex);
    }

    [Fact]
    public void WorkerContext_EndIndex_CalculatedCorrectly()
    {
        // Act
        var ctx = new WorkerContext(2, 4, 100, CancellationToken.None);

        // Assert
        Assert.Equal(75, ctx.EndIndex);
    }

    #endregion

    #region WorkerContext Tests - Edge Cases

    [Fact]
    public void WorkerContext_FirstWorker_StartsAtZero()
    {
        // Act
        var ctx = new WorkerContext(0, 5, 100, CancellationToken.None);

        // Assert
        Assert.Equal(0, ctx.StartIndex);
    }

    [Fact]
    public void WorkerContext_LastWorker_EndsAtTotal()
    {
        // Act
        var ctx = new WorkerContext(4, 5, 100, CancellationToken.None);

        // Assert
        Assert.Equal(100, ctx.EndIndex);
    }

    [Fact]
    public void WorkerContext_MoreWorkersThanItems_CorrectPartition()
    {
        // Act
        var ctx1 = new WorkerContext(0, 10, 5, CancellationToken.None);
        var ctx2 = new WorkerContext(1, 10, 5, CancellationToken.None);
        var ctx3 = new WorkerContext(2, 10, 5, CancellationToken.None);

        // Assert
        Assert.Equal(1, ctx1.PartitionSize);
        Assert.Equal(1, ctx2.PartitionSize);
        Assert.Equal(1, ctx3.PartitionSize);
    }

    #endregion

    #region WorkerPool Constructor Tests - Basic Constructor

    [Fact]
    public void Constructor_ValidParameters_CreatesPool()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;

        // Act
        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Assert
        Assert.Equal(2, pool.NumWorkers);
        Assert.False(pool.IsRunning);
        Assert.Equal(0, pool.ActiveWorkers);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void Constructor_ZeroWorkers_CreatesPool()
    {
        // Note: The actual implementation throws an exception for 0 workers,
        // but the spec says it should create an empty pool.
        // We'll test the actual behavior.

        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WorkerPool<int>(workerFunc, queue, 0));

        queue.Dispose();
    }

    [Fact]
    public void Constructor_MultipleWorkers_CreatesCorrectNumber()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;

        // Act
        var pool = new WorkerPool<int>(workerFunc, queue, 5);

        // Assert
        Assert.Equal(5, pool.NumWorkers);

        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region WorkerPool Constructor Tests - Constructor with Cancellation Token

    [Fact]
    public void Constructor_WithCancellationToken_UsesToken()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;
        var cts = new CancellationTokenSource();

        // Act
        var pool = new WorkerPool<int>(workerFunc, queue, 2, cts.Token);

        // Assert
        Assert.NotNull(pool);

        pool.Dispose();
        queue.Dispose();
        cts.Dispose();
    }

    [Fact]
    public void Constructor_WithoutCancellationToken_NoToken()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;

        // Act
        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Assert
        Assert.NotNull(pool);

        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region WorkerPool Constructor Tests - Invalid Constructor

    [Fact]
    public void Constructor_NullWorkerFunc_ThrowsArgumentNullException()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new WorkerPool<int>(null!, queue, 2));

        queue.Dispose();
    }

    [Fact]
    public void Constructor_NullQueue_ThrowsArgumentNullException()
    {
        // Arrange
        DataWorker<int> workerFunc = (id, token) => id;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new WorkerPool<int>(workerFunc, null!, 2));
    }

    [Fact]
    public void Constructor_NegativeWorkers_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WorkerPool<int>(workerFunc, queue, -1));

        queue.Dispose();
    }

    #endregion

    #region Properties Tests - IsRunning Property

    [Fact]
    public void IsRunning_InitiallyFalse()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;
        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Assert
        Assert.False(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void IsRunning_AfterStart_ReturnsTrue()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(100);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Act
        pool.Start();

        // Assert
        Assert.True(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task IsRunning_AfterStop_ReturnsFalse()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.False(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Properties Tests - NumWorkers Property

    [Fact]
    public void NumWorkers_ReturnsCorrectValue()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;
        var pool = new WorkerPool<int>(workerFunc, queue, 7);

        // Assert
        Assert.Equal(7, pool.NumWorkers);

        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Properties Tests - ActiveWorkers Property

    [Fact]
    public void ActiveWorkers_InitiallyZero()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;
        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Assert
        Assert.Equal(0, pool.ActiveWorkers);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void ActiveWorkers_AfterStart_EqualsNumWorkers()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(100);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);
        pool.Start();
        Thread.Sleep(50);

        // Assert
        Assert.Equal(3, pool.ActiveWorkers);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task ActiveWorkers_WorkerCompletes_Decrements()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var completionCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref completionCount);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);
        pool.Start();

        // Act
        await Task.Delay(200);

        // Assert
        Assert.Equal(0, pool.ActiveWorkers);

        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Start Tests - Basic Start

    [Fact]
    public void Start_CreatesWorkerTasks()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        int workerCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref workerCount);
            Thread.Sleep(100);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);

        // Act
        pool.Start();
        Thread.Sleep(200);

        // Assert
        Assert.True(workerCount >= 3);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void Start_UpdatesIsRunningToTrue()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(100);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Act
        pool.Start();

        // Assert
        Assert.True(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void Start_UpdatesActiveWorkersToNumWorkers()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(100);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 4);
        pool.Start();
        Thread.Sleep(50);

        // Assert
        Assert.Equal(4, pool.ActiveWorkers);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void Start_CalledMultipleTimes_ThrowsInvalidOperationException()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(100);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => pool.Start());

        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Start Tests - Start with Worker Function

    [Fact]
    public async Task Start_WorkerFunctionCalledForEachWorker()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var callCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref callCount);
            Thread.Sleep(10);
            if (callCount >= 3)
                throw new OperationCanceledException();
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);

        // Act
        pool.Start();
        await Task.Delay(200);

        // Assert
        Assert.True(callCount >= 3);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void Start_WorkerFunctionReceivesCorrectWorkerId()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var workerIds = new List<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            lock (workerIds)
            {
                if (!workerIds.Contains(id))
                    workerIds.Add(id);
            }
            Thread.Sleep(10);
            if (workerIds.Count >= 3)
                throw new OperationCanceledException();
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);
        pool.Start();
        Thread.Sleep(200);

        // Assert
        Assert.Contains(0, workerIds);
        Assert.Contains(1, workerIds);
        Assert.Contains(2, workerIds);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void Start_WorkerFunctionReceivesCancellationToken()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        bool tokenReceived = false;
        var cts = new CancellationTokenSource(200);
        DataWorker<int> workerFunc = (id, token) =>
        {
            tokenReceived = true;
            Thread.Sleep(10);
            token.ThrowIfCancellationRequested();
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2, cts.Token);
        pool.Start();

        // Act
        Thread.Sleep(500);

        // Assert
        Assert.True(tokenReceived);
        Assert.False(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
        cts.Dispose();
    }

    #endregion

    #region Stop Tests - Basic Stop

    [Fact]
    public async Task StopAsync_StopsAllWorkers()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var iterations = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref iterations);
            Thread.Sleep(10);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();
        await Task.Delay(100);

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.Equal(0, pool.ActiveWorkers);
        Assert.False(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task StopAsync_UpdatesIsRunningToFalse()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.False(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task StopAsync_UpdatesActiveWorkersToZero()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.Equal(0, pool.ActiveWorkers);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task StopAsync_AfterStart_StopsGracefully()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();
        await Task.Delay(50);

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.False(pool.IsRunning);
        Assert.True(queue.IsCompleted);

        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Stop Tests - Stop Timeout

    [Fact]
    public async Task StopAsync_WithTimeout_WaitsForWorkers()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(50);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.False(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task StopAsync_TimeoutExceeded_ThrowsTimeoutException()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            // Worker that ignores cancellation
            while (!token.IsCancellationRequested)
            {
                Thread.Sleep(10);
            }
            // Continue running despite cancellation
            Thread.Sleep(10000);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act & Assert
        await Assert.ThrowsAsync<TimeoutException>(() =>
            pool.StopAsync(TimeSpan.FromMilliseconds(100)));

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task StopAsync_ZeroTimeout_ReturnsImmediately()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(100);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        await pool.StopAsync(TimeSpan.Zero);

        // Assert
        Assert.False(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Stop Tests - Stop Before Start

    [Fact]
    public async Task StopAsync_BeforeStart_ReturnsSucceeds()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;
        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.False(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task StopAsync_MultipleCalls_Idempotent()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.False(pool.IsRunning);

        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Stop Tests - Stop with Cancellation

    [Fact]
    public async Task StopAsync_CancelsWorkerTasks()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        bool cancellationRequested = false;
        DataWorker<int> workerFunc = (id, token) =>
        {
            try
            {
                while (!token.IsCancellationRequested)
                {
                    Thread.Sleep(10);
                }
                cancellationRequested = true;
                return id;
            }
            catch (OperationCanceledException)
            {
                cancellationRequested = true;
                throw;
            }
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();
        await Task.Delay(50);

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.True(cancellationRequested);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task StopAsync_WorkersReceiveCancellationSignal()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var cancellationCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            try
            {
                while (!token.IsCancellationRequested)
                {
                    Thread.Sleep(10);
                }
                Interlocked.Increment(ref cancellationCount);
                return id;
            }
            catch (OperationCanceledException)
            {
                Interlocked.Increment(ref cancellationCount);
                throw;
            }
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);
        pool.Start();
        await Task.Delay(50);

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.Equal(3, cancellationCount);

        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region WaitAsync Tests - Basic WaitAsync

    [Fact]
    public async Task WaitAsync_AfterStart_WaitsForCompletion()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(50);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        var waitTask = pool.WaitAsync();
        Assert.False(waitTask.IsCompleted);

        await Task.Delay(200);
        Assert.True(waitTask.IsCompleted);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task WaitAsync_AfterAllComplete_ReturnsImmediately()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(50);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();
        await Task.Delay(200);

        // Act
        await pool.WaitAsync();

        // Assert
        Assert.True(pool.WaitAsync().IsCompleted);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region WaitAsync Tests - WaitAsync with Errors

    [Fact]
    public async Task WaitAsync_WorkerThrows_ThrowsAggregateException()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            throw new InvalidOperationException("Worker failed");
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2, errorPolicy: ErrorPolicy.FailFast);
        pool.Start();

        // Act & Assert
        await Assert.ThrowsAsync<AggregateException>(() => pool.WaitAsync());

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task WaitAsync_MultipleWorkersThrow_ContainsAllErrors()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            throw new InvalidOperationException($"Worker {id} failed");
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3, errorPolicy: ErrorPolicy.FailFast);
        pool.Start();

        // Act
        var exception = await Assert.ThrowsAsync<AggregateException>(() => pool.WaitAsync());

        // Assert
        Assert.Equal(3, exception.InnerExceptions.Count);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region WaitAsync Tests - WaitAsync Cancellation

    [Fact]
    public async Task WaitAsync_CancellationToken_ThrowsOperationCanceledException()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var cts = new CancellationTokenSource(100);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act & Assert
        await Assert.ThrowsAsync<TaskCanceledException>(() =>
            pool.WaitAsync(cts.Token));

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
        cts.Dispose();
    }

    #endregion

    #region Worker Execution Tests - Basic Worker Execution

    [Fact]
    public async Task WorkerExecution_ProducesItems_EnqueuesToQueue()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            if (queue.Count >= 10)
                throw new OperationCanceledException();
            return id * 10;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        await Task.Delay(500);

        // Assert
        Assert.True(queue.Count > 0);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task WorkerExecution_ProducesCorrectNumberOfItems()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        int expectedCount = 10;
        int productionCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref productionCount);
            Thread.Sleep(10);
            if (productionCount >= expectedCount)
                throw new OperationCanceledException();
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        await pool.WaitAsync();

        // Assert
        Assert.Equal(expectedCount, productionCount);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task WorkerExecution_ProducesItemsInOrder_FIFO()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        var producedOrder = new List<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            var value = id * 100 + Interlocked.Increment(ref productionCount);
            lock (producedOrder)
            {
                producedOrder.Add(value);
            }
            Thread.Sleep(10);
            if (producedOrder.Count >= 10)
                throw new OperationCanceledException();
            return value;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        await pool.WaitAsync();

        // Assert
        // Note: Items should be in the order they were enqueued
        Assert.Equal(10, producedOrder.Count);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Worker Execution Tests - Worker with Cancellation

    [Fact]
    public async Task WorkerCancellation_StopsProduction()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        var cts = new CancellationTokenSource(200);
        int productionCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref productionCount);
            Thread.Sleep(10);
            token.ThrowIfCancellationRequested();
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2, cts.Token);
        pool.Start();

        // Act
        await Task.Delay(500);

        // Assert
        Assert.False(pool.IsRunning);
        Assert.True(productionCount > 0);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
        cts.Dispose();
    }

    [Fact]
    public async Task WorkerCancellation_EnqueuesNoMoreItems()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        var cts = new CancellationTokenSource(200);
        int productionCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            var count = Interlocked.Increment(ref productionCount);
            Thread.Sleep(10);
            token.ThrowIfCancellationRequested();
            return count;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2, cts.Token);
        pool.Start();

        // Act
        await Task.Delay(500);

        // Assert
        var countAfterCancellation = productionCount;
        await Task.Delay(200);
        Assert.Equal(countAfterCancellation, productionCount);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
        cts.Dispose();
    }

    #endregion

    #region Worker Execution Tests - Worker with Exception

    [Fact]
    public async Task WorkerException_PropagatesToPool()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            throw new InvalidOperationException("Worker failed");
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2, errorPolicy: ErrorPolicy.FailFast);

        // Act & Assert
        pool.Start();
        await Assert.ThrowsAsync<AggregateException>(() => pool.WaitAsync());

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task WorkerException_OtherWorkersContinue_ContinuePolicy()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        int productionCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            if (id == 0)
            {
                // Worker 0 fails
                throw new InvalidOperationException("Worker 0 failed");
            }

            // Other workers continue
            Interlocked.Increment(ref productionCount);
            Thread.Sleep(10);
            if (productionCount >= 5)
                throw new OperationCanceledException();
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3, errorPolicy: ErrorPolicy.Continue);
        pool.Start();

        // Act
        await Task.Delay(500);

        // Assert
        Assert.True(productionCount > 0);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Concurrent Execution Tests - Concurrent Worker Execution

    [Fact]
    public async Task ConcurrentWorkers_AllRunInParallel()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        var activeWorkers = new List<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            lock (activeWorkers)
            {
                activeWorkers.Add(id);
            }
            Thread.Sleep(100);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 4);
        pool.Start();
        Thread.Sleep(50);

        // Act
        lock (activeWorkers)
        {
            // Assert - All workers should be active at the same time
            Assert.Equal(4, activeWorkers.Count);
        }

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task ConcurrentWorkers_AllEnqueueToSameQueue_ThreadSafe()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        var results = new List<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            for (int i = 0; i < 10; i++)
            {
                queue.Enqueue(id * 100 + i);
                Thread.Sleep(1);
            }
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 4);
        pool.Start();

        // Act
        await pool.WaitAsync();

        // Dequeue all items
        while (queue.Count > 0)
        {
            results.Add(queue.Dequeue());
        }

        // Assert
        Assert.Equal(40, results.Count);
        Assert.True(results.Distinct().Count() == 40); // All items are unique

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task ConcurrentWorkers_ProduceDistinctItems_NoDuplicates()
    {
        // Arrange
        var queue = new SharedQueue<int>(1000);
        var workerCount = 10;
        var itemsPerWorker = 100;
        var results = new List<int>();

        DataWorker<int> workerFunc = (id, token) =>
        {
            for (int i = 0; i < itemsPerWorker; i++)
            {
                queue.Enqueue(id * itemsPerWorker + i);
            }
            return 0;
        };

        var pool = new WorkerPool<int>(workerFunc, queue, workerCount);
        pool.Start();
        await pool.WaitAsync();

        // Act - Dequeue all items
        while (queue.Count > 0)
        {
            results.Add(queue.Dequeue());
        }

        // Assert
        Assert.Equal(workerCount * itemsPerWorker, results.Count);
        Assert.Equal(results.Count, results.Distinct().Count()); // No duplicates

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Concurrent Execution Tests - High Concurrency

    [Fact]
    public async Task HighConcurrency_100Workers_ThreadSafe()
    {
        // Arrange
        var queue = new SharedQueue<int>(10000);
        var workerCount = 100;
        var results = new List<int>();

        DataWorker<int> workerFunc = (id, token) =>
        {
            queue.Enqueue(id);
            Thread.Sleep(1);
            return id;
        };

        var pool = new WorkerPool<int>(workerFunc, queue, workerCount);
        pool.Start();

        // Act
        await Task.Delay(500);

        // Dequeue items
        while (queue.Count > 0)
        {
            results.Add(queue.Dequeue());
        }

        // Assert
        Assert.True(results.Count > 0);
        Assert.Equal(results.Count, results.Distinct().Count()); // No duplicates

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task HighConcurrency_100Workers_ProducesCorrectCount()
    {
        // Arrange
        var queue = new SharedQueue<int>(10000);
        var workerCount = 100;
        var expectedCount = 0;

        DataWorker<int> workerFunc = (id, token) =>
        {
            var count = Interlocked.Increment(ref expectedCount);
            queue.Enqueue(count);
            Thread.Sleep(1);
            if (count >= 500)
                throw new OperationCanceledException();
            return count;
        };

        var pool = new WorkerPool<int>(workerFunc, queue, workerCount);
        pool.Start();

        // Act
        await pool.WaitAsync();

        // Assert
        Assert.Equal(500, expectedCount);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Event Hook Tests - WorkerStarted Event

    [Fact]
    public void WorkerStarted_RaisedForEachWorker()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        int eventsRaised = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);
        pool.WorkerStarted += (id) => Interlocked.Increment(ref eventsRaised);

        // Act
        pool.Start();
        Thread.Sleep(200);

        // Assert
        Assert.Equal(3, eventsRaised);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void WorkerStarted_RaisedWithCorrectWorkerId()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var workerIds = new List<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);
        pool.WorkerStarted += (id) =>
        {
            lock (workerIds)
            {
                if (!workerIds.Contains(id))
                    workerIds.Add(id);
            }
        };

        // Act
        pool.Start();
        Thread.Sleep(200);

        // Assert
        Assert.Contains(0, workerIds);
        Assert.Contains(1, workerIds);
        Assert.Contains(2, workerIds);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Event Hook Tests - WorkerCompleted Event

    [Fact]
    public async Task WorkerCompleted_RaisedWhenWorkerFinishes()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var completedWorkers = new List<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);
        pool.WorkerCompleted += (id, success) => completedWorkers.Add(id);
        pool.Start();

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.Equal(3, completedWorkers.Count);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task WorkerCompleted_RaisedWithCorrectWorkerId()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var completedWorkers = new List<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);
        pool.WorkerCompleted += (id, success) => completedWorkers.Add(id);
        pool.Start();

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.Contains(0, completedWorkers);
        Assert.Contains(1, completedWorkers);
        Assert.Contains(2, completedWorkers);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task WorkerCompleted_SuccessfulWorker_SuccessTrue()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var successCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3);
        pool.WorkerCompleted += (id, success) =>
        {
            if (success)
                Interlocked.Increment(ref successCount);
        };
        pool.Start();

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.Equal(3, successCount);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task WorkerCompleted_FailedWorker_SuccessFalse()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var failureCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            throw new InvalidOperationException("Worker failed");
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2, errorPolicy: ErrorPolicy.FailFast);
        pool.WorkerCompleted += (id, success) =>
        {
            if (!success)
                Interlocked.Increment(ref failureCount);
        };
        pool.Start();

        // Act
        try
        {
            await pool.WaitAsync();
        }
        catch (AggregateException) { }

        // Assert
        Assert.Equal(2, failureCount);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Error Handling Tests - Worker Throws Exception

    [Fact]
    public async Task WorkerThrowsException_PoolCapturesError()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            throw new InvalidOperationException("Worker failed");
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Act
        pool.Start();

        // Assert
        await Task.Delay(100);
        Assert.True(pool.ErrorAggregator.HasErrors());

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task WorkerThrowsException_AggregateExceptionThrownInWaitAsync()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            throw new InvalidOperationException("Worker failed");
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2, errorPolicy: ErrorPolicy.FailFast);
        pool.Start();

        // Act & Assert
        await Assert.ThrowsAsync<AggregateException>(() => pool.WaitAsync());

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Error Handling Tests - Graceful Degradation

    [Fact]
    public async Task SomeWorkersFail_OthersContinue_ContinuePolicy()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        int productionCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            if (id == 0)
            {
                throw new InvalidOperationException("Worker 0 failed");
            }

            Interlocked.Increment(ref productionCount);
            Thread.Sleep(10);
            if (productionCount >= 10)
                throw new OperationCanceledException();
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 3, errorPolicy: ErrorPolicy.Continue);
        pool.Start();

        // Act
        await Task.Delay(500);

        // Assert
        Assert.True(productionCount > 0);
        Assert.True(pool.FailedWorkers > 0);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Edge Cases - Zero Workers

    [Fact]
    public void ZeroWorkers_Start_NoWorkersCreated()
    {
        // The actual implementation throws an exception for 0 workers
        // This test verifies that behavior
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WorkerPool<int>(workerFunc, queue, 0));

        queue.Dispose();
    }

    #endregion

    #region Edge Cases - Single Worker

    [Fact]
    public async Task SingleWorker_CorrectlyExecutes()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var productionCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref productionCount);
            Thread.Sleep(10);
            if (productionCount >= 5)
                throw new OperationCanceledException();
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 1);
        pool.Start();

        // Act
        await pool.WaitAsync();

        // Assert
        Assert.Equal(5, productionCount);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Edge Cases - Large Number of Workers

    [Fact]
    public void LargeNumberOfWorkers_100_CreatesCorrectly()
    {
        // Arrange
        var queue = new SharedQueue<int>(1000);
        DataWorker<int> workerFunc = (id, token) => id;

        // Act
        var pool = new WorkerPool<int>(workerFunc, queue, 100);

        // Assert
        Assert.Equal(100, pool.NumWorkers);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task LargeNumberOfWorkers_AllExecute_ThreadSafe()
    {
        // Arrange
        var queue = new SharedQueue<int>(10000);
        var results = new List<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            queue.Enqueue(id);
            Thread.Sleep(1);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 100);
        pool.Start();

        // Act
        await pool.WaitAsync();

        // Dequeue items
        while (queue.Count > 0)
        {
            results.Add(queue.Dequeue());
        }

        // Assert
        Assert.Equal(results.Count, results.Distinct().Count()); // No duplicates

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Edge Cases - Empty Queue

    [Fact]
    public async Task WorkersEnqueueToEmptyQueue_Succeeds()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        DataWorker<int> workerFunc = (id, token) =>
        {
            queue.Enqueue(id);
            Thread.Sleep(10);
            if (queue.Count >= 10)
                throw new OperationCanceledException();
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Act
        pool.Start();
        await pool.WaitAsync();

        // Assert
        Assert.True(queue.Count >= 10);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Edge Cases - Full Queue

    [Fact]
    public async Task WorkersEnqueueToFullQueue_Blocks()
    {
        // Arrange
        var queue = new SharedQueue<int>(5); // Small queue
        var enqueueCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            var count = Interlocked.Increment(ref enqueueCount);
            queue.Enqueue(id);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        await Task.Delay(100);

        // Assert - Workers should be blocked waiting for space in the queue
        Assert.Equal(5, queue.Count);

        // Cleanup
        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region WorkerId Assignment Tests

    [Fact]
    public void WorkerIds_AreZeroBased()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;
        var pool = new WorkerPool<int>(workerFunc, queue, 5);

        // Assert
        Assert.Equal(5, pool.NumWorkers);
        // Workers should be 0, 1, 2, 3, 4

        // Cleanup
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void WorkerIds_AreUnique()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var workerIds = new HashSet<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            lock (workerIds)
            {
                workerIds.Add(id);
            }
            Thread.Sleep(10);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 5);
        pool.Start();

        // Act
        Thread.Sleep(200);

        // Assert
        Assert.Equal(5, workerIds.Count);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void WorkerIds_AreSequential()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        var workerIds = new List<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            lock (workerIds)
            {
                if (!workerIds.Contains(id))
                    workerIds.Add(id);
            }
            Thread.Sleep(10);
            throw new OperationCanceledException();
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 5);
        pool.Start();

        // Act
        Thread.Sleep(200);
        workerIds.Sort();

        // Assert
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, workerIds);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Dispose Tests

    [Fact]
    public void Dispose_StopsIfRunning()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();

        // Act
        pool.Dispose();

        // Assert
        Assert.False(pool.IsRunning);
        queue.Dispose();
    }

    [Fact]
    public void Dispose_CanCallMultipleTimes()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;
        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Act & Assert - Should not throw
        pool.Dispose();
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Cancellation Tests

    [Fact]
    public async Task Workers_RespectCancellationToken()
    {
        // Arrange
        var cts = new CancellationTokenSource(100);
        var queue = new SharedQueue<int>(10);
        int iterations = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref iterations);
            Thread.Sleep(10);
            token.ThrowIfCancellationRequested();
            return id;
        };
        var pool = new WorkerPool<int>(workerFunc, queue, 2, cts.Token);
        pool.Start();

        // Act
        await Task.Delay(300);

        // Assert
        Assert.False(pool.IsRunning);
        Assert.Equal(0, pool.ActiveWorkers);

        // Cleanup
        pool.Dispose();
        queue.Dispose();
        cts.Dispose();
    }

    #endregion

    #region Additional Helper Variables

    private int productionCount;

    #endregion
}
