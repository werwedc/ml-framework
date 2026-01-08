using MLFramework.Data;
using Xunit;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for WorkerPool functionality.
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

    #region Constructor Tests

    [Fact]
    public void Constructor_ValidParameters_CreatesWorkerPool()
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
    public void Constructor_ZeroNumWorkers_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        DataWorker<int> workerFunc = (id, token) => id;

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WorkerPool<int>(workerFunc, queue, 0));

        queue.Dispose();
    }

    [Fact]
    public void Constructor_NegativeNumWorkers_ThrowsArgumentOutOfRangeException()
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

    #region Start Tests

    [Fact]
    public void Start_LaunchesCorrectNumberOfWorkers()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        int workerCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref workerCount);
            Thread.Sleep(100); // Give workers time to start
            throw new OperationCanceledException(); // Stop after one iteration
        };

        var pool = new WorkerPool<int>(workerFunc, queue, 3);

        // Act
        pool.Start();
        Thread.Sleep(200); // Wait for workers to start

        // Assert
        Assert.True(pool.IsRunning);
        Assert.Equal(3, pool.ActiveWorkers);

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public void Start_WhenAlreadyRunning_ThrowsInvalidOperationException()
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

    #region Worker Production Tests

    [Fact]
    public async Task Workers_ProduceBatchesContinuously()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        int productionCount = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref productionCount);
            Thread.Sleep(10);
            if (productionCount >= 10)
                throw new OperationCanceledException();
            return id;
        };

        var pool = new WorkerPool<int>(workerFunc, queue, 2);

        // Act
        pool.Start();
        await Task.Delay(500);

        // Assert
        Assert.True(productionCount >= 10);

        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task MultipleWorkers_AllEnqueueToQueue_ThreadSafe()
    {
        // Arrange
        var queue = new SharedQueue<int>(100);
        var results = new List<int>();
        DataWorker<int> workerFunc = (id, token) =>
        {
            Thread.Sleep(10);
            if (Interlocked.Add(ref results.Count, 0) >= 20)
                throw new OperationCanceledException();
            return id;
        };

        var pool = new WorkerPool<int>(workerFunc, queue, 4);

        // Act
        pool.Start();

        // Dequeue items
        var cts = new CancellationTokenSource(1000);
        try
        {
            while (!cts.Token.IsCancellationRequested && queue.Count < 20)
            {
                results.Add(queue.Dequeue());
            }
        }
        catch (OperationCanceledException) { }

        // Assert
        Assert.True(results.Count > 0);

        await pool.StopAsync(TimeSpan.FromSeconds(5));
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region Stop Tests

    [Fact]
    public async Task StopAsync_StopsWorkersGracefully()
    {
        // Arrange
        var queue = new SharedQueue<int>(10);
        int iterations = 0;
        DataWorker<int> workerFunc = (id, token) =>
        {
            Interlocked.Increment(ref iterations);
            Thread.Sleep(10);
            return id;
        };

        var pool = new WorkerPool<int>(workerFunc, queue, 2);
        pool.Start();
        await Task.Delay(200);

        // Act
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.False(pool.IsRunning);
        Assert.Equal(0, pool.ActiveWorkers);
        Assert.True(queue.IsCompleted);

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

    [Fact]
    public async Task StopAsync_BeforeStart_NoEffect()
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

    #endregion

    #region Worker Events Tests

    [Fact]
    public void WorkerStarted_RaisedWhenWorkerStarts()
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

        pool.Dispose();
        queue.Dispose();
    }

    [Fact]
    public async Task WorkerCompleted_RaisedWhenWorkerStops()
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

        // Act
        pool.Start();
        await pool.StopAsync(TimeSpan.FromSeconds(5));

        // Assert
        Assert.Equal(3, completedWorkers.Count);
        Assert.Contains(0, completedWorkers);
        Assert.Contains(1, completedWorkers);
        Assert.Contains(2, completedWorkers);

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

        // Act
        pool.Start();
        await Task.Delay(300);

        // Assert
        Assert.False(pool.IsRunning);
        Assert.Equal(0, pool.ActiveWorkers);

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

        // Act & Assert (should not throw)
        pool.Dispose();
        pool.Dispose();
        queue.Dispose();
    }

    #endregion

    #region WorkerContext Tests

    [Fact]
    public void WorkerContext_CorrectPartitioning_EvenDivision()
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
    public void WorkerContext_CorrectPartitioning_UnevenDivision()
    {
        // Arrange & Act
        var ctx1 = new WorkerContext(0, 3, 10, CancellationToken.None);
        var ctx2 = new WorkerContext(1, 3, 10, CancellationToken.None);
        var ctx3 = new WorkerContext(2, 3, 10, CancellationToken.None);

        // Assert
        Assert.Equal(0, ctx1.StartIndex);
        Assert.Equal(4, ctx1.EndIndex); // 4 items
        Assert.Equal(4, ctx2.StartIndex);
        Assert.Equal(7, ctx2.EndIndex); // 3 items
        Assert.Equal(7, ctx3.StartIndex);
        Assert.Equal(10, ctx3.EndIndex); // 3 items
    }

    [Fact]
    public void WorkerContext_InvalidWorkerId_ThrowsArgumentOutOfRangeException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WorkerContext(-1, 2, 10, CancellationToken.None));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WorkerContext(2, 2, 10, CancellationToken.None));
    }

    [Fact]
    public void WorkerContext_ContainsIndex_ReturnsCorrect()
    {
        // Arrange
        var ctx = new WorkerContext(0, 2, 10, CancellationToken.None);

        // Act & Assert
        Assert.True(ctx.ContainsIndex(0));
        Assert.True(ctx.ContainsIndex(4));
        Assert.False(ctx.ContainsIndex(5));
        Assert.False(ctx.ContainsIndex(9));
    }

    #endregion

    #region WaitAsync Tests

    [Fact]
    public async Task WaitAsync_WaitsForAllWorkers()
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

    #endregion
}
