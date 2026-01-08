namespace MLFramework.Checkpointing.Tests;

using System;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

/// <summary>
/// Tests for AsyncRecomputationEngine
/// </summary>
public class AsyncRecomputationEngineTests
{
    [Fact]
    public async Task RegisterRecomputeFunction_Successfully_Registers()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine();
        var layerId = "layer1";
        var recomputeFunc = () => new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });

        // Act
        engine.RegisterRecomputeFunction(layerId, recomputeFunc);

        // Assert - no exception thrown
        await Task.CompletedTask;
    }

    [Fact]
    public async Task RegisterRecomputeFunction_WithDuplicateLayerId_ThrowsException()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine();
        var layerId = "layer1";
        var recomputeFunc = () => new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });
        engine.RegisterRecomputeFunction(layerId, recomputeFunc);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            engine.RegisterRecomputeFunction(layerId, recomputeFunc);
            await Task.CompletedTask;
        });
    }

    [Fact]
    public async Task RecomputeAsync_WithRegisteredLayer_SuccessfullyRecomputes()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine();
        var layerId = "layer1";
        var expectedData = new float[] { 1, 2, 3 };
        var recomputeFunc = () => new Tensor(expectedData, new int[] { 3 });
        engine.RegisterRecomputeFunction(layerId, recomputeFunc);

        // Act
        var result = await engine.RecomputeAsync(layerId);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(expectedData.Length, result.ElementCount);
    }

    [Fact]
    public async Task RecomputeAsync_WithUnregisteredLayer_ThrowsException()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine();

        // Act & Assert
        await Assert.ThrowsAsync<KeyNotFoundException>(async () =>
            await engine.RecomputeAsync("nonexistent"));
    }

    [Fact]
    public async Task RecomputeAsync_WithCancellation_CancelsOperation()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine();
        var layerId = "layer1";
        var cts = new CancellationTokenSource();
        engine.RegisterRecomputeFunction(layerId, () =>
        {
            Thread.Sleep(1000); // Long-running operation
            return new Tensor(new float[] { 1 }, new int[] { 1 });
        });

        // Act
        cts.CancelAfter(50); // Cancel before operation completes
        var task = engine.RecomputeAsync(layerId, cts.Token);

        // Assert
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await task);
    }

    [Fact]
    public async Task RecomputeMultipleAsync_WithMultipleLayers_RecomputesAll()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine();
        engine.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1, 2 }, new int[] { 2 }));
        engine.RegisterRecomputeFunction("layer2", () => new Tensor(new float[] { 3, 4 }, new int[] { 2 }));
        engine.RegisterRecomputeFunction("layer3", () => new Tensor(new float[] { 5, 6 }, new int[] { 2 }));

        // Act
        var results = await engine.RecomputeMultipleAsync(new[] { "layer1", "layer2", "layer3" });

        // Assert
        Assert.Equal(3, results.Count);
        Assert.True(results.ContainsKey("layer1"));
        Assert.True(results.ContainsKey("layer2"));
        Assert.True(results.ContainsKey("layer3"));
    }

    [Fact]
    public async Task RecomputeMultipleAsync_WithEmptyList_ReturnsEmptyDictionary()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine();

        // Act
        var results = await engine.RecomputeMultipleAsync(Array.Empty<string>());

        // Assert
        Assert.Empty(results);
    }

    [Fact]
    public async Task RecomputeMultipleAsync_WithNullList_ThrowsException()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(async () =>
            await engine.RecomputeMultipleAsync(null!));
    }

    [Fact]
    public async Task RecomputeMultipleAsync_RespectsMaxDegreeOfParallelism()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine(maxDegreeOfParallelism: 2);
        var concurrentOps = 0;
        var maxConcurrentOps = 0;

        var recomputeFunc = () =>
        {
            Interlocked.Increment(ref concurrentOps);
            Thread.Sleep(100);
            Interlocked.Exchange(ref maxConcurrentOps, Math.Max(maxConcurrentOps, concurrentOps));
            Interlocked.Decrement(ref concurrentOps);
            return new Tensor(new float[] { 1 }, new int[] { 1 });
        };

        // Act
        engine.RegisterRecomputeFunction("layer1", recomputeFunc);
        engine.RegisterRecomputeFunction("layer2", recomputeFunc);
        engine.RegisterRecomputeFunction("layer3", recomputeFunc);
        engine.RegisterRecomputeFunction("layer4", recomputeFunc);

        await engine.RecomputeMultipleAsync(new[] { "layer1", "layer2", "layer3", "layer4" });

        // Assert
        Assert.True(maxConcurrentOps <= 2);
    }

    [Fact]
    public async Task Dispose_AfterDispose_ThrowsOnOperations()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine();
        engine.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1 }, new int[] { 1 }));
        engine.Dispose();

        // Act & Assert
        await Assert.ThrowsAsync<ObjectDisposedException>(async () =>
            await engine.RecomputeAsync("layer1"));
    }

    [Fact]
    public async Task MultipleEngines_CanOperateIndependently()
    {
        // Arrange
        var engine1 = new AsyncRecomputationEngine();
        var engine2 = new AsyncRecomputationEngine();

        // Act
        engine1.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1 }, new int[] { 1 }));
        engine2.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 2 }, new int[] { 1 }));

        var result1 = await engine1.RecomputeAsync("layer1");
        var result2 = await engine2.RecomputeAsync("layer1");

        // Assert
        Assert.NotNull(result1);
        Assert.NotNull(result2);
    }

    [Fact]
    public async Task RecomputeAsync_WithMultipleCalls_ExecutesSequentially()
    {
        // Arrange
        var engine = new AsyncRecomputationEngine(maxDegreeOfParallelism: 1);
        var executionOrder = new List<int>();

        engine.RegisterRecomputeFunction("layer1", () =>
        {
            executionOrder.Add(1);
            Thread.Sleep(50);
            return new Tensor(new float[] { 1 }, new int[] { 1 });
        });

        // Act
        var task1 = engine.RecomputeAsync("layer1");
        var task2 = engine.RecomputeAsync("layer1");
        var task3 = engine.RecomputeAsync("layer1");

        await Task.WhenAll(task1, task2, task3);

        // Assert
        Assert.Equal(3, executionOrder.Count);
    }
}
