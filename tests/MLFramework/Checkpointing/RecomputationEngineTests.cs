namespace MLFramework.Checkpointing.Tests;

using System;
using System.Threading;
using Xunit;

/// <summary>
/// Tests for RecomputationEngine
/// </summary>
public class RecomputationEngineTests
{
    [Fact]
    public void RegisterRecomputeFunction_Successfully_Registers()
    {
        // Arrange
        var engine = new RecomputationEngine();
        var layerId = "layer1";
        var recomputeFunc = () => new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });

        // Act
        engine.RegisterRecomputeFunction(layerId, recomputeFunc);

        // Assert
        Assert.True(engine.HasRecomputeFunction(layerId));
    }

    [Fact]
    public void RegisterRecomputeFunction_WithDuplicateLayerId_ThrowsException()
    {
        // Arrange
        var engine = new RecomputationEngine();
        var layerId = "layer1";
        var recomputeFunc = () => new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });
        engine.RegisterRecomputeFunction(layerId, recomputeFunc);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            engine.RegisterRecomputeFunction(layerId, recomputeFunc));
    }

    [Fact]
    public void RegisterRecomputeFunction_WithNullLayerId_ThrowsException()
    {
        // Arrange
        var engine = new RecomputationEngine();
        var recomputeFunc = () => new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            engine.RegisterRecomputeFunction(null!, recomputeFunc));
    }

    [Fact]
    public void RegisterRecomputeFunction_WithNullFunction_ThrowsException()
    {
        // Arrange
        var engine = new RecomputationEngine();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            engine.RegisterRecomputeFunction("layer1", null!));
    }

    [Fact]
    public void Recompute_WithRegisteredLayer_SuccessfullyRecomputes()
    {
        // Arrange
        var engine = new RecomputationEngine();
        var layerId = "layer1";
        var expectedData = new float[] { 1, 2, 3 };
        var recomputeFunc = () => new Tensor(expectedData, new int[] { 3 });
        engine.RegisterRecomputeFunction(layerId, recomputeFunc);

        // Act
        var result = engine.Recompute(layerId);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(expectedData.Length, result.ElementCount);
    }

    [Fact]
    public void Recompute_WithUnregisteredLayer_ThrowsException()
    {
        // Arrange
        var engine = new RecomputationEngine();

        // Act & Assert
        Assert.Throws<KeyNotFoundException>(() =>
            engine.Recompute("nonexistent"));
    }

    [Fact]
    public void Recompute_WithThrowingFunction_ThrowsRecomputationException()
    {
        // Arrange
        var engine = new RecomputationEngine();
        var layerId = "layer1";
        var recomputeFunc = () => throw new InvalidOperationException("Test exception");
        engine.RegisterRecomputeFunction(layerId, recomputeFunc);

        // Act & Assert
        Assert.Throws<RecomputationException>(() =>
            engine.Recompute(layerId));
    }

    [Fact]
    public void RecomputeMultiple_WithMultipleLayers_RecomputesAll()
    {
        // Arrange
        var engine = new RecomputationEngine();
        engine.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1, 2 }, new int[] { 2 }));
        engine.RegisterRecomputeFunction("layer2", () => new Tensor(new float[] { 3, 4 }, new int[] { 2 }));
        engine.RegisterRecomputeFunction("layer3", () => new Tensor(new float[] { 5, 6 }, new int[] { 2 }));

        // Act
        var results = engine.RecomputeMultiple(new[] { "layer1", "layer2", "layer3" });

        // Assert
        Assert.Equal(3, results.Count);
        Assert.True(results.ContainsKey("layer1"));
        Assert.True(results.ContainsKey("layer2"));
        Assert.True(results.ContainsKey("layer3"));
    }

    [Fact]
    public void RecomputeMultiple_WithEmptyList_ReturnsEmptyDictionary()
    {
        // Arrange
        var engine = new RecomputationEngine();

        // Act
        var results = engine.RecomputeMultiple(Array.Empty<string>());

        // Assert
        Assert.Empty(results);
    }

    [Fact]
    public void RecomputeMultiple_WithNullList_ThrowsException()
    {
        // Arrange
        var engine = new RecomputationEngine();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            engine.RecomputeMultiple(null!));
    }

    [Fact]
    public void HasRecomputeFunction_WithRegisteredLayer_ReturnsTrue()
    {
        // Arrange
        var engine = new RecomputationEngine();
        engine.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1 }, new int[] { 1 }));

        // Act
        var result = engine.HasRecomputeFunction("layer1");

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void HasRecomputeFunction_WithUnregisteredLayer_ReturnsFalse()
    {
        // Arrange
        var engine = new RecomputationEngine();

        // Act
        var result = engine.HasRecomputeFunction("nonexistent");

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void Clear_RemovesAllFunctions()
    {
        // Arrange
        var engine = new RecomputationEngine();
        engine.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1 }, new int[] { 1 }));
        engine.RegisterRecomputeFunction("layer2", () => new Tensor(new float[] { 1 }, new int[] { 1 }));

        // Act
        engine.Clear();

        // Assert
        Assert.False(engine.HasRecomputeFunction("layer1"));
        Assert.False(engine.HasRecomputeFunction("layer2"));
    }

    [Fact]
    public void GetStats_AfterRecomputations_ReturnsCorrectStatistics()
    {
        // Arrange
        var engine = new RecomputationEngine();
        engine.RegisterRecomputeFunction("layer1", () =>
        {
            Thread.Sleep(10); // Simulate some computation time
            return new Tensor(new float[] { 1, 2 }, new int[] { 2 });
        });

        // Act
        engine.Recompute("layer1");
        engine.Recompute("layer1");
        var stats = engine.GetStats();

        // Assert
        Assert.Equal(2, stats.TotalRecomputations);
        Assert.True(stats.TotalRecomputationTimeMs > 0);
        Assert.Equal(1, stats.RegisteredLayerCount);
        Assert.True(stats.Timestamp > DateTime.MinValue);
    }

    [Fact]
    public void GetStats_WithNoRecomputations_ReturnsZero()
    {
        // Arrange
        var engine = new RecomputationEngine();
        engine.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1 }, new int[] { 1 }));

        // Act
        var stats = engine.GetStats();

        // Assert
        Assert.Equal(0, stats.TotalRecomputations);
        Assert.Equal(0, stats.TotalRecomputationTimeMs);
        Assert.Equal(0, stats.AverageRecomputationTimeMs);
    }

    [Fact]
    public void GetStats_IncludesLayerStats()
    {
        // Arrange
        var engine = new RecomputationEngine();
        engine.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1 }, new int[] { 1 }));
        engine.RegisterRecomputeFunction("layer2", () => new Tensor(new float[] { 2 }, new int[] { 1 }));

        // Act
        engine.Recompute("layer1");
        engine.Recompute("layer2");
        var stats = engine.GetStats();

        // Assert
        Assert.True(stats.LayerStats.ContainsKey("layer1"));
        Assert.True(stats.LayerStats.ContainsKey("layer2"));
        Assert.Equal(1, stats.LayerStats["layer1"].CallCount);
        Assert.Equal(1, stats.LayerStats["layer2"].CallCount);
    }

    [Fact]
    public void GetLayerStats_WithExistingLayer_ReturnsStats()
    {
        // Arrange
        var engine = new RecomputationEngine();
        var layerId = "layer1";
        engine.RegisterRecomputeFunction(layerId, () => new Tensor(new float[] { 1 }, new int[] { 1 }));

        // Act
        engine.Recompute(layerId);
        var stats = engine.GetLayerStats(layerId);

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(layerId, stats!.LayerId);
        Assert.Equal(1, stats.CallCount);
        Assert.True(stats.TotalComputationTimeMs >= 0);
    }

    [Fact]
    public void GetLayerStats_WithNonExistentLayer_ReturnsNull()
    {
        // Arrange
        var engine = new RecomputationEngine();

        // Act
        var stats = engine.GetLayerStats("nonexistent");

        // Assert
        Assert.Null(stats);
    }

    [Fact]
    public void RegisterRecomputeFunction_WithDependencies_StoresDependencies()
    {
        // Arrange
        var engine = new RecomputationEngine();
        var dependencies = new[] { "input_layer", "prev_layer" };

        // Act
        engine.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1 }, new int[] { 1 }), dependencies);

        // Assert
        var stats = engine.GetStats();
        Assert.True(stats.LayerStats.ContainsKey("layer1"));
    }

    [Fact]
    public void RecomputeMultiple_WithDependencies_RespectsTopologicalOrder()
    {
        // Arrange
        var engine = new RecomputationEngine();
        var executionOrder = new List<string>();

        engine.RegisterRecomputeFunction("input", () =>
        {
            executionOrder.Add("input");
            return new Tensor(new float[] { 1 }, new int[] { 1 });
        });

        engine.RegisterRecomputeFunction("layer1", () =>
        {
            executionOrder.Add("layer1");
            return new Tensor(new float[] { 2 }, new int[] { 1 });
        }, new[] { "input" });

        engine.RegisterRecomputeFunction("layer2", () =>
        {
            executionOrder.Add("layer2");
            return new Tensor(new float[] { 3 }, new int[] { 1 });
        }, new[] { "layer1" });

        // Act
        engine.RecomputeMultiple(new[] { "layer2", "layer1", "input" });

        // Assert
        Assert.Equal(3, executionOrder.Count);
        Assert.Equal("input", executionOrder[0]); // Dependencies first
    }

    [Fact]
    public void Dispose_AfterDispose_ThrowsOnOperations()
    {
        // Arrange
        var engine = new RecomputationEngine();
        engine.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1 }, new int[] { 1 }));
        engine.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() =>
            engine.RegisterRecomputeFunction("layer2", () => new Tensor(new float[] { 1 }, new int[] { 1 })));

        Assert.Throws<ObjectDisposedException>(() =>
            engine.Recompute("layer1"));

        Assert.Throws<ObjectDisposedException>(() =>
            engine.GetStats());
    }

    [Fact]
    public void AverageRecomputationTimeMs_CalculatesCorrectly()
    {
        // Arrange
        var engine = new RecomputationEngine();
        engine.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1 }, new int[] { 1 }));
        engine.RegisterRecomputeFunction("layer2", () => new Tensor(new float[] { 2 }, new int[] { 1 }));

        // Act
        engine.Recompute("layer1");
        engine.Recompute("layer2");
        var stats = engine.GetStats();

        // Assert
        Assert.Equal(stats.TotalRecomputationTimeMs / 2.0, stats.AverageRecomputationTimeMs);
    }

    [Fact]
    public void MultipleEngines_CanOperateIndependently()
    {
        // Arrange
        var engine1 = new RecomputationEngine();
        var engine2 = new RecomputationEngine();

        // Act
        engine1.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 1 }, new int[] { 1 }));
        engine2.RegisterRecomputeFunction("layer1", () => new Tensor(new float[] { 2 }, new int[] { 1 }));

        var result1 = engine1.Recompute("layer1");
        var result2 = engine2.Recompute("layer1");

        // Assert
        Assert.True(engine1.HasRecomputeFunction("layer1"));
        Assert.True(engine2.HasRecomputeFunction("layer1"));
    }
}
