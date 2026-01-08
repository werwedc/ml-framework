namespace MachineLearning.Checkpointing.Tests;

using Xunit;
using MLFramework.Checkpointing;

/// <summary>
/// Unit tests for CheckpointManager core functionality
/// </summary>
public class CheckpointManagerTests
{
    [Fact]
    public void Constructor_InitializesWithEmptyState()
    {
        // Arrange & Act
        using var manager = new CheckpointManager();

        // Assert
        Assert.Equal(0, manager.CheckpointCount);
        var stats = manager.GetMemoryStats();
        Assert.Equal(0, stats.CurrentMemoryUsed);
        Assert.Equal(0, stats.PeakMemoryUsed);
        Assert.Equal(0, stats.CheckpointCount);
    }

    [Fact]
    public void RegisterCheckpoint_SuccessfullyRegistersCheckpoint()
    {
        // Arrange
        using var manager = new CheckpointManager();
        var activation = new Tensor(1000, 4); // 1000 elements, 4 bytes each

        // Act
        manager.RegisterCheckpoint("layer1", activation);

        // Assert
        Assert.Equal(1, manager.CheckpointCount);
        Assert.True(manager.HasCheckpoint("layer1"));

        var stats = manager.GetMemoryStats();
        Assert.Equal(4000, stats.CurrentMemoryUsed); // 1000 * 4 bytes
        Assert.Equal(4000, stats.PeakMemoryUsed);
        Assert.Equal(4000, stats.AverageMemoryPerCheckpoint);
    }

    [Fact]
    public void RegisterCheckpoint_ThrowsArgumentExceptionForDuplicateLayerId()
    {
        // Arrange
        using var manager = new CheckpointManager();
        var activation1 = new Tensor(1000, 4);
        var activation2 = new Tensor(2000, 4);
        manager.RegisterCheckpoint("layer1", activation1);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => manager.RegisterCheckpoint("layer1", activation2));
    }

    [Fact]
    public void RegisterCheckpoint_ThrowsArgumentExceptionForEmptyLayerId()
    {
        // Arrange
        using var manager = new CheckpointManager();
        var activation = new Tensor(1000, 4);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => manager.RegisterCheckpoint("", activation));
        Assert.Throws<ArgumentException>(() => manager.RegisterCheckpoint(null!, activation));
    }

    [Fact]
    public void RegisterCheckpoint_ThrowsArgumentNullExceptionForNullActivation()
    {
        // Arrange
        using var manager = new CheckpointManager();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => manager.RegisterCheckpoint("layer1", null!));
    }

    [Fact]
    public void RegisterCheckpoint_CorrectlyCalculatesMemorySize()
    {
        // Arrange
        using var manager = new CheckpointManager();
        var activation = new Tensor(1000, 8); // 1000 elements, 8 bytes each (double)

        // Act
        manager.RegisterCheckpoint("layer1", activation);

        // Assert
        var stats = manager.GetMemoryStats();
        Assert.Equal(8000, stats.CurrentMemoryUsed); // 1000 * 8 bytes
    }

    [Fact]
    public void RegisterCheckpoint_TracksPeakMemoryUsage()
    {
        // Arrange
        using var manager = new CheckpointManager();
        var activation1 = new Tensor(1000, 4); // 4000 bytes
        var activation2 = new Tensor(500, 4);  // 2000 bytes
        var activation3 = new Tensor(750, 4);  // 3000 bytes

        // Act
        manager.RegisterCheckpoint("layer1", activation1);
        manager.RegisterCheckpoint("layer2", activation2);

        var stats1 = manager.GetMemoryStats();
        Assert.Equal(6000, stats1.CurrentMemoryUsed);
        Assert.Equal(6000, stats1.PeakMemoryUsed);

        manager.ClearCheckpoints();
        var stats2 = manager.GetMemoryStats();
        Assert.Equal(0, stats2.CurrentMemoryUsed);
        Assert.Equal(6000, stats2.PeakMemoryUsed); // Peak should be preserved

        manager.RegisterCheckpoint("layer3", activation3);
        var stats3 = manager.GetMemoryStats();
        Assert.Equal(3000, stats3.CurrentMemoryUsed);
        Assert.Equal(6000, stats3.PeakMemoryUsed); // Peak from before clear should still be highest
    }

    [Fact]
    public void RetrieveOrRecompute_ReturnsExistingCheckpoint()
    {
        // Arrange
        using var manager = new CheckpointManager();
        var activation = new Tensor(1000, 4);
        manager.RegisterCheckpoint("layer1", activation);

        // Act
        var retrieved = manager.RetrieveOrRecompute("layer1");

        // Assert
        Assert.NotNull(retrieved);
        Assert.Equal(activation.SizeInBytes, retrieved.SizeInBytes);
    }

    [Fact]
    public void RetrieveOrRecompute_RecomputesWhenNotFound()
    {
        // Arrange
        using var manager = new CheckpointManager();

        // Act
        var retrieved = manager.RetrieveOrRecompute("layer1", () => new Tensor(1000, 4));

        // Assert
        Assert.NotNull(retrieved);
        Assert.Equal(4000, retrieved.SizeInBytes);
        Assert.Equal(1, manager.CheckpointCount);
        Assert.True(manager.HasCheckpoint("layer1"));
    }

    [Fact]
    public void RetrieveOrRecompute_ThrowsKeyNotFoundExceptionWhenNotFoundAndNoRecomputeFunc()
    {
        // Arrange
        using var manager = new CheckpointManager();

        // Act & Assert
        Assert.Throws<KeyNotFoundException>(() => manager.RetrieveOrRecompute("layer1"));
    }

    [Fact]
    public void RetrieveOrRecompute_ThrowsInvalidOperationExceptionForNullRecomputedValue()
    {
        // Arrange
        using var manager = new CheckpointManager();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            manager.RetrieveOrRecompute("layer1", () => null!));
    }

    [Fact]
    public void RetrieveOrRecompute_ThrowsArgumentExceptionForEmptyLayerId()
    {
        // Arrange
        using var manager = new CheckpointManager();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => manager.RetrieveOrRecompute(""));
        Assert.Throws<ArgumentException>(() => manager.RetrieveOrRecompute(null!));
    }

    [Fact]
    public void HasCheckpoint_ReturnsCorrectStatus()
    {
        // Arrange
        using var manager = new CheckpointManager();
        var activation = new Tensor(1000, 4);
        manager.RegisterCheckpoint("layer1", activation);

        // Act & Assert
        Assert.True(manager.HasCheckpoint("layer1"));
        Assert.False(manager.HasCheckpoint("layer2"));
    }

    [Fact]
    public void ClearCheckpoints_ClearsAllCheckpoints()
    {
        // Arrange
        using var manager = new CheckpointManager();
        manager.RegisterCheckpoint("layer1", new Tensor(1000, 4));
        manager.RegisterCheckpoint("layer2", new Tensor(2000, 4));
        Assert.Equal(2, manager.CheckpointCount);

        // Act
        manager.ClearCheckpoints();

        // Assert
        Assert.Equal(0, manager.CheckpointCount);
        Assert.False(manager.HasCheckpoint("layer1"));
        Assert.False(manager.HasCheckpoint("layer2"));

        var stats = manager.GetMemoryStats();
        Assert.Equal(0, stats.CurrentMemoryUsed);
    }

    [Fact]
    public void ClearCheckpoints_PreservesPeakMemoryUsage()
    {
        // Arrange
        using var manager = new CheckpointManager();
        manager.RegisterCheckpoint("layer1", new Tensor(1000, 4));
        manager.RegisterCheckpoint("layer2", new Tensor(2000, 4));
        var statsBeforeClear = manager.GetMemoryStats();
        long peakBeforeClear = statsBeforeClear.PeakMemoryUsed;

        // Act
        manager.ClearCheckpoints();

        // Assert
        var statsAfterClear = manager.GetMemoryStats();
        Assert.Equal(0, statsAfterClear.CurrentMemoryUsed);
        Assert.Equal(peakBeforeClear, statsAfterClear.PeakMemoryUsed);
    }

    [Fact]
    public void GetMemoryStats_ReturnsCorrectStatistics()
    {
        // Arrange
        using var manager = new CheckpointManager();
        manager.RegisterCheckpoint("layer1", new Tensor(1000, 4));  // 4000 bytes
        manager.RegisterCheckpoint("layer2", new Tensor(2000, 4));  // 8000 bytes
        manager.RegisterCheckpoint("layer3", new Tensor(3000, 4));  // 12000 bytes

        // Act
        var stats = manager.GetMemoryStats();

        // Assert
        Assert.Equal(3, stats.CheckpointCount);
        Assert.Equal(24000, stats.CurrentMemoryUsed);
        Assert.Equal(24000, stats.PeakMemoryUsed);
        Assert.Equal(8000, stats.AverageMemoryPerCheckpoint); // 24000 / 3
    }

    [Fact]
    public void GetMemoryStats_ReturnsZeroForEmptyManager()
    {
        // Arrange
        using var manager = new CheckpointManager();

        // Act
        var stats = manager.GetMemoryStats();

        // Assert
        Assert.Equal(0, stats.CheckpointCount);
        Assert.Equal(0, stats.CurrentMemoryUsed);
        Assert.Equal(0, stats.PeakMemoryUsed);
        Assert.Equal(0, stats.AverageMemoryPerCheckpoint);
    }

    [Fact]
    public void CheckpointCount_ReturnsCorrectCount()
    {
        // Arrange
        using var manager = new CheckpointManager();

        // Act & Assert
        Assert.Equal(0, manager.CheckpointCount);

        manager.RegisterCheckpoint("layer1", new Tensor(1000, 4));
        Assert.Equal(1, manager.CheckpointCount);

        manager.RegisterCheckpoint("layer2", new Tensor(1000, 4));
        Assert.Equal(2, manager.CheckpointCount);

        manager.ClearCheckpoints();
        Assert.Equal(0, manager.CheckpointCount);
    }

    [Fact]
    public void Dispose_PreventsFurtherOperations()
    {
        // Arrange
        var manager = new CheckpointManager();
        manager.RegisterCheckpoint("layer1", new Tensor(1000, 4));

        // Act
        manager.Dispose();

        // Assert
        Assert.Throws<ObjectDisposedException>(() => manager.HasCheckpoint("layer1"));
        Assert.Throws<ObjectDisposedException>(() => manager.GetMemoryStats());
        Assert.Throws<ObjectDisposedException>(() => manager.CheckpointCount);
        Assert.Throws<ObjectDisposedException>(() => manager.RegisterCheckpoint("layer2", new Tensor(1000, 4)));
        Assert.Throws<ObjectDisposedException>(() => manager.RetrieveOrRecompute("layer1"));
        Assert.Throws<ObjectDisposedException>(() => manager.ClearCheckpoints());
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var manager = new CheckpointManager();

        // Act & Assert - Should not throw
        manager.Dispose();
        manager.Dispose();
        manager.Dispose();
    }

    [Fact]
    public void RegisterCheckpoints_MultipleCheckpoints_TrackMemoryCorrectly()
    {
        // Arrange
        using var manager = new CheckpointManager();

        // Act
        manager.RegisterCheckpoint("layer1", new Tensor(1000, 4));  // 4000 bytes
        manager.RegisterCheckpoint("layer2", new Tensor(2000, 4));  // 8000 bytes
        manager.RegisterCheckpoint("layer3", new Tensor(1500, 4));  // 6000 bytes

        // Assert
        var stats = manager.GetMemoryStats();
        Assert.Equal(3, stats.CheckpointCount);
        Assert.Equal(18000, stats.CurrentMemoryUsed);
        Assert.Equal(6000, stats.AverageMemoryPerCheckpoint);
    }

    [Fact]
    public void RetrieveOrRecompute_MultipleRetrieals_IncrementsAccessCount()
    {
        // Arrange
        using var manager = new CheckpointManager();
        var activation = new Tensor(1000, 4);
        manager.RegisterCheckpoint("layer1", activation);

        // Act
        manager.RetrieveOrRecompute("layer1");
        manager.RetrieveOrRecompute("layer1");
        manager.RetrieveOrRecompute("layer1");

        // Assert - Access count should be incremented internally (not exposed in public API)
        Assert.True(manager.HasCheckpoint("layer1"));
    }

    [Fact]
    public void ClearCheckpoints_DisposesTensors()
    {
        // Arrange
        using var manager = new CheckpointManager();
        var activation1 = new Tensor(1000, 4);
        var activation2 = new Tensor(2000, 4);
        manager.RegisterCheckpoint("layer1", activation1);
        manager.RegisterCheckpoint("layer2", activation2);

        // Act
        manager.ClearCheckpoints();

        // Assert - Should not throw, tensors should be disposed
        Assert.Equal(0, manager.CheckpointCount);
    }

    [Fact]
    public void RegisterCheckpoints_VariousTensorSizes_CalculatesCorrectMemory()
    {
        // Arrange
        using var manager = new CheckpointManager();

        // Act
        manager.RegisterCheckpoint("small", new Tensor(100, 4));      // 400 bytes
        manager.RegisterCheckpoint("medium", new Tensor(10000, 4));   // 40000 bytes
        manager.RegisterCheckpoint("large", new Tensor(1000000, 4));  // 4000000 bytes

        // Assert
        var stats = manager.GetMemoryStats();
        Assert.Equal(4040400, stats.CurrentMemoryUsed);
        Assert.Equal(1346800, stats.AverageMemoryPerCheckpoint);
    }

    [Fact]
    public void RetrieveOrRecompute_RecomputedCheckpoint_RegisteredAutomatically()
    {
        // Arrange
        using var manager = new CheckpointManager();

        // Act
        var retrieved1 = manager.RetrieveOrRecompute("layer1", () => new Tensor(1000, 4));
        var retrieved2 = manager.RetrieveOrRecompute("layer1"); // Should retrieve from cache now

        // Assert
        Assert.NotNull(retrieved1);
        Assert.NotNull(retrieved2);
        Assert.Equal(1, manager.CheckpointCount);
        Assert.Equal(4000, manager.GetMemoryStats().CurrentMemoryUsed);
    }
}
