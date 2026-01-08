namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for DistributedStateCollector
/// </summary>
public class DistributedStateCollectorTests
{
    [Fact]
    public void Constructor_WithValidCoordinator_CreatesInstance()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);

        // Act
        var collector = new DistributedStateCollector(coordinator);

        // Assert
        Assert.NotNull(collector);
    }

    [Fact]
    public void Constructor_WithNullCoordinator_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new DistributedStateCollector(null!));
    }

    [Fact]
    public void CollectLocalState_WithValidModel_ReturnsStateDict()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);
        var model = new MockStateful();
        var expectedTensor = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800);
        model.GetStateDict()["weight"] = expectedTensor;

        // Act
        var result = collector.CollectLocalState(model);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(expectedTensor, result["weight"]);
    }

    [Fact]
    public void CollectLocalState_WithNullModel_ThrowsException()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => collector.CollectLocalState(null!));
    }

    [Fact]
    public void CollectLocalOptimizerState_WithValidOptimizer_ReturnsStateDict()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);
        var optimizer = new MockStateful();
        var expectedTensor = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800);
        optimizer.GetStateDict()["momentum"] = expectedTensor;

        // Act
        var result = collector.CollectLocalOptimizerState(optimizer);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(expectedTensor, result["momentum"]);
    }

    [Fact]
    public void MergeStates_WithEmptyArray_ReturnsEmptyState()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);
        var states = Array.Empty<StateDict>();

        // Act
        var result = collector.MergeStates(states);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(0, result.Count);
    }

    [Fact]
    public void MergeStates_WithSingleState_ReturnsSameState()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);
        var tensor = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        var state = new StateDict { ["weight"] = tensor };
        var states = new[] { state };

        // Act
        var result = collector.MergeStates(states);

        // Assert
        Assert.Equal(1, result.Count);
        Assert.Equal(tensor, result["weight"]);
    }

    [Fact]
    public void MergeStates_WithMultipleDDPStates_MergesCorrectly()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);

        var tensor = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        var state1 = new StateDict { ["weight"] = tensor };
        var state2 = new StateDict { ["weight"] = tensor }; // Same tensor in all ranks (DDP)
        var state3 = new StateDict { ["weight"] = tensor };
        var states = new[] { state1, state2, state3 };

        // Act
        var result = collector.MergeStates(states);

        // Assert
        Assert.Equal(1, result.Count); // Only one copy kept (DDP)
        Assert.Equal(tensor, result["weight"]);
    }

    [Fact]
    public void MergeStates_WithMultipleFSDPStates_MergesCorrectly()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);

        var tensor1 = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        var tensor2 = new MockTensor(new long[] { 20 }, TensorDataType.Float32, 80);
        var tensor3 = new MockTensor(new long[] { 30 }, TensorDataType.Float32, 120);

        // Different tensors in each rank (FSDP - sharded)
        var state1 = new StateDict { ["weight_shard_0"] = tensor1 };
        var state2 = new StateDict { ["weight_shard_1"] = tensor2 };
        var state3 = new StateDict { ["weight_shard_2"] = tensor3 };
        var states = new[] { state1, state2, state3 };

        // Act
        var result = collector.MergeStates(states);

        // Assert
        Assert.Equal(3, result.Count); // All shards kept (FSDP)
        Assert.Equal(tensor1, result["weight_shard_0"]);
        Assert.Equal(tensor2, result["weight_shard_1"]);
        Assert.Equal(tensor3, result["weight_shard_2"]);
    }

    [Fact]
    public void MergeStates_WithNullStates_IgnoresNulls()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);

        var tensor = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        var state1 = new StateDict { ["weight"] = tensor };
        var states = new StateDict?[] { state1, null, null };

        // Act
        var result = collector.MergeStates(states!);

        // Assert
        Assert.Equal(1, result.Count);
        Assert.Equal(tensor, result["weight"]);
    }

    [Fact]
    public void MergeStates_WithConsistencyCheck_PassesWhenMatching()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);

        var tensor = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        var state1 = new StateDict { ["weight"] = tensor };
        var state2 = new StateDict { ["weight"] = tensor };
        var states = new[] { state1, state2 };

        // Act
        var result = collector.MergeStates(states, checkConsistency: true);

        // Assert
        Assert.Equal(1, result.Count);
    }

    [Fact]
    public void MergeStates_WithConsistencyCheck_ThrowsWhenShapeMismatch()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);

        var tensor1 = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        var tensor2 = new MockTensor(new long[] { 20 }, TensorDataType.Float32, 80); // Different shape
        var state1 = new StateDict { ["weight"] = tensor1 };
        var state2 = new StateDict { ["weight"] = tensor2 };
        var states = new[] { state1, state2 };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            collector.MergeStates(states, checkConsistency: true));
    }

    [Fact]
    public void MergeStates_WithConsistencyCheck_ThrowsWhenDataTypeMismatch()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(4, 0);
        var collector = new DistributedStateCollector(coordinator);

        var tensor1 = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        var tensor2 = new MockTensor(new long[] { 10 }, TensorDataType.Float64, 80); // Different dtype
        var state1 = new StateDict { ["weight"] = tensor1 };
        var state2 = new StateDict { ["weight"] = tensor2 };
        var states = new[] { state1, state2 };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            collector.MergeStates(states, checkConsistency: true));
    }
}
