using MLFramework.Checkpointing.Distributed;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Checkpointing.Distributed;

/// <summary>
/// Tests for ModelParallelCheckpointStrategy
/// </summary>
public class ModelParallelCheckpointStrategyTests
{
    private MockDistributedCheckpointManager CreateMockManager(int rank)
    {
        return new MockDistributedCheckpointManager(rank, 4);
    }

    [Fact]
    public void ShouldCheckpoint_TensorParallel_CheckpointsOnlyOnFirstRank()
    {
        // Arrange
        var manager = CreateMockManager(0);
        var strategy = new ModelParallelCheckpointStrategy(manager, tensorParallelSize: 2);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act & Assert - Rank 0 should checkpoint
        Assert.True(strategy.ShouldCheckpoint("layer1", tensor, isTensorParallel: true));

        // Arrange - Rank 1
        var manager2 = CreateMockManager(1);
        var strategy2 = new ModelParallelCheckpointStrategy(manager2, tensorParallelSize: 2);

        // Act & Assert - Rank 1 should not checkpoint
        Assert.False(strategy2.ShouldCheckpoint("layer1", tensor, isTensorParallel: true));
    }

    [Fact]
    public void ShouldCheckpoint_NonTensorParallel_AlwaysCheckpoints()
    {
        // Arrange
        var manager = CreateMockManager(1);
        var strategy = new ModelParallelCheckpointStrategy(manager, tensorParallelSize: 2);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act & Assert
        Assert.True(strategy.ShouldCheckpoint("layer1", tensor, isTensorParallel: false));
    }

    [Fact]
    public void GetCheckpointRank_ReturnsCorrectRank()
    {
        // Arrange
        var manager = CreateMockManager(0);
        var strategy = new ModelParallelCheckpointStrategy(
            manager,
            tensorParallelSize: 2,
            layerToRankMap: new Dictionary<string, int> { { "layer1", 2 }, { "layer2", 3 } }
        );

        // Act & Assert
        Assert.Equal(2, strategy.GetCheckpointRank("layer1"));
        Assert.Equal(3, strategy.GetCheckpointRank("layer2"));
        Assert.Equal(0, strategy.GetCheckpointRank("unknown_layer"));
    }

    [Fact]
    public void RegisterLayerRank_StoresMappingCorrectly()
    {
        // Arrange
        var manager = CreateMockManager(0);
        var strategy = new ModelParallelCheckpointStrategy(manager, tensorParallelSize: 2);

        // Act
        strategy.RegisterLayerRank("layer1", 5);
        strategy.RegisterLayerRank("layer2", 7);

        // Assert
        Assert.Equal(5, strategy.GetCheckpointRank("layer1"));
        Assert.Equal(7, strategy.GetCheckpointRank("layer2"));
    }

    [Fact]
    public void Constructor_NullCheckpointManager_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
        {
            new ModelParallelCheckpointStrategy(null!, 2);
        });
    }

    private class MockDistributedCheckpointManager : DistributedCheckpointManager
    {
        public MockDistributedCheckpointManager(int rank, int worldSize)
            : base(rank, worldSize, null)
        {
        }
    }
}
