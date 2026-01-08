using MLFramework.Checkpointing.Distributed;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Checkpointing.Distributed;

/// <summary>
/// Tests for PipelineParallelCheckpointManager
/// </summary>
public class PipelineParallelCheckpointManagerTests
{
    private MockDistributedCheckpointManager CreateMockManager()
    {
        return new MockDistributedCheckpointManager(0, 4);
    }

    [Fact]
    public void RegisterStageCheckpoint_StoresCheckpointCorrectly()
    {
        // Arrange
        var distributedManager = CreateMockManager();
        var manager = new PipelineParallelCheckpointManager(distributedManager, numStages: 4, currentStage: 1);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act
        manager.RegisterStageCheckpoint("layer1", tensor, isBoundary: false);

        // Assert
        var checkpoints = manager.GetStageCheckpoints(1);
        Assert.Single(checkpoints);
        Assert.Equal(tensor, checkpoints[0]);
    }

    [Fact]
    public void RegisterStageCheckpoint_BoundaryCheckpoint_StoresInDistributedManager()
    {
        // Arrange
        var distributedManager = CreateMockManager();
        var manager = new PipelineParallelCheckpointManager(distributedManager, numStages: 4, currentStage: 1);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act
        manager.RegisterStageCheckpoint("layer1", tensor, isBoundary: true);

        // Assert
        Assert.True(distributedManager.LocalManager.HasCheckpoint("layer1"));
    }

    [Fact]
    public void GetStageCheckpoints_ReturnsCorrectCheckpoints()
    {
        // Arrange
        var distributedManager = CreateMockManager();
        var manager = new PipelineParallelCheckpointManager(distributedManager, numStages: 4, currentStage: 1);
        var tensor1 = Tensor.Zeros(new int[] { 10, 10 });
        var tensor2 = Tensor.Ones(new int[] { 5, 5 });

        // Act
        manager.RegisterStageCheckpoint("layer1", tensor1, isBoundary: false);
        manager.RegisterStageCheckpoint("layer2", tensor2, isBoundary: false);

        // Assert
        var checkpoints = manager.GetStageCheckpoints(1);
        Assert.Equal(2, checkpoints.Count);
        Assert.Contains(tensor1, checkpoints);
        Assert.Contains(tensor2, checkpoints);
    }

    [Fact]
    public void GetStageCheckpoints_WrongStage_ReturnsEmpty()
    {
        // Arrange
        var distributedManager = CreateMockManager();
        var manager = new PipelineParallelCheckpointManager(distributedManager, numStages: 4, currentStage: 1);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act
        manager.RegisterStageCheckpoint("layer1", tensor, isBoundary: false);

        // Assert
        var checkpoints = manager.GetStageCheckpoints(2);
        Assert.Empty(checkpoints);
    }

    [Fact]
    public void GetNextStageBoundaries_ReturnsBoundaryCheckpoints()
    {
        // Arrange
        var distributedManager = CreateMockManager();
        var manager = new PipelineParallelCheckpointManager(distributedManager, numStages: 4, currentStage: 1);
        var tensor1 = Tensor.Zeros(new int[] { 10, 10 });
        var tensor2 = Tensor.Ones(new int[] { 5, 5 });

        // Act
        manager.RegisterStageCheckpoint("layer1", tensor1, isBoundary: true);
        manager.RegisterStageCheckpoint("layer2", tensor2, isBoundary: false);

        // Assert
        var boundaries = manager.GetNextStageBoundaries();
        Assert.Single(boundaries);
        Assert.Equal(tensor1, boundaries[0]);
    }

    [Fact]
    public void ClearStageCheckpoints_RemovesAllCheckpoints()
    {
        // Arrange
        var distributedManager = CreateMockManager();
        var manager = new PipelineParallelCheckpointManager(distributedManager, numStages: 4, currentStage: 1);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        manager.RegisterStageCheckpoint("layer1", tensor, isBoundary: false);
        manager.RegisterStageCheckpoint("layer2", tensor, isBoundary: false);

        // Act
        manager.ClearStageCheckpoints();

        // Assert
        var checkpoints = manager.GetStageCheckpoints(1);
        Assert.Empty(checkpoints);
    }

    [Fact]
    public void Dispose_ReleasesResources()
    {
        // Arrange
        var distributedManager = CreateMockManager();
        var manager = new PipelineParallelCheckpointManager(distributedManager, numStages: 4, currentStage: 1);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        manager.RegisterStageCheckpoint("layer1", tensor, isBoundary: false);

        // Act
        manager.Dispose();

        // Assert - Should be able to dispose without throwing
        Assert.True(true);
    }

    [Fact]
    public void Constructor_NullCheckpointManager_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
        {
            new PipelineParallelCheckpointManager(null!, 4, 1);
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
