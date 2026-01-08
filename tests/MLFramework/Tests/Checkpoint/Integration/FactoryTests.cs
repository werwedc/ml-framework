namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for CheckpointIntegrationHelperFactory
/// </summary>
public class CheckpointIntegrationHelperFactoryTests
{
    [Fact]
    public void CreateHelper_FSDPModel_ReturnsFSDPCheckpointHelper()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.FullyShardedDataParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);

        // Act
        var helper = CheckpointIntegrationHelperFactory.CreateHelper(model, coordinator);

        // Assert
        Assert.NotNull(helper);
        Assert.IsType<FSDPCheckpointHelper>(helper);
    }

    [Fact]
    public void CreateHelper_DDPModel_ReturnsDDPCheckpointHelper()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.DataParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);

        // Act
        var helper = CheckpointIntegrationHelperFactory.CreateHelper(model, coordinator);

        // Assert
        Assert.NotNull(helper);
        Assert.IsType<DDPCheckpointHelper>(helper);
    }

    [Fact]
    public void CreateHelper_TPModel_ReturnsTensorParallelCheckpointHelper()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.TensorParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);

        // Act
        var helper = CheckpointIntegrationHelperFactory.CreateHelper(model, coordinator);

        // Assert
        Assert.NotNull(helper);
        Assert.IsType<TensorParallelCheckpointHelper>(helper);
    }

    [Fact]
    public void CreateHelper_UnsupportedStrategy_ThrowsArgumentException()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.PipelineParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => CheckpointIntegrationHelperFactory.CreateHelper(model, coordinator));
    }
}
