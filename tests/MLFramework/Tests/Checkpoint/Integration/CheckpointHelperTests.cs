namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for FSDPCheckpointHelper
/// </summary>
public class FSDPCheckpointHelperTests
{
    [Fact]
    public void Constructor_ValidInputs_CreatesHelper()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.FullyShardedDataParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);

        // Act & Assert
        var helper = new FSDPCheckpointHelper(model, coordinator);
        Assert.NotNull(helper);
    }

    [Fact]
    public void Constructor_NonFSDPModel_ThrowsArgumentException()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.DataParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new FSDPCheckpointHelper(model, coordinator));
    }

    [Fact]
    public void CollectLocalShard_EmptyStateDict_ReturnsShardData()
    {
        // Arrange
        var stateDict = new StateDict();
        var model = new MockDistributedModel(DistributedStrategy.FullyShardedDataParallel, 4, 1, stateDict);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var helper = new FSDPCheckpointHelper(model, coordinator);

        // Act
        var shardData = helper.CollectLocalShard();

        // Assert
        Assert.NotNull(shardData);
        Assert.NotNull(shardData.Data);
        Assert.NotNull(shardData.TensorInfo);
        Assert.Empty(shardData.TensorInfo);
    }

    [Fact]
    public void CollectLocalShard_WithTensors_ReturnsCorrectMetadata()
    {
        // Arrange
        var stateDict = new StateDict
        {
            ["layer1.weight"] = new MockTensor(new long[] { 128, 256 }, TensorDataType.Float32, 128 * 256 * 4),
            ["layer1.bias"] = new MockTensor(new long[] { 128 }, TensorDataType.Float32, 128 * 4)
        };
        var model = new MockDistributedModel(DistributedStrategy.FullyShardedDataParallel, 4, 1, stateDict);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var helper = new FSDPCheckpointHelper(model, coordinator);

        // Act
        var shardData = helper.CollectLocalShard();

        // Assert
        Assert.Equal(2, shardData.TensorInfo.Count);
        Assert.Contains(shardData.TensorInfo, t => t.Name == "layer1.weight");
        Assert.Contains(shardData.TensorInfo, t => t.Name == "layer1.bias");
    }

    [Fact]
    public void GetShardingMetadata_ReturnsCorrectMetadata()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.FullyShardedDataParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var helper = new FSDPCheckpointHelper(model, coordinator);

        // Act
        var metadata = helper.GetShardingMetadata();

        // Assert
        Assert.Equal("fsdp", metadata.Strategy);
        Assert.Equal(4, metadata.ShardCount);
        Assert.Equal("fp16", metadata.Precision);
        Assert.True(metadata.StrategySpecificInfo.ContainsKey("zero_stage"));
        Assert.True(metadata.StrategySpecificInfo.ContainsKey("sharding_strategy"));
    }
}

/// <summary>
/// Tests for DDPCheckpointHelper
/// </summary>
public class DDPCheckpointHelperTests
{
    [Fact]
    public void Constructor_ValidInputs_CreatesHelper()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.DataParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);

        // Act & Assert
        var helper = new DDPCheckpointHelper(model, coordinator);
        Assert.NotNull(helper);
    }

    [Fact]
    public void Constructor_NonDDPModel_ThrowsArgumentException()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.FullyShardedDataParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new DDPCheckpointHelper(model, coordinator));
    }

    [Fact]
    public void CollectFullState_EmptyStateDict_ReturnsShardData()
    {
        // Arrange
        var stateDict = new StateDict();
        var model = new MockDistributedModel(DistributedStrategy.DataParallel, 4, 1, stateDict);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var helper = new DDPCheckpointHelper(model, coordinator);

        // Act
        var shardData = helper.CollectFullState();

        // Assert
        Assert.NotNull(shardData);
        Assert.NotNull(shardData.Data);
        Assert.NotNull(shardData.TensorInfo);
        Assert.Empty(shardData.TensorInfo);
    }

    [Fact]
    public void CollectFullState_WithTensors_ReturnsCorrectMetadata()
    {
        // Arrange
        var stateDict = new StateDict
        {
            ["layer1.weight"] = new MockTensor(new long[] { 128, 256 }, TensorDataType.Float32, 128 * 256 * 4),
            ["layer1.bias"] = new MockTensor(new long[] { 128 }, TensorDataType.Float32, 128 * 4)
        };
        var model = new MockDistributedModel(DistributedStrategy.DataParallel, 4, 1, stateDict);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var helper = new DDPCheckpointHelper(model, coordinator);

        // Act
        var shardData = helper.CollectFullState();

        // Assert
        Assert.Equal(2, shardData.TensorInfo.Count);
        Assert.Contains(shardData.TensorInfo, t => t.Name == "layer1.weight");
        Assert.Contains(shardData.TensorInfo, t => t.Name == "layer1.bias");
    }

    [Fact]
    public void GetShardingMetadata_ReturnsCorrectMetadata()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.DataParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var helper = new DDPCheckpointHelper(model, coordinator);

        // Act
        var metadata = helper.GetShardingMetadata();

        // Assert
        Assert.Equal("ddp", metadata.Strategy);
        Assert.Equal(4, metadata.ShardCount);
        Assert.Equal("fp16", metadata.Precision);
        Assert.True(metadata.StrategySpecificInfo.ContainsKey("bucket_size"));
    }
}

/// <summary>
/// Tests for TensorParallelCheckpointHelper
/// </summary>
public class TensorParallelCheckpointHelperTests
{
    [Fact]
    public void Constructor_ValidInputs_CreatesHelper()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.TensorParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);

        // Act & Assert
        var helper = new TensorParallelCheckpointHelper(model, coordinator);
        Assert.NotNull(helper);
    }

    [Fact]
    public void Constructor_NonTPModel_ThrowsArgumentException()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.DataParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new TensorParallelCheckpointHelper(model, coordinator));
    }

    [Fact]
    public void CollectTPShard_EmptyStateDict_ReturnsShardData()
    {
        // Arrange
        var stateDict = new StateDict();
        var model = new MockDistributedModel(DistributedStrategy.TensorParallel, 4, 1, stateDict);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var helper = new TensorParallelCheckpointHelper(model, coordinator);

        // Act
        var shardData = helper.CollectTPShard();

        // Assert
        Assert.NotNull(shardData);
        Assert.NotNull(shardData.Data);
        Assert.NotNull(shardData.TensorInfo);
        Assert.Empty(shardData.TensorInfo);
    }

    [Fact]
    public void CollectTPShard_WithTensors_ReturnsCorrectMetadata()
    {
        // Arrange
        var stateDict = new StateDict
        {
            ["layer1.weight"] = new MockTensor(new long[] { 128, 64 }, TensorDataType.Float32, 128 * 64 * 4),
            ["layer1.bias"] = new MockTensor(new long[] { 128 }, TensorDataType.Float32, 128 * 4)
        };
        var model = new MockDistributedModel(DistributedStrategy.TensorParallel, 4, 1, stateDict);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var helper = new TensorParallelCheckpointHelper(model, coordinator);

        // Act
        var shardData = helper.CollectTPShard();

        // Assert
        Assert.Equal(2, shardData.TensorInfo.Count);
        Assert.Contains(shardData.TensorInfo, t => t.Name == "layer1.weight");
        Assert.Contains(shardData.TensorInfo, t => t.Name == "layer1.bias");
    }

    [Fact]
    public void GetShardingMetadata_ReturnsCorrectMetadata()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.TensorParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var helper = new TensorParallelCheckpointHelper(model, coordinator);

        // Act
        var metadata = helper.GetShardingMetadata();

        // Assert
        Assert.Equal("tensor_parallel", metadata.Strategy);
        Assert.Equal(4, metadata.ShardCount);
        Assert.Equal("bf16", metadata.Precision);
        Assert.True(metadata.StrategySpecificInfo.ContainsKey("tp_degree"));
        Assert.True(metadata.StrategySpecificInfo.ContainsKey("axis"));
    }
}
